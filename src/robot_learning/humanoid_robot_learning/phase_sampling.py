"""Phase-balanced batches and per-phase action-error reporting for mixed demo/correction datasets.

Raw collections record a ``phase`` per frame; ``raw_to_lerobot`` stores the phase ranges in
``<dataset>/phase_index.json``. A phase config (e.g. ``config/peg_insert_phases.json``) maps
phases to sampling pools with ordered first-match rules and gives fixed per-batch quotas::

    {"quotas": {"baseline": 3, "precision": 2, "recovery": 2, "completion": 1},
     "rules": [{"pool": "recovery", "phases": ["align"], "condition_prefix": "tilt"}, ...],
     "metric_phases": ["align", "recover_realign"]}

Frames of *baseline* datasets all go to the ``baseline`` pool; frames of the other datasets go to
the first matching rule's pool and are skipped otherwise (e.g. grasp-securing holds). Each batch
draws exactly ``quotas[pool]`` frames from every pool, with replacement, so a small correction
set is seen at a fixed rate regardless of its size. Action chunks themselves are unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import BatchSampler, SequentialSampler


def load_config(path):
    config = json.loads(Path(path).read_text())
    if "baseline" not in config["quotas"] or any(int(n) <= 0 for n in config["quotas"].values()):
        raise ValueError("phase quotas need a baseline pool and positive counts")
    return config


def phase_episodes(root):
    """Episode records (``phase_ranges`` relative to the dataset, ``condition``) or [] if unlabelled."""
    path = Path(root) / "phase_index.json"
    return json.loads(path.read_text())["episodes"] if path.exists() else []


def pool_for(phase, condition, rules):
    for rule in rules:
        if phase in rule["phases"] and condition.startswith(rule.get("condition_prefix", "")):
            return rule["pool"]
    return None


def frame_labels(episodes):
    """Yield (frame index within dataset, phase, condition) for every labelled frame."""
    for episode in episodes:
        for run in episode["phase_ranges"]:
            for index in range(run["start"], run["end"]):
                yield index, run["phase"], episode.get("condition", "")


def build_pools(sources, config, baseline):
    """``sources``: [(root, global offset, frame count)]; ``baseline``: set of source indices."""
    pools = {name: [] for name in config["quotas"]}
    for i, (root, offset, size) in enumerate(sources):
        if i in baseline:
            pools["baseline"].extend(range(offset, offset + size))
            continue
        for index, phase, condition in frame_labels(phase_episodes(root)):
            if not 0 <= index < size:
                raise ValueError(f"phase range outside {root}")
            pool = pool_for(phase, condition, config["rules"])
            if pool is not None:
                pools[pool].append(offset + index)
    return pools


def metric_flags(sources, total, phases):
    """Bool per global frame: labelled with one of ``phases`` (unlabelled frames are never inferred)."""
    flags = torch.zeros(total, dtype=torch.bool)
    for root, offset, size in sources:
        for index, phase, _ in frame_labels(phase_episodes(root)):
            if phase in phases:
                flags[offset + index] = True
    return flags


class PhaseBatchSampler(BatchSampler):
    """Exact per-pool quotas in every batch; an epoch has ceil(num_frames / batch) batches."""

    def __init__(self, pools, num_frames, quotas, seed=1000):
        batch = sum(quotas.values())
        super().__init__(SequentialSampler(range(num_frames)), batch, False)
        self.quotas = dict(quotas)
        self.pools = {name: torch.tensor(pools[name], dtype=torch.long) for name in self.quotas}
        empty = [name for name, pool in self.pools.items() if len(pool) == 0]
        if empty:
            raise ValueError(f"phase quotas need nonempty pools: {empty}")
        if any(int(p.min()) < 0 or int(p.max()) >= num_frames for p in self.pools.values()):
            raise ValueError("phase index lies outside the training dataset")
        self.num_batches = math.ceil(num_frames / batch)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(len(self)):
            batch = []
            for name, count in self.quotas.items():
                pool = self.pools[name]
                batch.extend(pool[torch.randint(len(pool), (count,), generator=generator)].tolist())
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[i] for i in order]


class PhaseErrorReporter:
    """Windowed mean |prediction - target| over chunk targets labelled with metric phases.

    Reporting only: it never changes gradients. ``target`` counts future actions whose own frame is
    labelled; ``example`` counts whole chunks whose starting frame is labelled. Padding excluded.
    """

    def __init__(self, flags, output, start_step=0, interval=500, note=""):
        self.flags, self.output, self.step, self.interval, self.note = flags, Path(output), start_step, interval, note
        self.sums, self.pending = [0.0, 0, 0.0, 0], 0

    def observe(self, prediction, target, padding, indices):
        flags = self.flags.to(prediction.device)
        positions = indices.reshape(-1, 1).to(prediction.device) + torch.arange(prediction.shape[1], device=prediction.device)
        valid = ~padding.bool()
        per_target = flags[positions.clamp(max=len(flags) - 1)] & (positions < len(flags)) & valid
        per_example = flags[indices.reshape(-1).to(prediction.device)][:, None] & valid
        error = (prediction.detach() - target).abs()
        for slot, mask in enumerate([per_target, per_example]):
            self.sums[2 * slot] += float((error * mask[:, :, None]).sum())
            self.sums[2 * slot + 1] += int(mask.sum()) * error.shape[-1]
        self.step += 1
        self.pending += 1
        if self.step % self.interval == 0:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        a, n, b, m = self.sums
        row = {"step": self.step, "window_updates": self.pending, "metric_target_l1": a / n if n else None,
               "metric_example_chunk_l1": b / m if m else None, "components": [n, m], "note": self.note}
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print("PHASE_METRICS", json.dumps(row), flush=True)
        self.sums, self.pending = [0.0, 0, 0.0, 0], 0
