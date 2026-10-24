"""Episode-level dataset over existing LeRobot roots (read in place), with optional episode selection.

Episodes are identified by their raw source (``raw``, ``seed`` from ``phase_index.json``) or, for
datasets without one, by (root, episode index), so a correction converted twice (e.g. with
``--repeat 2``) is used once. A selection plan restricts training to listed identities::

    {"identities": [[raw_or_root, seed_or_index], ...], "baseline_datasets": [0, 1]}

Each item is one frame with its future ``chunk_size`` actions (padding past the episode end is
masked). State/action normalization is computed from the selected frames only, unless an
existing ``normalization.json`` is given (continuations must keep the checkpoint's statistics).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from humanoid_robot_learning.phase_sampling import PhaseBatchSampler, phase_episodes, pool_for

CAMERAS = ("ego", "wrist_left")


def catalogue(roots, chunk_size, fps):
    """Load each root and list its episodes with identity, frame range and phase ranges."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets, records = [], []
    for dataset_id, root in enumerate(map(Path, roots)):
        data = LeRobotDataset("humanoid/pioneer_peg_insert", root=root, video_backend="pyav",
                              delta_timestamps={"action": [i / fps for i in range(chunk_size)]})
        if data.fps != fps:
            raise ValueError(f"{root} is {data.fps} Hz, expected {fps}")
        datasets.append(data)
        sources = phase_episodes(root)
        for index in range(data.meta.total_episodes):
            episode, source = data.meta.episodes[index], sources[index] if sources else None
            identity = [source["raw"], str(source["seed"])] if source else [str(root), str(index)]
            records.append({"dataset": dataset_id, "episode": index, "identity": identity,
                            "start": int(episode["dataset_from_index"]), "end": int(episode["dataset_to_index"]),
                            "condition": source.get("condition", "") if source else "",
                            "phases": source["phase_ranges"] if source else []})
    return datasets, records


def select(records, plan):
    unique = {}
    for record in records:
        unique.setdefault(tuple(record["identity"]), record)
    if plan is None:
        return list(unique.values())
    missing = [i for i in plan["identities"] if tuple(i) not in unique]
    if missing:
        raise ValueError(f"{len(missing)} selected episodes are not in the datasets, e.g. {missing[0]}")
    return sorted((unique[tuple(i)] for i in plan["identities"]), key=lambda r: (r["dataset"], r["episode"]))


def phase_at(episode, absolute):
    """Phase ranges are dataset-relative frame indices, like ``dataset_from_index``."""
    return next((run["phase"] for run in episode["phases"] if run["start"] <= absolute < run["end"]), None)


class ChunkDataset(Dataset):
    def __init__(self, roots, chunk_size=50, fps=25, selection=None, phase_config=None, normalization=None, baseline=None):
        self.roots = [str(r) for r in roots]
        self.datasets, records = catalogue(roots, chunk_size, fps)
        plan = json.loads(Path(selection).read_text()) if selection else None
        self.episodes = select(records, plan)
        if baseline is None:
            baseline = plan.get("baseline_datasets", []) if plan else range(max(1, len(roots) - 1))
        baseline = set(baseline)
        self.config = phase_config
        self.index, self.metric, self.pools = [], [], {name: [] for name in (phase_config or {"quotas": {}})["quotas"]}
        metric_phases = set(phase_config.get("metric_phases", [])) if phase_config else set()
        for episode in self.episodes:
            for absolute in range(episode["start"], episode["end"]):
                phase = phase_at(episode, absolute)
                self.metric.append(phase in metric_phases)
                if phase_config:
                    pool = "baseline" if episode["dataset"] in baseline else pool_for(phase or "", episode["condition"], phase_config["rules"])
                    if pool:
                        self.pools[pool].append(len(self.index))
                self.index.append((episode["dataset"], absolute))
        self.normalization = json.loads(Path(normalization).read_text()) if normalization else self.statistics()
        self.norm = {k: (torch.tensor(v["mean"]), torch.tensor(v["std"])) for k, v in self.normalization.items()}

    def statistics(self):
        """Mean/std of state and action over the selected frames (scalar columns only, no image decoding)."""
        sums = {k: [0.0, 0.0, 0] for k in ["observation.state", "action"]}
        for dataset_id, data in enumerate(self.datasets):
            indices = [i for d, i in self.index if d == dataset_id]
            table = data.hf_dataset.select(indices).with_format("numpy", columns=list(sums))
            for begin in range(0, len(indices), 8192):
                batch = table[begin:begin + 8192]
                for key, acc in sums.items():
                    values = np.asarray(batch[key], dtype=np.float64)
                    acc[0], acc[1], acc[2] = acc[0] + values.sum(0), acc[1] + (values * values).sum(0), acc[2] + len(values)
        stats = {}
        for key, (total, squares, n) in sums.items():
            mean = total / n
            stats[key] = {"mean": mean.tolist(), "std": np.sqrt(np.maximum(squares / n - mean * mean, 1e-8)).tolist()}
        return stats

    def describe(self):
        return {"roots": self.roots, "episodes": len(self.episodes), "frames": len(self),
                "pool_sizes": {k: len(v) for k, v in self.pools.items()}}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        dataset_id, absolute = self.index[index]
        item = self.datasets[dataset_id][absolute]

        def normalize(key):
            return (item[key] - self.norm[key][0]) / self.norm[key][1]
        return {"images": torch.stack([item[f"observation.images.{name}"] for name in CAMERAS]),
                "state": normalize("observation.state"), "action": normalize("action"),
                "padding": item["action_is_pad"], "metric": torch.tensor(self.metric[index])}

    def sampler(self, seed, batch_size):
        """Phase-quota batches if a phase config was given, else ordinary shuffled batches."""
        if self.config:
            return PhaseBatchSampler(self.pools, len(self), self.config["quotas"], seed)
        return ShuffledBatches(len(self), batch_size, seed)


class ShuffledBatches:
    """Each epoch a seed+epoch permutation, so a resumed run can skip exactly the batches already seen."""

    def __init__(self, num_frames, batch_size, seed):
        self.num_frames, self.batch_size, self.seed, self.epoch = num_frames, batch_size, seed, 0

    def __len__(self):
        return -(-self.num_frames // self.batch_size)

    def __iter__(self):
        order = torch.randperm(self.num_frames, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        self.epoch += 1
        for start in range(0, self.num_frames, self.batch_size):
            yield order[start:start + self.batch_size]
