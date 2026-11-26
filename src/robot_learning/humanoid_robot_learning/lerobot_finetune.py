"""Pinned-LeRobot trainer entry point for continuing a policy on demos + accumulated corrections.

Wraps ``lerobot.scripts.lerobot_train`` (same CLI) and adds, via flags parsed before LeRobot's:

* ``--extra-dataset ROOT`` (repeatable): concatenate more LeRobot datasets with the same schema.
  Each dataset builds its own action chunks, so chunks never cross datasets/episodes, and nothing
  is copied (useful for reading prior rounds in place on NAS).
* Saved normalization is kept on resume. Upstream replaces checkpoint statistics with the new
  mixture's, shifting inputs under already-trained weights.
* ``--phase-config JSON``: fixed per-batch phase quotas (see ``phase_sampling``). The main
  dataset and every extra dataset except the last are baseline, unless ``--baseline-datasets``
  says otherwise.
* ``--phase-metrics``: also log ACT action error on frames in the config's ``metric_phases``.

  python -m humanoid_robot_learning.lerobot_finetune \\
      --config_path=<ckpt>/pretrained_model/train_config.json --resume=true \\
      --dataset.root=datasets/pioneer_peg_insert --extra-dataset=datasets/corrections_r1 \\
      --phase-config=src/robot_learning/config/peg_insert_phases.json --steps=174000 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch


class IndexedConcat(torch.utils.data.ConcatDataset):
    """ConcatDataset exposing LeRobot attributes, with ``index`` replaced by the global index."""

    def __init__(self, datasets):
        super().__init__(datasets)
        base = datasets[0]
        for other in datasets[1:]:
            if other.meta.features != base.meta.features or other.fps != base.fps:
                raise ValueError("concatenated datasets differ in schema or fps")
        self.meta, self.fps = base.meta, base.fps
        self.num_frames, self.num_episodes = len(self), sum(d.num_episodes for d in datasets)
        self.episodes = None

    def __getitem__(self, index):
        item = dict(super().__getitem__(index))
        item["index"] = torch.tensor(index, dtype=torch.int64)
        return item


def parse_own_flags(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--extra-dataset", type=Path, action="append", default=[])
    p.add_argument("--phase-config", type=Path)
    p.add_argument("--baseline-datasets", type=int, nargs="*", help="source indices (0 = --dataset.root) fully in the baseline pool")
    p.add_argument("--phase-metrics", action="store_true")
    return p.parse_known_args(argv)


def make_dataset_factory(original, own, state):
    def make_dataset(cfg):
        datasets = [original(cfg)]
        for root in own.extra_dataset:
            extra = deepcopy(cfg)
            extra.dataset.root, extra.dataset.episodes = str(root), None
            datasets.append(original(extra))
        dataset = IndexedConcat(datasets)
        offsets = [sum(len(d) for d in datasets[:i]) for i in range(len(datasets))]
        state["sources"] = [(root, offset, len(d)) for root, offset, d in zip([cfg.dataset.root, *own.extra_dataset], offsets, datasets)]
        state["output_dir"], state["checkpoint"], state["seed"] = Path(cfg.output_dir), cfg.checkpoint_path, cfg.seed or 1000
        return dataset
    return make_dataset


def phase_loader_class(original_loader, own, state):
    from .phase_sampling import PhaseBatchSampler, build_pools, load_config

    class PhaseLoader(original_loader):
        def __init__(self, dataset, *args, **kwargs):
            if not isinstance(dataset, IndexedConcat):
                return super().__init__(dataset, *args, **kwargs)
            config = load_config(own.phase_config)
            if args or kwargs.get("batch_size") != sum(config["quotas"].values()):
                raise ValueError(f"--batch_size must equal the phase quota total {sum(config['quotas'].values())}")
            sources = state["sources"]
            baseline = set(own.baseline_datasets if own.baseline_datasets is not None else range(max(1, len(sources) - 1)))
            pools = build_pools(sources, config, baseline)
            report = {"quotas": config["quotas"], "pool_sizes": {k: len(v) for k, v in pools.items()}, "baseline_sources": sorted(baseline)}
            print("PHASE_BALANCE", json.dumps(report), flush=True)
            state["output_dir"].mkdir(parents=True, exist_ok=True)
            (state["output_dir"] / "phase_balance.json").write_text(json.dumps(report, indent=2))
            for key in ("batch_size", "shuffle", "sampler", "drop_last"):
                kwargs.pop(key, None)
            kwargs["batch_sampler"] = PhaseBatchSampler(pools, len(dataset), config["quotas"], state["seed"])
            super().__init__(dataset, *args, **kwargs)
    return PhaseLoader


def metric_forward(original_forward, own, state):
    from .phase_sampling import PhaseErrorReporter, load_config, metric_flags

    def forward(policy, batch):
        if "reporter" not in state:
            sources = state["sources"]
            flags = metric_flags(sources, sources[-1][1] + sources[-1][2], set(load_config(own.phase_config)["metric_phases"]))
            start = 0
            if state["checkpoint"]:
                start = json.loads((Path(state["checkpoint"]) / "training_state/training_step.json").read_text())["step"]
            state["reporter"] = PhaseErrorReporter(flags, state["output_dir"] / "phase_metrics.jsonl", start,
                                                   note="normalized ACT action MAE; training VAE posterior")
        captured = []
        hook = policy.model.register_forward_hook(lambda module, inputs, output: captured.append(output[0].detach()))
        try:
            result = original_forward(policy, batch)
        finally:
            hook.remove()
        if policy.training:
            state["reporter"].observe(captured[0], batch["action"], batch["action_is_pad"], batch["index"])
        return result
    return forward


def keep_saved_normalization(original):
    def make_processors(*args, **kwargs):
        # Keep device/feature overrides, but let saved state files supply the original means/stds.
        for pipeline, step in [("preprocessor_overrides", "normalizer_processor"), ("postprocessor_overrides", "unnormalizer_processor")]:
            overrides = deepcopy(kwargs.get(pipeline, {}))
            overrides.get(step, {}).pop("stats", None)
            kwargs[pipeline] = overrides
        return original(*args, **kwargs)
    return make_processors


def main():
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.scripts import lerobot_train

    own, rest = parse_own_flags(sys.argv[1:])
    if own.phase_metrics and own.phase_config is None:
        raise SystemExit("--phase-metrics needs --phase-config (for metric_phases)")
    sys.argv = [sys.argv[0], *rest]
    state = {}
    patches = [(lerobot_train, "make_pre_post_processors", keep_saved_normalization(lerobot_train.make_pre_post_processors)),
               (lerobot_train, "make_dataset", make_dataset_factory(lerobot_train.make_dataset, own, state))]
    if own.phase_config:
        patches.append((torch.utils.data, "DataLoader", phase_loader_class(torch.utils.data.DataLoader, own, state)))
    if own.phase_metrics:
        patches.append((ACTPolicy, "forward", metric_forward(ACTPolicy.forward, own, state)))
    originals = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    for obj, name, value in patches:
        setattr(obj, name, value)
    try:
        lerobot_train.main()
    finally:
        if "reporter" in state:
            state["reporter"].flush()
        for obj, name, value in originals:
            setattr(obj, name, value)


if __name__ == "__main__":
    main()
