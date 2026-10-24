"""Train or continue the flow transformer on existing LeRobot datasets (no conversion or image copies).

  # from scratch: ImageNet vision, 1k warmup + cosine, phase-quota batches of 8
  python -m humanoid_il.flow_transformer.train --root outputs/train/flow --dataset datasets/pioneer_peg_insert \\
      --dataset datasets/corr_r1 --selection outputs/selections/half.json \\
      --phase-config src/robot_learning/config/peg_insert_phases.json --steps 100000
  # branch a finished checkpoint onto new data at a constant low LR (keeps its normalization)
  python -m humanoid_il.flow_transformer.train --root outputs/train/flow_r2 --dataset ... \\
      --resume-from outputs/train/flow/checkpoints/100000.pt --schedule constant --lr 1e-5 --vision-lr 1e-6 --steps 150000

Rerunning the same command resumes ``<root>/checkpoints/latest.pt`` (model, EMA, AdamW, RNG and
the exact sampler position). ``--stop-at`` saves and exits early without changing the schedule.
Checkpoints are ``<step>.pt`` with EMA weights used for evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ChunkDataset
from .policy import FlowConfig, FlowPolicy, flow_loss

EMA_DECAY = 0.999
GRAD_CLIP = 1.0


def learning_rates(step, args):
    """(vision, transformer) LR for this update."""
    if args.schedule == "constant":
        return args.vision_lr, args.lr
    warm = min((step + 1) / args.warmup, 1.0)
    cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * max(0, step - args.warmup) / max(1, args.steps - args.warmup)))
    return args.vision_lr * warm * cosine, args.lr * warm * cosine


class ResumeBatches:
    """Continue the sampler at the saved global batch instead of restarting the epoch."""

    def __init__(self, sampler, consumed):
        self.sampler, self.skip = sampler, consumed % len(sampler)
        self.sampler.epoch = consumed // len(sampler)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        skip, self.skip = self.skip, 0
        for i, batch in enumerate(self.sampler):
            if i >= skip:
                yield batch


def build_model(args, dataset_normalization):
    config = FlowConfig(refinement_steps=args.refinement_steps)
    model = FlowPolicy(config, pretrained_vision=args.resume_from is None).to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    vision = list(model.vision.parameters())
    ids = {id(p) for p in vision}
    head = [p for p in model.parameters() if id(p) not in ids]
    optimizer = torch.optim.AdamW([{"params": vision, "lr": args.vision_lr}, {"params": head, "lr": args.lr}], weight_decay=1e-4)
    return config, model, ema, optimizer


def load_state(path, config, model, ema, optimizer, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["config"] != config.dictionary():
        raise ValueError(f"{path} has a different architecture: {saved['config']}")
    model.load_state_dict(saved["model"])
    ema.load_state_dict(saved["ema"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    if device == "cuda" and saved.get("cuda_rng"):
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved


def save_checkpoint(root, payload):
    directory = root / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{payload['step']:06d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    latest = directory / "latest.link"
    latest.unlink(missing_ok=True)
    latest.hardlink_to(path)  # no duplicate payload on disk
    latest.replace(directory / "latest.pt")
    print("FLOW_CHECKPOINT", payload["step"], flush=True)


def resume(args, config, model, ema, optimizer):
    """Returns (step, sampler_start, normalization); sampler_start is where this run's data began."""
    latest = args.root / "checkpoints/latest.pt"
    if latest.exists():
        saved = load_state(latest, config, model, ema, optimizer, args.device)
        if saved["training_steps"] != args.steps or saved["schedule"] != schedule(args):
            raise ValueError("refusing to change the target or schedule of an existing run; branch with --resume-from")
        return saved["step"], saved.get("sampler_start", 0), saved["normalization"]
    if args.resume_from:
        saved = load_state(args.resume_from, config, model, ema, optimizer, args.device)
        return saved["step"], saved["step"], saved["normalization"]
    return 0, 0, None


def schedule(args):
    out = {"mode": "warmup_cosine" if args.schedule == "cosine" else "constant_continuation", "steps": args.steps}
    if args.schedule == "constant":
        out.update(main_lr=args.lr, vision_lr=args.vision_lr)
    return out


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    phase_config = json.loads(args.phase_config.read_text()) if args.phase_config else None
    config, model, ema, optimizer = build_model(args, None)
    step, sampler_start, normalization = resume(args, config, model, ema, optimizer)
    if normalization is not None:
        args.root.mkdir(parents=True, exist_ok=True)
        (args.root / "normalization.json").write_text(json.dumps(normalization, indent=2))
    dataset = ChunkDataset(args.dataset, config.chunk_size, args.fps, args.selection, phase_config,
                           args.root / "normalization.json" if normalization is not None else None)
    print("FLOW_DATA", json.dumps(dataset.describe()), flush=True)
    write_run_config(args, config, dataset)
    sampler = ResumeBatches(dataset.sampler(args.seed, args.batch_size), step - sampler_start)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=args.device == "cuda",
                        persistent_workers=args.workers > 0)
    target = min(args.steps, step + 1) if args.smoke else (args.stop_at or args.steps)
    model.train()
    log = {"loss": [], "metric_sum": 0.0, "metric_count": 0, "clock": time.monotonic()}
    while step < target:
        for batch in loader:
            if step >= target:
                break
            step = update(args, model, ema, optimizer, batch, step, log)
            if not args.smoke and (step % args.save_freq == 0 or step == target):
                save_checkpoint(args.root, {"config": config.dictionary(), "normalization": dataset.normalization,
                                            "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                                            "step": step, "training_steps": args.steps, "schedule": schedule(args),
                                            "sampler_start": sampler_start, "rng": torch.get_rng_state(),
                                            "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                                            "python_rng": random.getstate(), "numpy_rng": np.random.get_state()})
    print("FLOW_SMOKE_OK" if args.smoke else f"FLOW_TRAINED_TO {step}", flush=True)


def update(args, model, ema, optimizer, batch, step, log):
    batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
    for group, lr in zip(optimizer.param_groups, learning_rates(step, args)):
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = flow_loss(model, batch["images"], batch["state"], batch["action"], batch["padding"], metric_mask=batch["metric"])
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite flow loss; refusing update")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not torch.isfinite(grad):
        raise RuntimeError("nonfinite gradients; refusing update")
    optimizer.step()
    with torch.no_grad():
        for e, m in zip(ema.parameters(), model.parameters()):
            e.lerp_(m, 1 - EMA_DECAY)
    step += 1
    log["loss"].append(float(loss))
    log["metric_sum"] += metrics["metric_sum"]
    log["metric_count"] += metrics["metric_count"]
    if step % 100 == 0 or args.smoke:
        row = {"step": step, "velocity_mse": float(np.mean(log["loss"])), "lr": optimizer.param_groups[1]["lr"], "grad_norm": float(grad),
               "metric_phase_velocity_mse": log["metric_sum"] / log["metric_count"] if log["metric_count"] else None,
               "elapsed_seconds": round(time.monotonic() - log["clock"], 2)}
        print("FLOW_TRAIN", json.dumps(row), flush=True)
        with (args.root / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        log.update(loss=[], metric_sum=0.0, metric_count=0)
    return step


def write_run_config(args, config, dataset):
    args.root.mkdir(parents=True, exist_ok=True)
    run = {"model": config.dictionary(), "data": dataset.describe(), "selection": str(args.selection) if args.selection else None,
           "phase_config": str(args.phase_config) if args.phase_config else None, "schedule": schedule(args),
           "lr": args.lr, "vision_lr": args.vision_lr, "warmup": args.warmup if args.schedule == "cosine" else 0,
           "batch_size": args.batch_size, "ema_decay": EMA_DECAY, "seed": args.seed,
           "resume_from": str(args.resume_from) if args.resume_from else None}
    (args.root / "config.json").write_text(json.dumps(run, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="run directory (checkpoints, metrics, config)")
    p.add_argument("--dataset", type=Path, action="append", required=True, help="LeRobot root; repeat; index 0 is the base demos")
    p.add_argument("--selection", type=Path, help="episode selection plan (identities, baseline_datasets)")
    p.add_argument("--phase-config", type=Path, help="phase-quota batches; batch size = quota total")
    p.add_argument("--steps", type=int, default=100000, help="final global step of this schedule")
    p.add_argument("--stop-at", type=int, help="save and exit at this step (resume later)")
    p.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--lr", type=float, default=1e-4, help="transformer/heads LR")
    p.add_argument("--vision-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--refinement-steps", type=int, default=10, help="Euler steps stored as the default for sampling")
    p.add_argument("--resume-from", type=Path, help="branch from this checkpoint (weights, optimizer, step, normalization)")
    p.add_argument("--save-freq", type=int, default=10000)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true", help="one real-data update, no checkpoint")
    args = p.parse_args()
    if args.phase_config and sum(json.loads(args.phase_config.read_text())["quotas"].values()) != args.batch_size:
        p.error("--batch-size must equal the phase quota total")
    if args.stop_at is not None and not 0 < args.stop_at <= args.steps:
        p.error("--stop-at must be in (0, --steps]")
    train(args)


if __name__ == "__main__":
    main()
