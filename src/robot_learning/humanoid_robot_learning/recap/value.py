"""RECAP value function: a distributional value over normalised return, trained on Monte Carlo returns.

    python -m humanoid_robot_learning.recap.value --dataset datasets/recap/drawer_stow_r1 --scene drawer_stow \\
        --out outputs/recap/value_r1

pi*0.6 (arXiv 2511.14759): the value predicts a distribution over B bins of the return normalised to [-1, 0], trained
with cross-entropy against the episode's empirical (Monte Carlo) return, no bootstrapping. Train it on everything
collected so far (demos, every round's rollouts and corrections: merge them first with humanoid-merge-takes); it is
used offline only, to label advantages (recap.label), so it reads privileged sim state (recap/data.py), not images.

The target return is spread over its two neighbouring bins (two-hot), so the expected value is exact. A held-out set
of whole episodes (--val_frac) picks the training step (best held-out cross-entropy) and measures it against a
constant baseline (the training targets' histogram).
Writes <out>/value.pt (weights + config, load with ValueFunction.load) and <out>/metrics.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from humanoid_robot_learning.recap.data import load_frames

BINS = 201
HIDDEN = (256, 256)


@dataclass
class ValueConfig:
    scene: str
    robot: str
    fps: int
    max_s: float                # time limit the returns are normalised by
    num_inputs: int
    bins: int = BINS
    hidden: tuple[int, ...] = HIDDEN
    mean: list[float] = field(default_factory=list)     # input standardisation (training frames)
    std: list[float] = field(default_factory=list)


def make_mlp(num_inputs: int, hidden: tuple[int, ...], bins: int) -> nn.Sequential:
    layers, width = [], num_inputs
    for h in hidden:
        layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.ReLU()]
        width = h
    return nn.Sequential(*layers, nn.Linear(width, bins))


class ValueFunction(nn.Module):
    """Logits over `bins` returns in [-1, 0]; value() is their expectation."""

    def __init__(self, config: ValueConfig):
        super().__init__()
        self.config = config
        self.net = make_mlp(config.num_inputs, tuple(config.hidden), config.bins)
        self.register_buffer("support", torch.linspace(-1.0, 0.0, config.bins))
        self.register_buffer("mean", torch.tensor(config.mean or [0.0] * config.num_inputs))
        self.register_buffer("std", torch.tensor(config.std or [1.0] * config.num_inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net((inputs - self.mean) / self.std)

    @torch.no_grad()
    def value(self, inputs: np.ndarray, batch: int = 8192) -> np.ndarray:
        """Expected normalised return of each row of `inputs` (recap/data.py's value inputs)."""
        self.eval()
        out = [torch.softmax(self(torch.as_tensor(inputs[i:i + batch])), -1) @ self.support
               for i in range(0, len(inputs), batch)]
        return torch.cat(out).numpy()

    def save(self, path: Path) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ValueFunction":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ValueConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()


def two_hot(returns: torch.Tensor, bins: int) -> torch.Tensor:
    """(N,) returns in [-1, 0] -> (N, bins) weights on the two nearest bins (their mean is the return)."""
    pos = (returns.clamp(-1.0, 0.0) + 1.0) * (bins - 1)
    lo = pos.floor().long().clamp(max=bins - 2)
    frac = pos - lo
    target = torch.zeros(len(returns), bins)
    target.scatter_(1, lo[:, None], (1 - frac)[:, None])
    target.scatter_add_(1, lo[:, None] + 1, frac[:, None])
    return target


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits, -1)).sum(-1).mean()


def split_episodes(episodes: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """Boolean mask of held-out frames: whole episodes, at least one when val_frac > 0."""
    unique = np.unique(episodes)
    n_val = max(1, round(val_frac * len(unique))) if val_frac > 0 else 0
    held = np.random.default_rng(seed).choice(unique, n_val, replace=False)
    return np.isin(episodes, held)


def train(model: ValueFunction, x: torch.Tensor, target: torch.Tensor, val: tuple | None, args) -> None:
    """AdamW on the cross-entropy; with held-out `val` (inputs, targets), keeps the weights of the best val CE: a few
    hundred episodes are memorised quickly, and an overconfident value distribution has a poor held-out CE."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gen = torch.Generator().manual_seed(args.seed)
    every, best = max(1, args.steps // 20), (math.inf, None)
    for step in range(args.steps):
        model.train()
        idx = torch.randint(len(x), (min(args.batch_size, len(x)),), generator=gen)
        loss = cross_entropy(model(x[idx]), target[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % every == 0 or step == args.steps - 1:
            val_ce = held_out_ce(model, *val) if val else math.nan
            if val and val_ce < best[0]:
                best = (val_ce, copy.deepcopy(model.state_dict()))
            print(f"[VALUE] step {step}: train CE {loss.item():.4f}, val CE {val_ce:.4f}", flush=True)
    if best[1] is not None:
        model.load_state_dict(best[1])


@torch.no_grad()
def held_out_ce(model: ValueFunction, x: torch.Tensor, target: torch.Tensor) -> float:
    model.eval()
    return cross_entropy(model(x), target).item()


@torch.no_grad()
def evaluate(model: ValueFunction, x: torch.Tensor, target: torch.Tensor, returns: torch.Tensor,
             prior: torch.Tensor) -> dict:
    """Held-out cross-entropy of the model and of the constant `prior` histogram, and the value's MAE."""
    model.eval()
    value = torch.softmax(model(x), -1) @ model.support
    baseline = -(target * torch.log(prior.clamp_min(1e-8))).sum(-1).mean()
    return {"val_ce": cross_entropy(model(x), target).item(), "val_ce_constant": baseline.item(),
            "val_mae": (value - returns).abs().mean().item(), "val_mae_constant": (returns.mean() - returns).abs().mean().item()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__.split("\n\n", 2)[2],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="LeRobot dataset with its recap/ sidecar (all data so far, merged)")
    p.add_argument("--scene", required=True, help="humanoid_mujoco_scenes scene of the snapshots")
    p.add_argument("--robot", default="8dof", help="sim robot of the snapshots")
    p.add_argument("--out", required=True, help="output folder (value.pt, metrics.json)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--max_s", type=float, default=40.0, help="rollout time limit (s): the return's normaliser")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.1, help="fraction of episodes held out")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    frames = load_frames(args.dataset, args.scene, args.robot, args.fps, args.max_s)
    val = split_episodes(frames.episode, args.val_frac, args.seed)
    x, r = torch.as_tensor(frames.inputs), torch.as_tensor(frames.returns, dtype=torch.float32)
    std = x[~val].std(0)
    config = ValueConfig(args.scene, args.robot, args.fps, args.max_s, x.shape[1], mean=x[~val].mean(0).tolist(),
                         std=torch.where(std > 1e-6, std, torch.ones_like(std)).tolist())
    model, target = ValueFunction(config), two_hot(r, BINS)
    print(f"[VALUE] {len(x)} frames, {len(np.unique(frames.episode))} episodes, {x.shape[1]} inputs; "
          f"{int(val.sum())} frames held out", flush=True)
    train(model, x[~val], target[~val], (x[val], target[val]) if val.any() else None, args)
    metrics = {"frames": len(x), "train_frames": int((~val).sum())}
    if val.any():
        metrics |= evaluate(model, x[val], target[val], r[val], target[~val].mean(0))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "value.pt")
    (out / "metrics.json").write_text(json.dumps({**metrics, "args": vars(args)}, indent=2))
    print(f"[VALUE] {json.dumps(metrics)}\n[VALUE] saved {out / 'value.pt'}", flush=True)


if __name__ == "__main__":
    main()
