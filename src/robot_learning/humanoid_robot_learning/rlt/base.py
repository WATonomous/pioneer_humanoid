"""The frozen base policy (a Pioneer-1 checkpoint): batched context tokens and proposed action chunks for N envs."""
from __future__ import annotations

import numpy as np
import torch

from humanoid_robot_learning.sim.evaluate import load_policy, training_task


class FrozenBase:
    def __init__(self, path: str, device: str = "cuda", task: str | None = None):
        from lerobot.utils.control_utils import prepare_observation_for_inference

        self.policy, self.pre, self.post = load_policy(path, device)
        if self.policy.config.type != "pioneer1":
            raise ValueError(f"RLT here wraps a pioneer1 checkpoint, got {self.policy.config.type!r}")
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        self.task = task or training_task(path)
        if self.task is None:
            raise ValueError("no task text: pass task=")
        self._prepare = prepare_observation_for_inference
        self.cameras = {k.removeprefix("observation.images."): tuple(ft.shape[1:])
                        for k, ft in self.policy.config.image_features.items()}
        self.action_dim = int(self.policy.config.output_features["action"].shape[0])
        self.horizon = self.policy.config.horizon

    def _batch(self, observations: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        """Raw env observations -> one preprocessed (normalized, tokenized) batch on the device."""
        items = [self.pre(self._prepare(dict(o), self.device, self.task, None)) for o in observations]
        return {k: torch.cat([it[k] for it in items]) if isinstance(items[0][k], torch.Tensor) else items[0][k]
                for k in items[0]}

    @torch.no_grad()
    def __call__(self, observations: list[dict[str, np.ndarray]], sample: bool = True, prefix: torch.Tensor | None = None):
        """-> context (B, N, D), key mask (B, N) or None, normalized state (B, S), normalized chunk (B, horizon, A) or None.
        ``prefix`` (B, d, A): normalized actions held fixed at the chunk's start (real-time chunking, as Pioneer-1 runs
        natively); without it, frequent replanning breaks the base's motion (its README: 12/100 replanning every 8)."""
        pol = self.policy
        batch = self._batch(observations)
        stacked = pol._stack_images(batch)
        stacked = {k: v.unsqueeze(1) if k in ("observation.images", "observation.state") else v for k, v in stacked.items()}
        context, mask = pol.encode(stacked)
        chunk = pol.sample(context, prefix=prefix, mask=mask) if sample else None
        return context, mask, batch["observation.state"], chunk

    @torch.no_grad()
    def unnormalize(self, chunk: torch.Tensor) -> np.ndarray:
        """(B, C, A) normalized actions -> (B, C, A) joint targets + gripper, numpy."""
        b, c, a = chunk.shape
        out = self.post(chunk.reshape(b * c, a))
        return out.reshape(b, c, a).cpu().numpy()
