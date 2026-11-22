"""A frozen Pioneer-1 whose flow sampler starts from a given noise (DSRL: RL picks the noise, the policy stays fixed).

For a batch of observations: ``features`` (pooled context tokens + normalised state) for the RL networks, and
``chunk(noise)``: Pioneer-1's plan when the Euler integration starts from that noise instead of a random draw. With
``noise ~ N(0, I)`` this is exactly the base policy.
"""
from __future__ import annotations

import numpy as np
import torch
from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.control_utils import prepare_observation_for_inference

from humanoid_robot_learning.sim.evaluate import load_policy


class FrozenPioneer1:
    def __init__(self, path: str, device: str = "cuda"):
        self.policy, self.pre, self.post = load_policy(path, device)
        self.policy.requires_grad_(False)
        cfg = self.policy.config
        if cfg.n_obs_steps != 1:
            raise ValueError("DSRL wrapper needs n_obs_steps = 1")
        self.cfg, self.device = cfg, torch.device(device)
        self.cameras = {k.removeprefix("observation.images."): tuple(ft.shape[1:]) for k, ft in cfg.image_features.items()}
        self.horizon, self.n_exec, self.prefix_len = cfg.horizon, cfg.n_action_steps, cfg.replan_prefix
        self.action_dim = int(cfg.output_features["action"].shape[0])
        self.noise_shape = (self.horizon, self.action_dim)

    @torch.no_grad()
    def encode(self, observations: list[dict], tasks: list[str]):
        """Raw env observations -> (context tokens, key mask, RL features (B, D + state))."""
        rows = [self.pre(prepare_observation_for_inference(dict(o), self.device, task=t, robot_type=None))
                for o, t in zip(observations, tasks)]
        batch = {k: torch.cat([r[k] for r in rows]) for k in (OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)}
        images = torch.stack([torch.cat([r[k] for r in rows]) for k in self.policy.cameras], dim=1)   # (B, cams, 3, H, W)
        history = {OBS_STATE: batch[OBS_STATE][:, None], OBS_IMAGES: images[:, None],
                   OBS_LANGUAGE_TOKENS: batch[OBS_LANGUAGE_TOKENS], OBS_LANGUAGE_ATTENTION_MASK: batch[OBS_LANGUAGE_ATTENTION_MASK]}
        context, mask = self.policy.encode(history)
        if mask is None:
            pooled = context.mean(1)
        else:
            m = mask.float()[..., None]
            pooled = (context * m).sum(1) / m.sum(1)
        return context, mask, torch.cat([pooled, batch[OBS_STATE]], dim=-1)

    @torch.no_grad()
    def chunk(self, context, mask, noise: torch.Tensor, prefix: torch.Tensor | None = None) -> torch.Tensor:
        """Pioneer-1's sampler (same Euler steps and clean RTC prefix) from ``noise`` (B, horizon, A): normalised plan."""
        p = self.policy
        steps, (b, horizon) = self.cfg.flow_steps, noise.shape[:2]
        x = noise.clone()
        d = 0 if prefix is None else prefix.shape[1]
        if d:
            x[:, :d] = prefix
        for i in range(steps):
            t = torch.full((b, horizon), i / steps, device=x.device)
            t[:, :d] = 1.0
            v = p.velocity(x, t, context, mask)
            v[:, :d] = 0.0
            x = x + v / steps
        return x

    @torch.no_grad()
    def unnormalise(self, plan: torch.Tensor) -> np.ndarray:
        """(B, T, A) normalised -> (B, T, A) env actions."""
        b, t, a = plan.shape
        return self.post(plan.reshape(b * t, a)).reshape(b, t, a).cpu().numpy()
