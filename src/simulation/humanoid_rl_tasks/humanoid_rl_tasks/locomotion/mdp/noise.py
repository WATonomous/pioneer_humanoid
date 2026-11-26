"""Observation corruption for the ray-cast height scan (the "lidar" input).

Adds what stock uniform/Gaussian noise cannot: per-beam bias, beam dropout, and beams dead for a whole
episode. It is a :class:`NoiseModelCfg`, so it corrupts one observation term only, and the dropout rates
ramp in over ``ramp_steps`` so a warm-started checkpoint adapts instead of meeting full corruption at once.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, NoiseModel, NoiseModelCfg


class LidarNoiseModel(NoiseModel):
    """Per-beam bias (resampled per episode), Gaussian range noise, then beam dropout.

    A beam drops if the per-step mask fires (``dropout_prob``) or it is dead for the episode
    (``dropout_episode_prob``); dropped beams read ``dropout_value``.
    """

    def __init__(self, noise_model_cfg: LidarNoiseCfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        # copy the parameters, don't hold the config: the manager stores this instance back into the
        # config (``noise.func``), and the reference cycle would make config validation recurse forever
        self._range_std = float(noise_model_cfg.range_std)
        self._bias_std = float(noise_model_cfg.per_ray_bias_std)
        self._drop_prob = float(noise_model_cfg.dropout_prob)
        self._dead_prob = float(noise_model_cfg.dropout_episode_prob)
        self._fill = float(noise_model_cfg.dropout_value)
        self._ramp_steps = int(noise_model_cfg.ramp_steps)
        self._noise_model_cfg = None  # drop the back-reference (see above)
        # allocated lazily: the beam count is known at the first batch
        self._bias: torch.Tensor | None = None
        self._dead: torch.Tensor | None = None
        # policy steps seen; drives the ramp (not reset per episode)
        self._steps = 0

    def _ensure_state(self, data: torch.Tensor) -> None:
        """Allocate the fixed-pattern bias and dead-beam mask on the first call."""
        if self._bias is not None and self._bias.shape == data.shape:
            return
        self._bias = torch.zeros_like(data)
        self._dead = torch.zeros(data.shape, dtype=torch.bool, device=data.device)
        self._resample_bias(None)
        self._resample_dead(None)

    def _ramp(self) -> float:
        """Fraction of the full dropout rate currently in effect, in [0, 1]."""
        if self._ramp_steps <= 0:
            return 1.0
        return min(1.0, self._steps / self._ramp_steps)

    def _env_ids(self, env_ids: Sequence[int] | None) -> slice | Sequence[int]:
        return slice(None) if env_ids is None else env_ids

    def _resample_bias(self, env_ids: Sequence[int] | None) -> None:
        ids = self._env_ids(env_ids)
        self._bias[ids] = self._bias_std * torch.randn_like(self._bias[ids])

    def _resample_dead(self, env_ids: Sequence[int] | None) -> None:
        ids = self._env_ids(env_ids)
        prob = self._dead_prob * self._ramp()
        shape = self._dead[ids].shape
        self._dead[ids] = torch.rand(shape, device=self._dead.device) < prob

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Redraw the per-episode corruption for the reset environments."""
        if self._bias is None:
            return
        self._resample_bias(env_ids)
        self._resample_dead(env_ids)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        self._ensure_state(data)
        ramp = self._ramp()
        self._steps += 1

        out = data + self._bias
        if self._range_std > 0.0:
            out = out + self._range_std * torch.randn_like(out)

        if self._drop_prob > 0.0 or self._dead_prob > 0.0:
            prob = self._drop_prob * ramp + self._dead.to(out.dtype)
            dropped = torch.rand_like(out) < prob
            out = torch.where(dropped, torch.full_like(out, self._fill), out)
        return out


@configclass
class LidarNoiseCfg(NoiseModelCfg):
    """Configuration for :class:`LidarNoiseModel` (see that class for details)."""

    class_type: type = LidarNoiseModel

    # the model does its own work; the base class's single-term noise is unused
    noise_cfg: NoiseCfg | None = None

    range_std: float = 0.05
    """Standard deviation of the Gaussian range error, in metres."""
    per_ray_bias_std: float = 0.01
    """Standard deviation of the fixed per-beam offset, in metres (resampled per episode)."""
    dropout_prob: float = 0.02
    """Per-beam, per-step probability of losing the return (about 4 beams of 187)."""
    dropout_episode_prob: float = 0.02
    """Per-beam probability of being dead for a whole episode (about 4 beams of 187)."""
    dropout_value: float = 0.0
    """Value written for a dropped beam (zero, the same "no usable return" encoding ``mdp.height_scan`` uses for NaN)."""
    ramp_steps: int = 12_000
    """Policy steps over which the dropout rates ramp from zero to full (~500 iterations at 24 steps each)."""
