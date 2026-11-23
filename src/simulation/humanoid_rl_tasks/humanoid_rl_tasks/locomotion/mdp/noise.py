"""Observation corruption for the ray-cast terrain scan (the "lidar" input).

The height scanner is the only range-sensing observation in these tasks: a
17 x 11 grid of rays (10 cm spacing, 1.6 m x 1.0 m footprint) fired at the
terrain, turned into ``sensor_z - hit_z - 0.5`` metres by ``mdp.height_scan``.
Isaac Lab's stock noise terms can only add uniform or Gaussian error, which
models range noise but not the failure modes that matter when this input comes
off a real sensor instead of a physics ray cast:

* beams that return nothing at all (range dropout), and
* a fixed subset of beams that stays dead for a whole episode (dirty lens,
  occlusion, a bad channel).

Both are implemented here as a :class:`NoiseModelCfg`, so they corrupt exactly
one observation term and cannot leak into the joint/IMU channels. The manager
applies the model before clipping, and calls :meth:`LidarNoiseModel.reset` on
every environment reset, which is where the per-episode dead beams are drawn.

The corruption rates ramp in over ``ramp_steps`` policy steps. A run that warm
starts from a clean-observation checkpoint therefore sees the noise grow in
instead of getting the full corruption on its first rollout.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, NoiseModel, NoiseModelCfg


class LidarNoiseModel(NoiseModel):
    """Additive range noise, a fixed per-beam bias, and beam dropout.

    Order of operations on each call:

    1. add the fixed per-beam bias (resampled per episode in :meth:`reset`),
    2. add zero-mean Gaussian range noise,
    3. overwrite dropped beams with ``dropout_value``.

    A beam is dropped if either the transient mask fires this step
    (``dropout_prob`` per beam per step) or it is one of the beams marked
    dead for the whole episode (``dropout_episode_prob``).
    """

    def __init__(self, noise_model_cfg: LidarNoiseCfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        # Copy the parameters instead of holding the config: the observation
        # manager stores this instance back into the term's config
        # (``noise.func``). Keeping the config here would close a reference
        # cycle, and Isaac Lab validates configs by walking object attributes
        # recursively, so the next time that config is validated (e.g. building
        # a second env from it) the walk never terminates.
        self._range_std = float(noise_model_cfg.range_std)
        self._bias_std = float(noise_model_cfg.per_ray_bias_std)
        self._drop_prob = float(noise_model_cfg.dropout_prob)
        self._dead_prob = float(noise_model_cfg.dropout_episode_prob)
        self._fill = float(noise_model_cfg.dropout_value)
        self._ramp_steps = int(noise_model_cfg.ramp_steps)
        self._noise_model_cfg = None  # drop the back-reference (see above)
        # allocated lazily: the number of beams is only known once the
        # observation term produces its first batch.
        self._bias: torch.Tensor | None = None
        self._dead: torch.Tensor | None = None
        # policy steps seen by this model; drives the ramp. Deliberately not
        # reset per episode -- environment resets must not restart the ramp.
        self._steps = 0

    # ------------------------------------------------------------------ setup
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
            return  # the first call allocates and draws; nothing to redraw yet
        self._resample_bias(env_ids)
        self._resample_dead(env_ids)

    # ------------------------------------------------------------------- noise
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

    # The custom model below does all of its own work; the base class's
    # single-term noise is unused, and the manager never calls it.
    noise_cfg: NoiseCfg | None = None

    range_std: float = 0.05
    """Standard deviation of the Gaussian range error, in metres.

    The scan is metres of terrain height below the base on a 10 cm grid, so
    5 cm is half a cell: large enough that a policy cannot memorise the map,
    small enough that stair edges stay visible.
    """
    per_ray_bias_std: float = 0.01
    """Standard deviation of the fixed per-beam offset, in metres (resampled per episode)."""
    dropout_prob: float = 0.02
    """Per-beam, per-step probability of losing the return (about 4 beams of 187)."""
    dropout_episode_prob: float = 0.02
    """Per-beam probability of being dead for a whole episode (about 4 beams of 187)."""
    dropout_value: float = 0.0
    """Value written for a dropped beam.

    ``mdp.height_scan`` already replaces NaN hits with zero, so zero is the
    established "no usable return" encoding here. It also sits mid-range rather
    than reading as a cliff, so a dropped beam costs the policy information
    without telling it that a wall is in front of it.
    """
    ramp_steps: int = 12_000
    """Policy steps over which the dropout rates ramp from zero to full.

    One iteration is ``num_steps_per_env`` = 24 policy steps, so the default
    ramps in over ~500 iterations.
    """
