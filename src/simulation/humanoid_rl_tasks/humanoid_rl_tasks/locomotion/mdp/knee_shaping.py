"""Small, task-local knee shaping without changing physical target limits."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

__all__ = ["knee_target_overshoot_smooth_l1", "knee_swing_flexion_deficit_l2"]


def _joint_names(asset, asset_cfg: SceneEntityCfg) -> list[str]:
    ids = asset_cfg.joint_ids
    if isinstance(ids, slice):
        return list(asset.joint_names[ids])
    return [asset.joint_names[index] for index in ids]


class knee_target_overshoot_smooth_l1(ManagerTermBase):
    """Cost raw POSITION targets beyond the action term's existing knee bounds.

    Unlike a cost on already-clipped targets, this still changes when a policy
    is far into saturation. The tail is linear and uncapped, avoiding a huge
    quadratic cost without discarding the signal to return toward the bounds.
    Scale, offset and bounds come from the actual action term, not URDF limits.
    Resolve the action mapping and validate bounds once at manager setup, after
    Isaac Lab resolves SceneEntityCfg. Read live scale/offset on every call so
    action randomization remains effective. No action object or tensor is
    modified, and the reward owns its cache independently of other terms.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._beta = cfg.params.get("beta", 0.1)
        if not math.isfinite(self._beta) or self._beta <= 0.0:
            raise ValueError("Knee overshoot beta must be finite and positive.")
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        names = _joint_names(asset, asset_cfg)
        if len(names) != 2 or set(names) != {"Knee_L", "Knee_R"}:
            raise ValueError("Knee overshoot must select exactly Knee_L and Knee_R.")
        self._action_term = env.action_manager.get_term(cfg.params.get("action_name", "joint_pos"))
        term = self._action_term
        if getattr(term, "_asset", asset) is not asset:
            raise ValueError("Knee action term and selected joints must use the same asset.")
        action_names = list(term._joint_names)
        if len(action_names) != len(set(action_names)) or any(name not in action_names for name in names):
            raise ValueError("Knee action mapping is missing or ambiguous.")
        self._action_indices = [action_names.index(name) for name in names]
        raw = term.raw_actions
        if raw.ndim != 2 or raw.shape[1] != len(action_names):
            raise ValueError("Unexpected raw knee action shape.")
        clip = getattr(term, "_clip", None)
        if clip is None:
            raise ValueError("Knee overshoot requires configured processed target bounds.")
        try:
            bounds = torch.broadcast_to(clip, (*raw.shape, 2))[:, self._action_indices]
        except RuntimeError as error:
            raise ValueError("Knee target bounds have an incompatible shape.") from error
        if not torch.isfinite(bounds).all() or (bounds[..., 0] >= bounds[..., 1]).any():
            raise ValueError("Knee target bounds must be finite and strictly ordered.")
        self._bounds = bounds

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        action_name: str = "joint_pos",
        beta: float = 0.1,
    ) -> torch.Tensor:
        # Isaac passes the resolved, setup-time parameters again on each call.
        term = self._action_term
        unclipped = (term.raw_actions * term._scale + term._offset)[:, self._action_indices]
        projected = torch.clamp(unclipped, min=self._bounds[..., 0], max=self._bounds[..., 1])
        return F.smooth_l1_loss(unclipped, projected, reduction="none", beta=self._beta).sum(dim=-1)


class knee_swing_flexion_deficit_l2(ManagerTermBase):
    """Cost insufficient bend in each knee's own terrain-only swing phase.

    Negative position is flexion for both corrected knees. The term is neutral
    while standing, in double stance/flight, outside the swing-time window, and
    once minimum bend is achieved. It never matches simultaneous knee angles
    or rewards excessive crouching. Smoothly ramp in after toe-off.

    Joint mapping, sensor references and fixed scalar checks belong to manager
    setup, not the rollout loop. Joint positions, commands and sensor timers
    remain live. No phase state is cached, so ordinary environment resets need
    no additional reward-state bookkeeping.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._minimum_flexion = cfg.params.get("minimum_flexion", 0.35)
        self._min_air_time = cfg.params.get("min_air_time", 0.02)
        self._max_air_time = cfg.params.get("max_air_time", 0.45)
        self._air_time_ramp = cfg.params.get("air_time_ramp", 0.06)
        self._move_threshold = cfg.params.get("move_threshold", 0.1)
        values = (
            self._minimum_flexion, self._min_air_time, self._max_air_time,
            self._air_time_ramp, self._move_threshold,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Knee swing parameters must be finite.")
        if (self._minimum_flexion <= 0.0 or self._min_air_time < 0.0
                or self._air_time_ramp <= 0.0 or self._move_threshold < 0.0):
            raise ValueError("Invalid knee swing flexion, time ramp, or movement threshold.")
        if self._max_air_time <= self._min_air_time + self._air_time_ramp:
            raise ValueError("Knee swing window must extend beyond its toe-off ramp.")
        asset_cfg = cfg.params["asset_cfg"]
        self._asset = env.scene[asset_cfg.name]
        if _joint_names(self._asset, asset_cfg) != ["Knee_L", "Knee_R"]:
            raise ValueError("Knee swing shaping requires ordered Knee_L, Knee_R selection.")
        self._joint_ids = asset_cfg.joint_ids
        self._sensors = [
            env.scene.sensors[cfg.params[key].name]
            for key in ("left_sensor_cfg", "right_sensor_cfg")
        ]
        self._command_name = cfg.params.get("command_name", "base_velocity")
        expected = (env.num_envs, 1)
        for sensor in self._sensors:
            data = sensor.data
            if (data.current_air_time is None or data.current_contact_time is None
                    or tuple(data.current_air_time.shape) != expected
                    or tuple(data.current_contact_time.shape) != expected):
                raise ValueError("Knee swing shaping requires one terrain-contact body per foot sensor.")

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        left_sensor_cfg: SceneEntityCfg,
        right_sensor_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        minimum_flexion: float = 0.35,
        min_air_time: float = 0.02,
        max_air_time: float = 0.45,
        air_time_ramp: float = 0.06,
        move_threshold: float = 0.1,
    ) -> torch.Tensor:
        positions = self._asset.data.joint_pos[:, self._joint_ids]
        sensor_data = [sensor.data for sensor in self._sensors]
        air = torch.cat([data.current_air_time for data in sensor_data], dim=1)
        contact = torch.cat([data.current_contact_time for data in sensor_data], dim=1)
        single_stance = (contact > 0.0).sum(dim=1) == 1
        moving = env.command_manager.get_command(self._command_name)[:, :2].norm(dim=-1) > self._move_threshold
        ramp = ((air - self._min_air_time) / self._air_time_ramp).clamp(0.0, 1.0)
        swing = (contact <= 0.0) & (air > self._min_air_time) & (air <= self._max_air_time)
        flexion = (-positions).clamp_min(0.0)
        deficit = (self._minimum_flexion - flexion).clamp_min(0.0).square()
        return (deficit * ramp * swing).sum(dim=1) * single_stance * moving
