"""Small, task-local knee shaping without changing physical target limits."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from isaaclab.managers import SceneEntityCfg

__all__ = ["knee_target_overshoot_smooth_l1", "knee_swing_flexion_deficit_l2"]


def _joint_names(asset, asset_cfg: SceneEntityCfg) -> list[str]:
    ids = asset_cfg.joint_ids
    if isinstance(ids, slice):
        return list(asset.joint_names[ids])
    return [asset.joint_names[index] for index in ids]


def knee_target_overshoot_smooth_l1(
    env,
    asset_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    beta: float = 0.1,
) -> torch.Tensor:
    """Cost raw POSITION targets beyond the action term's existing knee bounds.

    Unlike a cost on already-clipped targets, this still changes when a policy
    is far into saturation. The tail is linear and uncapped, avoiding a huge
    quadratic cost without discarding the signal to return toward the bounds.
    Scale, offset and bounds come from the actual action term, not URDF limits.
    No action or target tensor is modified.
    """
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("Knee overshoot beta must be finite and positive.")
    asset = env.scene[asset_cfg.name]
    names = _joint_names(asset, asset_cfg)
    if len(names) != 2 or set(names) != {"Knee_L", "Knee_R"}:
        raise ValueError("Knee overshoot must select exactly Knee_L and Knee_R.")
    term = env.action_manager.get_term(action_name)
    if getattr(term, "_asset", asset) is not asset:
        raise ValueError("Knee action term and selected joints must use the same asset.")
    action_names = list(term._joint_names)
    if len(action_names) != len(set(action_names)) or any(name not in action_names for name in names):
        raise ValueError("Knee action mapping is missing or ambiguous.")
    indices = [action_names.index(name) for name in names]
    raw = term.raw_actions
    if raw.ndim != 2 or raw.shape[1] != len(action_names):
        raise ValueError("Unexpected raw knee action shape.")
    clip = getattr(term, "_clip", None)
    if clip is None:
        raise ValueError("Knee overshoot requires configured processed target bounds.")
    try:
        bounds = torch.broadcast_to(clip, (*raw.shape, 2))[:, indices]
    except RuntimeError as error:
        raise ValueError("Knee target bounds have an incompatible shape.") from error
    # Bounds do not change during this task. Validate once, avoiding a device
    # synchronization at every reward step in a 4,096-environment rollout.
    validation_key = (tuple(names), id(clip), tuple(raw.shape))
    if getattr(term, "_wato_knee_bounds_validated", None) != validation_key:
        if not torch.isfinite(bounds).all() or (bounds[..., 0] >= bounds[..., 1]).any():
            raise ValueError("Knee target bounds must be finite and strictly ordered.")
        term._wato_knee_bounds_validated = validation_key
    unclipped = (raw * term._scale + term._offset)[:, indices]
    projected = torch.clamp(unclipped, min=bounds[..., 0], max=bounds[..., 1])
    return F.smooth_l1_loss(unclipped, projected, reduction="none", beta=beta).sum(dim=-1)


def knee_swing_flexion_deficit_l2(
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
    """Cost insufficient bend in each knee's own terrain-only swing phase.

    Negative position is flexion for both corrected knees. The term is neutral
    while standing, in double stance/flight, outside the swing-time window, and
    once minimum bend is achieved. It never matches simultaneous knee angles
    or rewards excessive crouching. Smoothly ramp in after toe-off.
    """
    values = (minimum_flexion, min_air_time, max_air_time, air_time_ramp, move_threshold)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Knee swing parameters must be finite.")
    if minimum_flexion <= 0.0 or min_air_time < 0.0 or air_time_ramp <= 0.0 or move_threshold < 0.0:
        raise ValueError("Invalid knee swing flexion, time ramp, or movement threshold.")
    if max_air_time <= min_air_time + air_time_ramp:
        raise ValueError("Knee swing window must extend beyond its toe-off ramp.")
    asset = env.scene[asset_cfg.name]
    if _joint_names(asset, asset_cfg) != ["Knee_L", "Knee_R"]:
        raise ValueError("Knee swing shaping requires ordered Knee_L, Knee_R selection.")
    positions = asset.data.joint_pos[:, asset_cfg.joint_ids]
    sensors = [env.scene.sensors[cfg.name] for cfg in (left_sensor_cfg, right_sensor_cfg)]
    air = torch.cat([sensor.data.current_air_time.reshape(positions.shape[0], -1) for sensor in sensors], dim=1)
    contact = torch.cat(
        [sensor.data.current_contact_time.reshape(positions.shape[0], -1) for sensor in sensors], dim=1
    )
    if air.shape != positions.shape or contact.shape != positions.shape:
        raise ValueError("Knee swing shaping requires one terrain-contact body per foot sensor.")
    single_stance = (contact > 0.0).sum(dim=1) == 1
    moving = env.command_manager.get_command(command_name)[:, :2].norm(dim=-1) > move_threshold
    ramp = ((air - min_air_time) / air_time_ramp).clamp(0.0, 1.0)
    swing = (contact <= 0.0) & (air > min_air_time) & (air <= max_air_time)
    flexion = (-positions).clamp_min(0.0)
    deficit = (minimum_flexion - flexion).clamp_min(0.0).square()
    return (deficit * ramp * swing).sum(dim=1) * single_stance * moving
