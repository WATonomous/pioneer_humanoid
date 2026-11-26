"""Completion, reward and reset terms for the Onshape stair course (coordinates relative to the env origin)."""

from __future__ import annotations

import torch


def course_complete(
    env,
    goal_distance: float,
    lateral_limit: float = 2.0,
    minimum_height: float = 0.65,
    tilt_limit: float = 0.8,
    asset_name: str = "robot",
    axis: int = 0,
) -> torch.Tensor:
    """Reach the flat exit upright, inside the course (not just the top)."""
    data = env.scene[asset_name].data
    position = data.root_pos_w - env.scene.env_origins
    gravity = data.projected_gravity_b
    finite = torch.isfinite(position).all(dim=1) & torch.isfinite(gravity).all(dim=1)
    tilt = torch.linalg.vector_norm(gravity[:, :2], dim=1)
    return (
        finite
        & (position[:, axis] >= goal_distance)
        & (position[:, 1 - axis].abs() <= lateral_limit)
        # The goal is on the flat exit at spawn height, not on elevated stairs.
        & (position[:, 2] >= minimum_height)
        & (tilt <= tilt_limit)
        # XY tilt alone is also zero when fully upside-down.
        & (gravity[:, 2] < 0.0)
    )


def course_out_of_bounds(
    env,
    lateral_limit: float = 2.0,
    backward_limit: float = -0.7,
    asset_name: str = "robot",
    axis: int = 0,
) -> torch.Tensor:
    """Reset before the robot can leave its lane or bypass the stair flights."""
    position = env.scene[asset_name].data.root_pos_w - env.scene.env_origins
    return (
        ~torch.isfinite(position).all(dim=1)
        | (position[:, 1 - axis].abs() > lateral_limit)
        | (position[:, axis] < backward_limit)
    )


def face_course_direction(
    env,
    deadband: float = 0.2,
    asset_name: str = "robot",
    target: float = 0.0,
) -> torch.Tensor:
    """Squared yaw deviation from the course axis beyond a deadband (yaw-rate tracking alone lets a constant offset score perfectly)."""
    yaw = env.scene[asset_name].data.heading_w
    err = (yaw - target + torch.pi) % (2.0 * torch.pi) - torch.pi
    excess = (err.abs() - deadband).clamp_min(0.0)
    return excess * excess


def course_retreat_cost(env, asset_name: str = "robot", axis: int = 0) -> torch.Tensor:
    """Backward speed along the course, ``max(-v, 0)``; stacks on the symmetric progress term so sustained retreat costs twice what advance earns."""
    backward = -env.scene[asset_name].data.root_lin_vel_w[:, axis]
    return torch.clamp_min(backward, 0.0)


def course_forward_progress(env, asset_name: str = "robot", axis: int = 0, cap: float | None = None) -> torch.Tensor:
    """Signed forward speed along the course; linear through zero, so standing earns nothing. ``cap`` stops rushing from paying."""
    vel = env.scene[asset_name].data.root_lin_vel_w[:, axis]
    if cap is not None:
        vel = torch.clamp_max(vel, cap)
    return vel


def _band_from_profile(
    env,
    profile: list,
    position: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Interpolate a static ``[[position, band], ...]`` table at ``position`` (table cached on the env)."""
    key = (id(profile), len(profile), tuple(profile[0]), tuple(profile[-1]))
    cache = getattr(env, "_foot_swing_band_tables", None)
    if cache is None or cache[0] != key:
        table = torch.tensor(
            [[float(entry[0]), float(entry[1])] for entry in profile],
            device=device,
            dtype=torch.float32,
        )
        cache = (key, table)
        setattr(env, "_foot_swing_band_tables", cache)
    table = cache[1]
    positions = table[:, 0]
    bands = table[:, 1]
    idx = torch.searchsorted(positions, position.contiguous()).clamp(1, positions.numel() - 1)
    pos0, pos1 = positions[idx - 1], positions[idx]
    band0, band1 = bands[idx - 1], bands[idx]
    frac = ((position - pos0) / (pos1 - pos0).clamp_min(1e-6)).clamp(0.0, 1.0)
    return band0 + frac * (band1 - band0)


def _path_forward_offset(env, centre, positions, batch, swing_idx, stance_idx):
    """How far ahead of the stance foot the swing foot is, along the staircase tangent at the robot's angle about ``centre``."""
    root = env.scene["robot"].data.root_pos_w
    angle = torch.atan2(root[:, 1] - centre[1], root[:, 0] - centre[0])
    tx = -torch.sin(angle)
    ty = torch.cos(angle)
    dx = positions[batch, swing_idx, 0] - positions[batch, stance_idx, 0]
    dy = positions[batch, swing_idx, 1] - positions[batch, stance_idx, 1]
    return dx * tx + dy * ty


def foot_swing_clearance(
    env,
    left_sensor_cfg,
    right_sensor_cfg,
    feet_cfg,
    command_name: str,
    margin: float = 0.03,
    min_forward_speed: float = 0.0,
    forward_axis: int = 0,
    max_clearance: float | None = None,
    band: float | None = None,
    band_profile: list | None = None,
    band_profile_axis: int | None = None,
    path_centre: tuple | None = None,
    excess_coef: float = 0.0,
    back_allow: float | None = None,
    phase_ramp: float = 0.06,
    min_command_norm: float = 0.1,
) -> torch.Tensor:
    """Reward swing-foot height above the stance foot during single stance.

    Heights are foot-to-foot, double stance earns nothing, and the term is gated on commanded speed (strict ``>``
    ``min_command_norm``: set it below a slow command or the term reads zero).

    Optional shaping:
    * ``min_forward_speed``: payout ramps with forward speed, so a raised foot while standing pays nothing.
    * ``max_clearance``: clamp, so the term cannot be farmed by flinging the foot up.
    * ``band`` / ``excess_coef``: payout peaks at ``band`` and costs quadratically above it.
    * ``band_profile``: ``[[position, band], ...]`` per-position band, indexed along ``band_profile_axis``.
    * ``back_allow`` / ``phase_ramp``: payout only once the swing foot is at or ahead of the stance foot
      (no credit for a rear "butt-kick"); ``path_centre`` measures "ahead" along a spiral's tangent.
    """
    asset = env.scene[feet_cfg.name]
    positions = asset.data.body_pos_w[:, feet_cfg.body_ids, :]
    heights = positions[:, :, 2]
    forward = positions[:, :, forward_axis]
    sensors = [env.scene.sensors[cfg.name] for cfg in (left_sensor_cfg, right_sensor_cfg)]
    contact_time = torch.cat([sensor.data.current_contact_time for sensor in sensors], dim=1)
    in_contact = contact_time > 0.0
    single_stance = in_contact.int().sum(dim=1) == 1
    batch = torch.arange(heights.shape[0], device=heights.device)
    stance_idx = torch.argmax(in_contact.int(), dim=1)
    swing_idx = 1 - stance_idx
    clearance = (heights[batch, swing_idx] - heights[batch, stance_idx] - margin).clamp_min(0.0)
    if max_clearance is not None:
        clearance = clearance.clamp_max(max_clearance)
    if band_profile is not None:
        band_axis = forward_axis if band_profile_axis is None else band_profile_axis
        band_here = _band_from_profile(env, band_profile, positions[batch, stance_idx, band_axis], heights.device)
        excess = (clearance - band_here).clamp_min(0.0)
        reward = torch.minimum(clearance, band_here) - excess_coef * excess.square()
    elif band is not None:
        excess = (clearance - band).clamp_min(0.0)
        reward = clearance.clamp_max(band) - excess_coef * excess.square()
    else:
        reward = clearance
    reward = torch.where(single_stance, reward, torch.zeros_like(reward))
    reward = reward * (env.command_manager.get_command(command_name)[:, :2].norm(dim=1) > min_command_norm)
    if back_allow is not None:
        if path_centre is not None:
            d_fwd = _path_forward_offset(env, path_centre, positions, batch, swing_idx, stance_idx)
        else:
            d_fwd = forward[batch, swing_idx] - forward[batch, stance_idx]
        reward = reward * ((d_fwd - back_allow) / phase_ramp).clamp(0.0, 1.0)
    if min_forward_speed > 0.0:
        fwd = env.scene[feet_cfg.name].data.root_lin_vel_w[:, forward_axis]
        reward = reward * (fwd / min_forward_speed).clamp(0.0, 1.0)
    return reward


def foot_backswing_cost(
    env,
    left_sensor_cfg,
    right_sensor_cfg,
    feet_cfg,
    asset_name: str = "robot",
    forward_axis: int = 0,
    deadband: float = 0.20,
    path_centre: tuple | None = None,
) -> torch.Tensor:
    """Squared rearward excursion of each airborne foot beyond ``deadband`` (the rear "butt-kick").

    ``path_centre`` measures "behind" along a spiral's tangent instead of a world axis.
    """
    if deadband < 0.0:
        raise ValueError("Backswing deadband cannot be negative.")
    asset = env.scene[feet_cfg.name]
    foot_pos = asset.data.body_pos_w[:, feet_cfg.body_ids, :]
    root = env.scene[asset_name].data.root_pos_w
    if path_centre is not None:
        angle = torch.atan2(root[:, 1] - path_centre[1], root[:, 0] - path_centre[0])
        tx = -torch.sin(angle).unsqueeze(-1)
        ty = torch.cos(angle).unsqueeze(-1)
        offset = (foot_pos[..., 0] - root[:, 0:1]) * tx + (foot_pos[..., 1] - root[:, 1:2]) * ty
    else:
        offset = foot_pos[..., forward_axis] - root[:, forward_axis].unsqueeze(-1)
    behind = (-offset - deadband).clamp_min(0.0)
    sensors = [env.scene.sensors[cfg.name] for cfg in (left_sensor_cfg, right_sensor_cfg)]
    contact_time = torch.cat([sensor.data.current_contact_time for sensor in sensors], dim=1)
    in_air = (contact_time <= 0.0).float()
    return (behind.square() * in_air).sum(dim=1)


def hip_abduction_splay_l2(
    env,
    asset_cfg,
    deadband: float = 0.40,
) -> torch.Tensor:
    """Squared excess of ``|q|`` beyond ``deadband`` on the selected hip joints (a leg clearing risers by abducting and dragging the foot sideways)."""
    if deadband < 0.0:
        raise ValueError("Abduction deadband cannot be negative.")
    joints = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    excess = (joints.abs() - deadband).clamp_min(0.0)
    return excess.square().sum(dim=1)


def feet_impact(
    env,
    left_sensor_cfg,
    right_sensor_cfg,
    feet_cfg,
) -> torch.Tensor:
    """Squared downward foot speed at contact (soft landings), from the per-foot terrain sensors."""
    asset = env.scene[feet_cfg.name]
    vel_z = asset.data.body_lin_vel_w[:, feet_cfg.body_ids, 2]
    sensors = [env.scene.sensors[cfg.name] for cfg in (left_sensor_cfg, right_sensor_cfg)]
    contact = torch.cat(
        [(sensor.data.current_contact_time > 0.0) for sensor in sensors], dim=1
    )
    return (torch.clamp_min(-vel_z, 0.0) ** 2 * contact).sum(dim=1)


def feet_impact_contact(env, sensor_cfg, asset_cfg) -> torch.Tensor:
    """Squared downward foot speed on contact frames, from the single multi-body ``contact_forces`` sensor.

    ``sensor_cfg`` and ``asset_cfg`` must use the same body names with ``preserve_order=True``.
    """
    asset = env.scene[asset_cfg.name]
    vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    sensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    if contact.shape != vel_z.shape:
        raise ValueError(
            f"feet_impact_contact: sensor '{sensor_cfg.name}' selects {contact.shape} "
            f"bodies but the asset selects {vel_z.shape}; both cfg must use the same "
            f"body names with preserve_order=True."
        )
    return (torch.clamp_min(-vel_z, 0.0) ** 2 * contact).sum(dim=1)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by the (w, x, y, z) quaternion ``quat`` (local copy: this module stays torch-only for CPU tests)."""
    w = quat[..., 0:1]
    q = quat[..., 1:4]
    t = 2.0 * torch.cross(q, vec, dim=-1)
    return vec + w * t + torch.cross(q, t, dim=-1)


def foot_flat_l2(
    env,
    left_sensor_cfg,
    right_sensor_cfg,
    feet_cfg,
    local_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    deadband: float = 0.23,
    landing_window: float = 0.15,
    up_axis: int = 2,
) -> torch.Tensor:
    """Cost a foot that lands off-flat, over the first ``landing_window`` seconds of contact.

    Tilt is the angle between the foot body's ``local_up`` axis and world up, beyond ``deadband``. Only the landing
    is priced: over a whole stance the foot legitimately pitches 40+ degrees. The weight carries the sign.
    """
    if deadband < 0.0:
        raise ValueError("Foot-flat deadband cannot be negative.")
    if landing_window <= 0.0:
        raise ValueError("Foot-flat landing window must be positive.")
    quat = env.scene[feet_cfg.name].data.body_quat_w[:, feet_cfg.body_ids, :]
    axis = torch.tensor(local_up, dtype=quat.dtype, device=quat.device)
    up_world = _quat_apply(quat, axis.expand(quat.shape[:-1] + (3,)))
    tilt = torch.acos(up_world[..., up_axis].clamp(-1.0, 1.0))
    excess = (tilt - deadband).clamp_min(0.0)
    
    # Each foot is tracked by its own single-body terrain-contact sensor, so read
    # column 0 of each. Indexing these sensors with the *robot's* foot body ids
    # (what this did before) is out of bounds for a single-body sensor and trips
    # the CUDA index kernel -- invisible until now because the reward manager
    # skips zero-weight terms, and this term sat at weight 0.
    contact_times = []
    for sensor_cfg in (left_sensor_cfg, right_sensor_cfg):
        sensor = env.scene.sensors[sensor_cfg.name]
        # Select the cfg's body rather than assuming the sensor holds exactly
        # one: the shared 'contact_forces' sensor tracks every body, and only
        # the per-foot sensors used by the earlier tasks were single-body.
        times = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
        if times.shape[1] != 1:
            raise ValueError(
                f"foot_flat_l2 expects one body per sensor cfg, but '{sensor_cfg.name}' "
                f"selects {times.shape[1]} (bodies {sensor.body_names})."
            )
        contact_times.append(times[:, 0])
    contact_time = torch.stack(contact_times, dim=1)  # (num_envs, 2)

    landing = (contact_time > 0.0) & (contact_time <= landing_window)
    return (excess.square() * landing).sum(dim=1)


def course_no_progress(
    env,
    asset_name: str = "robot",
    axis: int = 0,
    min_displacement: float = 2.0,
    min_time: float = 15.0,
    spawn: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Penalized reset when the robot has not advanced ``min_displacement`` along the course by ``min_time`` (carries the fall penalty, so freezing costs)."""
    ref_x, ref_y = spawn if spawn is not None else (0.0, 0.0)
    base = env.scene.env_origins[:, axis] + (ref_x if axis == 0 else ref_y)
    # Signed forward displacement (not |.|): backing up must not satisfy a
    # forward-progress check.
    disp = env.scene[asset_name].data.root_pos_w[:, axis] - base
    time_ok = env.episode_length_buf >= int(min_time / env.step_dt)
    return time_ok & (disp < min_displacement)


def complete_stair_course_levels(
    env,
    env_ids,
    goal_distance: float,
    lateral_limit: float = 2.0,
    minimum_height: float = 0.65,
    tilt_limit: float = 0.8,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Terrain curriculum: promote a full upright traversal, demote other completed episodes."""
    completed = course_complete(env, goal_distance, lateral_limit, minimum_height, tilt_limit, asset_name)
    has_rollout = env.episode_length_buf[env_ids] > 0
    move_up = completed[env_ids] & ~env.termination_manager.terminated[env_ids] & has_rollout
    move_down = ~move_up & has_rollout
    terrain = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up=move_up, move_down=move_down)
    return terrain.terrain_levels.float().mean()


def reset_root_state_course_spawn(
    env,
    env_ids,
    asset_cfg,
    spawn_x: float,
    spawn_y: tuple[float, ...],
    x_jitter: float = 0.15,
    y_jitter: float = 0.06,
    yaw: float = 1.5707963,
    velocity_range: dict | None = None,
) -> None:
    """Reset onto one of several flat spawn slots (``spawn_y``) along the course, so episodes start at the tall risers too."""
    # Local import: this module is imported by CPU unit tests that stay torch-only.
    from isaaclab.utils import math as math_utils

    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    slots = torch.tensor(spawn_y, device=env.device, dtype=root_states.dtype)
    slot = slots[torch.randint(len(spawn_y), (n,), device=env.device)]
    xs = spawn_x + (torch.rand(n, device=env.device, dtype=root_states.dtype) - 0.5) * 2.0 * x_jitter
    ys = slot + (torch.rand(n, device=env.device, dtype=root_states.dtype) - 0.5) * 2.0 * y_jitter
    positions = torch.stack([xs, ys, root_states[:, 2]], dim=-1) + env.scene.env_origins[env_ids]
    zeros = torch.zeros(n, device=env.device, dtype=root_states.dtype)
    orientations = math_utils.quat_from_euler_xyz(zeros, zeros, torch.full_like(zeros, yaw))
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids)

    velocity_range = velocity_range or {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    ranges = torch.tensor(
        [velocity_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=env.device, dtype=root_states.dtype,
    )
    asset.write_root_velocity_to_sim(
        math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=env.device).to(root_states.dtype),
        env_ids,
    )


_PUSH_BASE_RANGE: dict[str, dict] = {}


def push_speed_schedule(
    env,
    env_ids,
    term_name: str = "push_robot",
    start_scale: float = 0.35,
    final_scale: float = 1.0,
    ramp_steps: int = 6000,
) -> torch.Tensor:
    """Ramp the push velocity range from ``start_scale`` to ``final_scale`` over ``ramp_steps`` env steps (24 per iteration)."""
    term = env.event_manager.get_term_cfg(term_name)
    full = _PUSH_BASE_RANGE.setdefault(term_name, dict(term.params["velocity_range"]))
    progress = min(1.0, float(env.common_step_counter) / max(1, ramp_steps))
    scale = start_scale + (final_scale - start_scale) * progress
    term.params["velocity_range"] = {k: (lo * scale, hi * scale) for k, (lo, hi) in full.items()}
    return torch.tensor([scale], device=env.device)
