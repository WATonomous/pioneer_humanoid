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
        # flat exit at spawn height, not elevated stairs
        & (position[:, 2] >= minimum_height)
        & (tilt <= tilt_limit)
        # xy tilt is also zero when upside-down
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


def course_retreat_cost(env, asset_name: str = "robot", axis: int = 0) -> torch.Tensor:
    """Backward speed along the course, ``max(-v, 0)``.

    Stacks on the symmetric progress term so sustained retreat costs twice what advance earns.
    """
    backward = -env.scene[asset_name].data.root_lin_vel_w[:, axis]
    return torch.clamp_min(backward, 0.0)


def course_forward_progress(env, asset_name: str = "robot", axis: int = 0, cap: float | None = None) -> torch.Tensor:
    """Signed forward speed along the course.

    Linear through zero, so standing earns nothing and any forward motion pays. With ``cap`` set,
    speed above it pays no extra (no rushing).
    """
    vel = env.scene[asset_name].data.root_lin_vel_w[:, axis]
    if cap is not None:
        vel = torch.clamp_max(vel, cap)
    return vel


def feet_impact_contact(env, sensor_cfg, asset_cfg) -> torch.Tensor:
    """Squared downward foot speed on contact frames, from the single multi-body ``contact_forces`` sensor.

    ``sensor_cfg`` and ``asset_cfg`` must use the same body names with ``preserve_order=True``;
    columns are matched by position.
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
    """Reset the robot onto one of several flat spawn slots (``spawn_y``) along the course.

    Slots are gap centres between parts, where the whole sole sits on flat ground, so the robot spawns
    standing; the jitters stay small for the same reason. Spreading the spawn lets an episode start at the tall risers.
    """
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
    """Ramp the push velocity range from ``start_scale`` to ``final_scale`` so the gait settles first.

    ``ramp_steps`` counts env steps (24 per iteration). The configured range is cached in this module
    because the event manager forwards every param to the push function.
    """
    term = env.event_manager.get_term_cfg(term_name)
    full = _PUSH_BASE_RANGE.setdefault(term_name, dict(term.params["velocity_range"]))
    progress = min(1.0, float(env.common_step_counter) / max(1, ramp_steps))
    scale = start_scale + (final_scale - start_scale) * progress
    term.params["velocity_range"] = {k: (lo * scale, hi * scale) for k, (lo, hi) in full.items()}
    return torch.tensor([scale], device=env.device)
