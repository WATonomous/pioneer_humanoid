"""Path-relative reward and termination terms for a spiral (helix) staircase.

The walking direction rotates, so progress, facing and out-of-bounds are expressed relative to the path:
* the path coordinate is the height, which rises monotonically and cannot be faked by circling;
* the path direction is the tangent at the robot's angle about the helix axis (counter-clockwise upward);
* the lateral coordinate is the radius (treads narrow towards the axis, so the robot must keep the walking line).

The height rate is low-pass filtered (~0.7 s) before it is rewarded: the pelvis bob (2-3 cm at ~2 Hz) has a
larger derivative than the climb rate itself, and a raw rate would pay for pumping the pelvis.
"""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def spiral_frame(env, centre, asset_name: str = "robot"):
    """The robot's angle about the helix axis, its radius, and the unit tangent (num_envs, 2) of travel."""
    position = env.scene[asset_name].data.root_pos_w
    dx = position[:, 0] - centre[0]
    dy = position[:, 1] - centre[1]
    angle = torch.atan2(dy, dx)
    radius = torch.hypot(dx, dy)
    tangent = torch.stack([-torch.sin(angle), torch.cos(angle)], dim=1)
    return angle, radius, tangent


def height_rate(env, asset_name: str = "robot", tau: float = 0.7):
    """Low-pass-filtered root height and its rate (m/s). The filter state lives on the env; a respawn restarts it with a zero rate."""
    z = env.scene[asset_name].data.root_pos_w[:, 2].clone()
    key = "_spiral_height_filter"
    ema = getattr(env, key, None)
    if ema is None or ema.shape != z.shape:
        setattr(env, key, z)
        return z, torch.zeros_like(z)
    alpha = env.step_dt / tau
    jump = (z - ema).abs() > 0.5
    ema_new = torch.where(jump, z, ema + alpha * (z - ema))
    rate = torch.where(jump, torch.zeros_like(z), (ema_new - ema) / env.step_dt)
    setattr(env, key, ema_new)
    return ema_new, rate


# --------------------------------------------------------------------------- #
# Rewards
# --------------------------------------------------------------------------- #
def spiral_progress(
    env,
    asset_name: str = "robot",
    tau: float = 0.7,
    slope_factor: float = 1.0,
    cap: float | None = None,
):
    """Speed along the staircase: the filtered height-gain rate divided by ``slope_factor`` (sine of the incline).

    Expressed in the straight course's m/s units so the balance against the fall penalty carries over; the
    measurement stays the height.
    """
    _, rate = height_rate(env, asset_name, tau)
    progress = rate / max(slope_factor, 1e-6)
    if cap is not None:
        progress = torch.clamp_max(progress, cap)
    return progress


def spiral_path_progress(
    env,
    centre,
    asset_name: str = "robot",
    cap: float | None = 0.5,
):
    """Speed along the staircase tangent: the dense path-relative analogue of ``course_forward_progress``.

    The height rate is zero on the flat approach; this pays every step along the path. It is farmable by circling in
    principle; the no-progress termination (height gain) catches that.
    """
    _, _, tangent = spiral_frame(env, centre, asset_name)
    velocity = env.scene[asset_name].data.root_lin_vel_w[:, :2]
    progress = (velocity * tangent).sum(dim=1)
    if cap is not None:
        progress = torch.clamp_max(progress, cap)
    return progress


def stance_height_l2(
    env,
    feet_cfg,
    target_height: float = 0.84,
    asset_name: str = "robot",
):
    """Squared error of the root's height above the foot it stands on (scan-free standing height).

    The height scanner sees the staircase rising around the robot, so the terrain-relative version charges a
    systematic -13 per episode here. A crouched robot cannot clear the risers at all.
    """
    asset = env.scene[asset_name]
    root_z = asset.data.root_pos_w[:, 2]
    foot_z = asset.data.body_pos_w[:, feet_cfg.body_ids, 2].amin(dim=1)
    # The foot's origin sits 0.0365 m above its sole.
    height = root_z - foot_z + 0.0365
    return (height - target_height).square()


def spiral_riser_bonus(
    env,
    riser: float,
    standing_height: float = 0.845,
    asset_name: str = "robot",
    tau: float = 0.5,
    min_interval_steps: int = 0,
):
    """A discrete bonus for each riser the robot reaches (height under the robot is the riser count).

    Makes a successful step-up an unmistakable reward. The count is a high-water mark with half-riser hysteresis
    (only new height pays): a plain counter paid for pelvis bob, and a counter that could decrement was farmed by
    oscillating the height.

    ``min_interval_steps`` caps how often it pays, so the bonus is not worth more per second the faster the robot
    moves (otherwise the policy sprinted the stairs at 2.5-3.5 risers/s in a scrambling hop).
    """
    ema, _ = height_rate(env, asset_name, tau)
    key = "_spiral_riser_level"
    level = getattr(env, key, None)
    if level is None or level.shape != ema.shape:
        # Start the mark at the height the robot is already standing at, not
        # at zero: the multi-slot spawn can drop it a metre up the staircase,
        # and a zero mark would then pay for every riser below it (one per
        # step, ~20 in a row) without the robot moving.
        level = torch.round((ema - standing_height) / riser)
    height = ema - standing_height
    # A respawn (back on the floor) resets the mark.
    level = torch.where(height < -0.2, torch.zeros_like(level), level)
    # Rise only, and only past the middle of the next riser.
    above = height > (level + 0.5) * riser
    gained = above.float()
    # The mark always advances, so a riser that is too soon to be paid is
    # skipped for good rather than queued up and paid later.
    level = level + gained
    payment = gained
    if min_interval_steps > 0:
        step = env.episode_length_buf.to(gained.dtype)
        last = getattr(env, "_spiral_bonus_step", None)
        if last is None or last.shape != gained.shape:
            last = torch.full_like(gained, -1.0e9)
        allowed = (step - last) >= float(min_interval_steps)
        payment = gained * allowed.float()
        last = torch.where(payment > 0, step, last)
        env._spiral_bonus_step = last
    setattr(env, key, level)
    return payment


def spiral_retreat_cost(
    env,
    asset_name: str = "robot",
    tau: float = 0.7,
    slope_factor: float = 1.0,
):
    """Magnitude of the filtered height lost (non-negative; the weight carries the sign)."""
    _, rate = height_rate(env, asset_name, tau)
    return torch.clamp_min(-rate / max(slope_factor, 1e-6), 0.0)


def spiral_face_tangent(
    env,
    centre,
    deadband: float = 0.15,
    asset_name: str = "robot",
):
    """Squared yaw deviation from the staircase tangent at the robot's current angle, beyond a deadband."""
    angle, _, _ = spiral_frame(env, centre, asset_name)
    target = angle + math.pi / 2.0
    yaw = env.scene[asset_name].data.heading_w
    err = (yaw - target + math.pi) % (2.0 * math.pi) - math.pi
    return torch.clamp(err.abs() - deadband, min=0.0).square()


def spiral_radial_error(
    env,
    centre,
    walking_radius: float,
    deadband: float = 0.15,
    asset_name: str = "robot",
):
    """Squared deviation of the walking radius from the walking line, beyond a deadband (the treads end 0.14 m past it)."""
    _, radius, _ = spiral_frame(env, centre, asset_name)
    return torch.clamp((radius - walking_radius).abs() - deadband, min=0.0).square()


# --------------------------------------------------------------------------- #
# Terminations
# --------------------------------------------------------------------------- #
def spiral_complete(
    env,
    goal_height: float,
    centre,
    radius_min: float,
    radius_max: float,
    tilt_limit: float = 0.8,
    asset_name: str = "robot",
):
    """Reach the top of the staircase, upright and still on the treads."""
    data = env.scene[asset_name].data
    position = data.root_pos_w
    _, radius, _ = spiral_frame(env, centre, asset_name)
    gravity = data.projected_gravity_b
    finite = torch.isfinite(position).all(dim=1) & torch.isfinite(gravity).all(dim=1)
    tilt = torch.linalg.vector_norm(gravity[:, :2], dim=1)
    return (
        finite
        & (position[:, 2] >= goal_height)
        & (radius >= radius_min)
        & (radius <= radius_max)
        & (tilt <= tilt_limit)
        # XY tilt alone is also zero when fully upside-down.
        & (gravity[:, 2] < 0.0)
    )


def spiral_no_progress(
    env,
    min_gain: float = 1.0,
    min_time: float = 15.0,
    spawn_height: float = 0.84,
    asset_name: str = "robot",
):
    """Reset (with the fall penalty) if the robot has not climbed in a while.

    Height gain is measured from the per-env spawn height recorded by ``reset_root_state_spiral_slots``, so a raised
    slot is not immune.
    """
    z = env.scene[asset_name].data.root_pos_w[:, 2]
    spawn = getattr(env, "_spiral_spawn_height", None)
    if spawn is None or spawn.shape != z.shape:
        spawn = torch.full_like(z, spawn_height)
    gain = z - spawn
    time_ok = env.episode_length_buf >= int(min_time / env.step_dt)
    return time_ok & (gain < min_gain)


def spiral_out_of_bounds(
    env,
    centre,
    radius_min: float,
    radius_max: float,
    min_height: float = 0.2,
    max_height: float = 8.0,
    asset_name: str = "robot",
):
    """Left the staircase: off the outer edge, or off the top."""
    position = env.scene[asset_name].data.root_pos_w
    _, radius, _ = spiral_frame(env, centre, asset_name)
    return (
        (radius < radius_min)
        | (radius > radius_max)
        | (position[:, 2] < min_height)
        | (position[:, 2] > max_height)
    )


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
def reset_root_state_spiral_slots(
    env,
    env_ids,
    asset_cfg,
    slots: tuple[tuple[float, float, float, float], ...],
    riser: float = 0.12,
    standing_height: float = 0.845,
    xy_jitter: float = 0.05,
    yaw_jitter: float = 0.05,
    velocity_range: dict | None = None,
) -> None:
    """Reset the robot onto one of several treads along the staircase.

    ``slots`` are measured poses ``(x, y, z, yaw)`` (see tools/onshape_course/spiral_spawn_slots.py). Starting every
    episode on the flat approach lets the policy stand still; spreading the spawn puts half the batch in the climb.
    The reset also re-seeds the height filter, the spawn height and the riser high-water mark.
    """
    # Local import: this module is imported by CPU unit tests that stay torch-only.
    from isaaclab.utils import math as math_utils

    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    table = torch.tensor(slots, device=env.device, dtype=root_states.dtype)
    chosen = table[torch.randint(len(slots), (n,), device=env.device)]

    def _jitter(width: float) -> torch.Tensor:
        return (torch.rand(n, device=env.device, dtype=root_states.dtype) - 0.5) * 2.0 * width

    positions = torch.stack([chosen[:, 0] + _jitter(xy_jitter), chosen[:, 1] + _jitter(xy_jitter), chosen[:, 2]], dim=-1)
    positions = positions + env.scene.env_origins[env_ids]
    zeros = torch.zeros(n, device=env.device, dtype=root_states.dtype)
    orientations = math_utils.quat_from_euler_xyz(zeros, zeros, chosen[:, 3] + _jitter(yaw_jitter))
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

    # Re-seed the shaping state for the envs that just respawned. The filter
    # is set to the spawn height so the respawn does not read as a fall, the
    # spawn height feeds the no-progress termination, and the riser mark is
    # this tread's riser count (so only new height above the slot pays).
    #
    # The buffers are sized by the env count, never by the reset batch: a
    # partial reset (the usual case) must not reallocate them, or every other
    # env's mark and spawn height is wiped and the riser bonus pays again for
    # height it already climbed.
    num_envs = env.scene.num_envs
    z = positions[:, 2]

    def _buffer(name: str, fill: float) -> torch.Tensor:
        buf = getattr(env, name, None)
        if buf is None or buf.shape != (num_envs,):
            buf = torch.full((num_envs,), fill, device=env.device, dtype=root_states.dtype)
            setattr(env, name, buf)
        return buf

    _buffer("_spiral_height_filter", 0.0)[env_ids] = z
    _buffer("_spiral_spawn_height", standing_height)[env_ids] = z
    _buffer("_spiral_riser_level", 0.0)[env_ids] = torch.round((z - standing_height) / riser)
    # The riser-bonus payment clock: allow a payment straight away in the new
    # episode. Without this the clock keeps the previous episode's step count,
    # episode_length_buf restarts at 0, and no riser is ever paid again.
    _buffer("_spiral_bonus_step", -1.0e9)[env_ids] = -1.0e9


def tread_step_bonus(
    env,
    sensor_cfg,
    asset_cfg,
    riser: float = 0.12,
    threshold: float = 1.0,
):
    """Pay a footfall that lands on a tread above the other foot (0 to 1 risers at each contact onset).

    The root-based riser bonus cannot tell a step that advances from one that does not; a step back onto the same
    tread pays nothing.
    """
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    contact = forces > threshold
    previous = getattr(env, "_tread_step_prev", None)
    if previous is None or previous.shape != contact.shape:
        previous = contact.clone()
    onset = contact & ~previous
    env._tread_step_prev = contact.clone()
    z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    above_other = (z - z.flip(1)).clamp(min=0.0) / max(riser, 1e-6)
    return (onset.float() * above_other.clamp(max=1.0)).sum(dim=1)
