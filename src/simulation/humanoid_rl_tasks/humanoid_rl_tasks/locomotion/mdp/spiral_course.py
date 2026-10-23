"""Path-relative reward and termination terms for a spiral (helix) staircase.

On the straight Onshape course the walking direction is a fixed world axis, so
progress, facing and out-of-bounds can all be read off one coordinate. A
spiral staircase rotates that direction continuously, so every one of those
terms has to be expressed relative to the *path* instead. On a helix:

  * the path coordinate is the **height**. The staircase is a helicoid, so the
    height rises monotonically along the path (each tread is a flat wedge, but
    the wedge you are standing on gets higher as you go round). Height is
    therefore the only progress measure that cannot be faked: orbiting the
    bottom of the staircase gains nothing, and descending is negative.
  * the path *direction* is the **tangent** at the robot's current angle about
    the helix axis (the staircase winds counter-clockwise upward).
  * the *lateral* coordinate is the **radius**: the treads occupy an annulus
    and the robot has to stay on the walking line, because the tread gets
    tangentially narrower towards the axis.

The height rate has to be low-pass filtered before it is used as a reward.
The pelvis bobs 2-3 cm at ~2 Hz while walking, and the derivative of that bob
is +/-0.2-0.4 m/s -- larger than the climb rate itself (0.15 m/s on this
course) -- so the raw rate is dominated by the gait, and worse, it would pay
the policy for pumping its pelvis. A first-order filter with a time constant
of ~0.7 s attenuates the bob by an order of magnitude and leaves the climb
untouched.
"""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def spiral_frame(env, centre, asset_name: str = "robot"):
    """The robot's angle about the helix axis, its radius, and the tangent.

    Returns (angle, radius, tangent) where ``tangent`` is a (num_envs, 2)
    unit vector in the world xy plane: the direction of travel along the
    staircase (counter-clockwise, i.e. upward).
    """
    position = env.scene[asset_name].data.root_pos_w
    dx = position[:, 0] - centre[0]
    dy = position[:, 1] - centre[1]
    angle = torch.atan2(dy, dx)
    radius = torch.hypot(dx, dy)
    tangent = torch.stack([-torch.sin(angle), torch.cos(angle)], dim=1)
    return angle, radius, tangent


def height_rate(env, asset_name: str = "robot", tau: float = 0.7):
    """A low-pass-filtered root height and its rate of change (m/s).

    The filter state lives on the env, so it survives across the reward and
    termination terms. A reset (the robot respawns) shows up as a jump in the
    height and restarts the filter with a zero rate, so the respawn does not
    register as a huge fall.
    """
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
    """Forward progress along the staircase, as the speed along the path.

    The reward is the filtered rate of height gain, divided by ``slope_factor``
    -- the sine of the staircase's incline -- so that it is expressed in the
    same units as the straight course's speed-along-the-course term (m/s).
    That keeps the reward landscape (and the balance against the fall
    penalty) comparable between the two tasks, while the *measurement* stays
    the height, which is what cannot be faked on a spiral.

    ``slope_factor`` is a constant: the staircase's geometry does not change
    under the robot's feet, so the conversion never turns a descent into
    progress.
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
    """The speed along the staircase's *tangent* -- the path-relative analogue
    of the straight course's ``course_forward_progress``.

    The height-gain rate is the measure that cannot be faked, but it is zero
    for the whole approach (the robot walks 41 degrees of arc on the flat
    before the first tread, earning nothing) and it only pays once a riser
    has actually been cleared. This term is the dense one: it pays for every
    step the robot takes along the path, on the approach and on the staircase
    alike, exactly as the straight course's progress term pays for every step
    along its straight course.

    It is farmable in principle -- a robot could circle the bottom of the
    staircase and collect it without climbing -- and the guard against that
    is the *no-progress* termination, which measures height gain and so
    catches a robot that never climbs.
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
    """Squared error of the root's height above the foot it stands on.

    The straight course measures the base height with the height scanner
    (terrain-relative). On a spiral that does not work: the scan sees the
    staircase rising around the robot, so the mean ray height is metres above
    the tread the robot is standing on and the term charges a systematic -13
    per episode (it is switched off for this task).

    This is the scan-free equivalent: the feet are *on* the treads, so the
    lower foot's height is the terrain height, and the root's height above it
    is the standing height. It matters here because the forward kinematics
    put the maximum clearable riser at 0.182 m for a hip at 0.837 m but only
    0.143 m for a hip at 0.780 m -- a crouched robot cannot clear this
    staircase's 0.12 m risers at all.
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
):
    """A discrete bonus for each riser of the staircase the robot reaches.

    The staircase is a stack of equal risers, so the height under the robot
    *is* the riser count. Paying when the count goes up makes a successful
    step-up an immediate, unmistakable reward, which is what the policy needs:
    with only the filtered height *rate* to learn from, a sampled attempt
    looks like either -16 (a fall) or +0.9 (the filtered burst), and the
    successes in the batch do not stand out from the failures.

    The count is a **high-water mark** with a half-riser hysteresis: it only
    rises when the height passes the middle of the next riser, and it never
    falls again (except on a respawn). Two things that were tried first and
    both got gamed:

    * a plain counter (level = floor(height / riser)) flipped back and forth
      on the pelvis bob -- 2-3 cm of bob against a 12 cm riser -- and paid
      for risers that were never climbed (0.8 per step, a riser every four
      steps, physically impossible);
    * a counter that could also *decrement* was farmed by the policy
      *oscillating* its height: up half a riser, back down, up again, and
      each upward crossing paid (again 0.98 per step) while the net height
      stayed at zero.

    A high-water mark cannot be gamed: going up and down pays exactly once,
    and only *new* height pays.
    """
    ema, _ = height_rate(env, asset_name, tau)
    key = "_spiral_riser_level"
    level = getattr(env, key, None)
    if level is None or level.shape != ema.shape:
        level = torch.zeros_like(ema)
    height = ema - standing_height
    # A respawn (back on the floor) resets the mark.
    level = torch.where(height < -0.2, torch.zeros_like(level), level)
    # Rise only, and only past the middle of the next riser.
    above = height > (level + 0.5) * riser
    gained = above.float()
    level = level + gained
    setattr(env, key, level)
    return gained


def spiral_retreat_cost(
    env,
    asset_name: str = "robot",
    tau: float = 0.7,
    slope_factor: float = 1.0,
):
    """The magnitude of the height *lost* (the same filter as the progress).

    Non-negative, so the term's weight carries the sign: descending the
    staircase is charged. (The clamp is ``clamp_min``: the raw value is
    negative while the robot is climbing, and clamping it at zero from *below*
    is what turns it into a magnitude. With ``clamp_max`` the term returned
    the negative value, the -2.0 weight turned it into a *reward* for
    climbing, and the policy was paid twice for every step it took
    upwards.)
    """
    _, rate = height_rate(env, asset_name, tau)
    return torch.clamp_min(-rate / max(slope_factor, 1e-6), 0.0)


def spiral_face_tangent(
    env,
    centre,
    deadband: float = 0.15,
    asset_name: str = "robot",
):
    """Squared yaw deviation from the staircase's tangent, beyond a deadband.

    The straight course's ``face_course_direction`` compares the yaw against a
    fixed world angle; on a spiral the target rotates, so the target is the
    tangent at the robot's current angle.
    """
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
    """Squared deviation of the walking radius from the staircase's line.

    The treads are an annulus, and they get tangentially narrower towards the
    axis, so a robot that drifts inwards runs out of tread under its foot.
    The outer edge is close too (0.14 m past the walking line on this
    staircase), so drifting outwards means stepping off.
    """
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

    The height gain is measured against the standing height at the spawn, so
    a robot that shuffles around the bottom tread, or walks back down, never
    satisfies it.
    """
    z = env.scene[asset_name].data.root_pos_w[:, 2]
    gain = z - spawn_height
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
