"""Completion and curriculum for a forward up–walk–down stair course.

Only the Onshape stair-course tasks import these terms. All coordinates are relative to the
terrain's spawn origin, so tiles can be translated freely in the training map.
"""

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
    """Squared yaw deviation from the course (+X) axis, beyond a deadband.

    The corridor course never commands turning (heading is fixed at zero), but
    yaw-*rate* tracking only constrains angular velocity: a constant yaw offset
    scores perfectly while the robot walks sideways or backward, then follows
    its rotated command frame the wrong way up the course (observed: full
    ascent followed by a slow backward re-climb and topple). Small yaw
    transients inside ``deadband`` stay free for balance recovery; beyond it
    the cost grows quadratically, symmetric left/right. Corridor-only: open
    terrain tasks need turning, so this lives in the Onshape stair-course MDP module.
    """
    yaw = env.scene[asset_name].data.heading_w
    err = (yaw - target + torch.pi) % (2.0 * torch.pi) - torch.pi
    excess = (err.abs() - deadband).clamp_min(0.0)
    return excess * excess


def course_retreat_cost(env, asset_name: str = "robot", axis: int = 0) -> torch.Tensor:
    """Non-negative backward speed along the course (the retreat magnitude).

    The symmetric progress term prices retreat at the same rate it pays
    advance, which proved too cheap: the policy climbs (~10 s) then spends
    ~50 s moonwalking back down facing forward. This term stacks an equal
    cost on top, so sustained retreat costs 2x what advance earns, while brief
    recovery steps back cost little. Tracked separately so retreat volume is
    visible in logs (adjudication metric for the backward mode).

    Returns ``max(-v, 0)`` (>= 0) so the negative weight is a genuine cost.
    (A previous ``clamp_max(v, 0)`` returned the *negative* backward velocity,
    which, combined with the negative weight, paid the robot to retreat and
    silently cancelled the symmetric penalty.)
    """
    backward = -env.scene[asset_name].data.root_lin_vel_w[:, axis]
    return torch.clamp_min(backward, 0.0)


def course_forward_progress(env, asset_name: str = "robot", axis: int = 0, cap: float | None = None) -> torch.Tensor:
    """Signed forward speed along the course (+X in the env frame).

    Every tile is axis-aligned: local +X runs spawn → ascent → walkway →
    descent → exit, matching world +X. Unlike the inherited exponential
    velocity-tracking reward, this term is linear through zero, so standing
    still earns nothing while any forward motion pays immediately — including
    the first unsteady steps of a policy that has not yet learned to walk.
    Backward motion is punished symmetrically; retreat never reaches the goal.
    With cap set, forward speed above the cap pays no extra (kills rushing
    while keeping the stand-still pays-nothing property).
    """
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
    """Interpolate a static ``[[position, band], ...]`` table at ``position``.

    The table is a task constant (the course's own shape), so it is uploaded
    once and cached on the env; only the interpolation runs per step.
    """
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
    """How far ahead of the stance foot the swing foot is, along the path.

    On a spiral staircase the forward direction rotates, so the offset is
    measured against the staircase's tangent at the robot's current angle
    about ``centre`` (the staircase winds counter-clockwise upward).
    """
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
) -> torch.Tensor:
    """Reward swing-foot height above the stance foot during single stance.

    Air time alone pays for shuffling; toe clearance is what negotiates risers.
    Heights are measured foot-to-foot, so no terrain query is needed and the
    term stays correct on the approach, flights, and walkway alike. Double
    stance earns nothing, so standing straddled across two treads cannot harvest
    it — only a genuine step pays. Gated on commanded motion like the air-time
    reward so standing-reserve environments stay neutral.

    With ``min_forward_speed`` set, the payout additionally ramps with measured
    forward speed along ``forward_axis`` (capped at 1.0). A held-up foot while
    standing still then earns exactly zero, while a genuine step during forward
    motion pays fully; the ramp is linear so there is no gradient cliff.

    With ``max_clearance`` set, the measured clearance is clamped from above.
    The term is otherwise *unbounded*, so a policy can farm it by flinging a
    foot arbitrarily high above the stance foot (observed as a toppling
    foot-wave once the forward-speed gate was satisfied by a slow creep). The
    cap should sit just above the tallest riser the task must clear.

    With ``band``/``excess_coef`` set, the payout peaks at ``band`` and then
    *costs* quadratically above it, so "just enough" is optimal rather than
    "more is free" (the return can go negative; the weight carries the sign).

    ``band_profile`` replaces the scalar band with a *per-position* one: a
    static ``[[position, band], ...]`` table of the clearance the course
    actually demands at that spot. A single global band has to be tall enough
    for the tallest riser, so it also pays a high-knee lift on the flat
    approach and on the descents, where nothing needs clearing; the profile
    pays only for the riser that is actually coming, which is what a person
    does. The table is interpolated linearly between samples, so the band
    moves smoothly with the robot. ``band_profile_axis`` says which coordinate
    the table is indexed by (the default is the forward axis; a spiral
    staircase indexes it by height, because height is the path coordinate
    there).

    With ``back_allow``/``phase_ramp`` set, the payout is gated to the *forward*
    phase of the swing: it ramps in only once the swing foot is at or ahead of
    the stance foot (``back_allow`` is the tolerated rearward offset). Paying
    height while the foot is behind the body is what teaches the rear
    "butt-kick" swing, so the gate removes that credit; a genuine forward
    knee-led step is unaffected.

    ``path_centre`` makes that gate path-relative: on a spiral staircase the
    forward direction rotates, so the swing foot is measured against the
    staircase's tangent at the robot's current angle about ``path_centre``
    instead of against a fixed world axis. The default (None) keeps the
    straight-course behaviour.
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
    reward = reward * (env.command_manager.get_command(command_name)[:, :2].norm(dim=1) > 0.1)
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
    """Cost lifting the swing foot *behind* the body (the rear "butt-kick").

    Squared rearward excursion of each airborne foot beyond ``deadband``,
    measured from the root along ``forward_axis``. Toe-off legitimately leaves
    the foot a little behind the body, so the deadband must clear that; only
    the exaggerated rear-lift that precedes an inefficient step pays. Returns a
    non-negative magnitude (the term's weight carries the sign).

    ``path_centre`` makes the measurement path-relative: on a spiral staircase
    "behind" is behind *along the staircase*, so the excursion is measured
    against the tangent at the robot's current angle about that centre.
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
    """Cost hip ab/adduction splay beyond ``deadband``, on either side.

    Measured on the Onshape course: the RIGHT leg cleared risers by abducting the
    hip (~0.9 rad) with a near-straight knee (``|knee| ~ 0.05``), dragging the foot
    ~0.5 m out to the side, while the LEFT leg tucked properly (knee ~0.9 rad, hip
    ~0). The only Hip_A term in the reward set is the base ``joint_deviation_hip``
    L1 at weight -0.1 -- two orders of magnitude below the forward-progress income
    -- so a splayed swing was effectively free.

    Returns, per environment, the sum over the selected hip joints of the squared
    excess of ``|q|`` beyond ``deadband``. The absolute value keeps it
    sign-agnostic, so a one-sided splay and a symmetric wide stance both pay,
    while normal walking (measured inside ~0.4 rad) stays free. Weight carries the
    magnitude; the term itself is non-negative.
    """
    if deadband < 0.0:
        raise ValueError("Abduction deadband cannot be negative.")
    joints = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    excess = (joints.abs() - deadband).clamp_min(0.0)
    return excess.square().sum(dim=1)


def hip_roll_splay_l2(
    env,
    asset_cfg,
    deadband: float = 0.15,
) -> torch.Tensor:
    """Cost hip *roll* splay beyond ``deadband``, on either side.

    Same shape as :func:`hip_abduction_splay_l2`, for the third hip DOF
    (``Hip_R_*``). Re-measured on the Onshape course *after* the knee-tuck fix:
    the right leg's lateral swing was as much hip roll as abduction -- a
    persistent -0.38 rad every stride against the left leg's +0.05 -- while the
    right foot still dragged 0.64 m out to the side (left: 0.03 m). Nothing in
    the reward set priced Hip_R at all: the base ``joint_deviation_hip`` L1 sits
    at weight -0.1, two orders of magnitude under the forward-progress income.

    The deadband follows the same rule as the abduction term -- calibrate to the
    leg the policy already swings correctly rather than to a guess. The left
    leg's measured roll (mean +0.05, max +0.33) defines a narrow free band, so
    normal walking stays free while a one-sided splay pays.
    """
    if deadband < 0.0:
        raise ValueError("Roll deadband cannot be negative.")
    return hip_abduction_splay_l2(env, asset_cfg, deadband)


def foot_lateral_splay_l2(
    env,
    left_sensor_cfg,
    right_sensor_cfg,
    feet_cfg,
    asset_name: str = "robot",
    deadband: float = 0.20,
    max_excess: float = 0.50,
    power: float = 2.0,
) -> torch.Tensor:
    """Cost each *airborne* foot's sideways excursion from the body centreline.

    The third and last piece of the one-sided right-leg circumduction fix. The
    two hip terms above price the splay in *joint* space, but measurement showed
    joint space does not track the foot: the right hip came down to +0.39 rad
    abduction / -0.38 rad roll yet the right foot still dragged 0.64 m out to
    the side (left: 0.03 m). Every term that could have seen the lateral
    excursion was blind to it: ``foot_swing_clearance`` gates on the fore/aft
    phase, ``knee_swing_flexion`` is a sagittal joint angle, and
    ``feet_crossing_l2`` only fires when the feet come *together* -- so an
    outward splay, which *grows* the separation, never triggers it. This term
    prices exactly the quantity that was free: the foot's lateral offset from
    the root.

    Measured in the same frame as the stride diagnostic (``dx = foot - root``
    along the body's lateral axis), per airborne foot, and gated on air time so
    a planted foot pays nothing however wide the stance. The offset is taken in a
    yaw-aligned frame (body +Y), so the heading wobble the policy legitimately
    uses to navigate does not read as splay -- the frame ``feet_crossing_l2``
    uses. Sign-agnostic (``|offset|``), so a one-sided splay and a symmetric wide
    stance both pay; the term itself is non-negative and the weight carries the
    sign.

    ``deadband`` follows the house rule -- calibrate to the leg that already
    swings correctly. Re-measured on model 1999 over 21 strides per foot (the
    new ``dx_max`` channel of the PLAYARC probe): the left foot's per-swing peak
    lateral excursion is 0.04-0.15 m (mean 0.08), so 0.20 m leaves a correct
    swing free with margin, while the right foot's 0.47-0.69 m (mean 0.57) pays
    on every airborne frame.

    ``power`` shapes the cost: 2.0 (the default) is the squared excess, whose
    gradient vanishes at the deadband edge; 1.0 is the linear excess, which
    bites immediately past the deadband with a constant gradient -- the shape to
    reach for when a squared term sits at a non-zero equilibrium. ``max_excess``
    caps the per-foot excess before the power, so neither shape can run away in
    a fallen/flailing episode and blow up the value function (a capped quadratic
    is unchanged for every normal offset).
    """
    if deadband < 0.0:
        raise ValueError("Lateral splay deadband cannot be negative.")
    if max_excess <= 0.0:
        raise ValueError("Lateral splay max_excess must be positive.")
    if power <= 0.0:
        raise ValueError("Lateral splay power must be positive.")
    asset = env.scene[feet_cfg.name]
    feet = asset.data.body_pos_w[:, feet_cfg.body_ids, :]
    root = env.scene[asset_name].data.root_pos_w.unsqueeze(1)
    rel = feet - root
    yaw = env.scene[asset_name].data.heading_w
    # Body-frame left axis (body +Y) expressed in world is (-sin yaw, cos yaw),
    # so the body-lateral component of a world offset is that dot product.
    lateral = -torch.sin(yaw).unsqueeze(-1) * rel[..., 0] + torch.cos(yaw).unsqueeze(-1) * rel[..., 1]
    excess = (lateral.abs() - deadband).clamp(min=0.0, max=max_excess)
    sensors = [env.scene.sensors[cfg.name] for cfg in (left_sensor_cfg, right_sensor_cfg)]
    contact_time = torch.cat([sensor.data.current_contact_time for sensor in sensors], dim=1)
    in_air = (contact_time <= 0.0).float()
    return (excess.pow(power) * in_air).sum(dim=1)


def body_over_support_l2(
    env,
    left_sensor_cfg,
    right_sensor_cfg,
    feet_cfg,
    asset_name: str = "robot",
    deadband: float = 0.08,
    max_excess: float = 0.40,
    power: float = 1.0,
    min_planted: int = 1,
) -> torch.Tensor:
    """Cost the lateral offset of the *support centre* from the body centreline.

    ``foot_lateral_splay_l2`` prices each foot against the root, which is the
    right target for an individual leg wandering -- but it cannot fix a body that
    has moved *itself* off its own base of support, because moving the legs
    around is not the same as moving the pelvis. Measurement on the fixed policy
    separated the two: at spawn the feet sit at -0.116 / +0.173 m (separation
    0.289 m), and while walking the *planted* feet sit at -0.003 / +0.198 m
    (separation 0.201 m). The leg pair is therefore not splitting apart at all --
    the whole stance is shifted ~+0.10 m in world x, i.e. the pelvis is riding
    that far to the *left* of the midpoint of its feet, which is what makes the
    right leg read as "splayed". This term prices that offset directly: the mean
    lateral position of the *contacting* feet (the support centre) measured
    against the root, so the body has to keep itself over the feet it is standing
    on. Gating on contact keeps a swinging foot -- which is legitimately 0.1-0.3 m
    out -- from being read as a body offset; with both feet planted it is the
    true stance centre, and during single support it is the stance foot itself.

    ``deadband`` is calibrated the same way as its sibling: the symmetric spawn
    posture already sits at +0.028 m (the two feet are not perfectly symmetric
    about the root), so 0.08 m leaves that natural asymmetry and a normal slight
    lean free. Same ``power`` / ``max_excess`` convention as the sibling term
    (linear by default, capped so a fall cannot produce a runaway return).

    ``min_planted`` selects when the term prices the offset. With the default 1
    it fires whenever *any* foot is planted, which makes it demand the pelvis sit
    over the stance foot during single support -- and that is fatal: tried that
    way on `onshape_train34` (weight -30.0) and `onshape_train34b` (-8.0), both
    wrecked the gait (`course_complete` 0.83 -> 0.57 and 0.60, `base_tilt`
    doubled) even at -8.0, where its own logged cost was only -0.003. So in that
    form the *direction* is wrong, not the magnitude: the pelvis-left-of-feet
    posture is load-bearing while a leg swings, and pricing it away stops the
    walk. With ``min_planted=2`` the term only prices the offset in *double*
    support -- both feet on the ground -- where the pelvis can be re-centred over
    a real base of support without any of the weight being carried on the stance
    foot alone, which is what makes it safe to ask for.
    """
    if deadband < 0.0:
        raise ValueError("Support-centre deadband cannot be negative.")
    if max_excess <= 0.0:
        raise ValueError("Support-centre max_excess must be positive.")
    if power <= 0.0:
        raise ValueError("Support-centre power must be positive.")
    asset = env.scene[feet_cfg.name]
    feet = asset.data.body_pos_w[:, feet_cfg.body_ids, :]
    root = env.scene[asset_name].data.root_pos_w.unsqueeze(1)
    rel = feet - root
    yaw = env.scene[asset_name].data.heading_w
    lateral = -torch.sin(yaw).unsqueeze(-1) * rel[..., 0] + torch.cos(yaw).unsqueeze(-1) * rel[..., 1]
    sensors = [env.scene.sensors[cfg.name] for cfg in (left_sensor_cfg, right_sensor_cfg)]
    contact_time = torch.cat([sensor.data.current_contact_time for sensor in sensors], dim=1)
    planted = (contact_time > 0.0).float()
    n_planted = planted.sum(dim=1)
    centre = (lateral * planted).sum(dim=1) / n_planted.clamp_min(1.0)
    excess = (centre.abs() - deadband).clamp(min=0.0, max=max_excess)
    return excess.pow(power) * (n_planted >= float(min_planted))


def stillness_hunger(
    env,
    asset_name: str = "robot",
    axis: int = 0,
    min_speed: float = 0.1,
) -> torch.Tensor:
    """Flat per-step cost for staying below min course-direction speed.

    Complements gated rewards: instead of merely paying zero for stillness,
    stillness actively costs. 1 where |forward speed| < min_speed, else 0;
    weight carries the magnitude. Breaks stand-still optima that survive on
    small tracking leftovers.
    """
    vel = env.scene[asset_name].data.root_lin_vel_w[:, axis].abs()
    return (vel < min_speed).float()


def feet_impact(
    env,
    left_sensor_cfg,
    right_sensor_cfg,
    feet_cfg,
) -> torch.Tensor:
    """Squared downward foot speed at contact (soft landings).

    Swing motion is free; only the touchdown spike pays, quadratically. Uses
    the terrain contact sensors so flight/holds read zero.
    """
    asset = env.scene[feet_cfg.name]
    vel_z = asset.data.body_lin_vel_w[:, feet_cfg.body_ids, 2]
    sensors = [env.scene.sensors[cfg.name] for cfg in (left_sensor_cfg, right_sensor_cfg)]
    contact = torch.cat(
        [(sensor.data.current_contact_time > 0.0) for sensor in sensors], dim=1
    )
    return (torch.clamp_min(-vel_z, 0.0) ** 2 * contact).sum(dim=1)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by the (w, x, y, z) quaternion ``quat`` (broadcast over leading dims).

    Local copy so this MDP module keeps depending on nothing but torch (the CPU
    unit tests import it directly).
    """
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
    """Cost a foot that *lands* off-flat, over the first ``landing_window`` seconds of contact.

    Measured on the Onshape course (model 1999, foot-body orientation at all 36
    touchdowns of a 20 s clip): the right foot lands at a clear angle -- 18.8 deg
    mean, 53.9 deg max off horizontal -- while the left lands flat (2.9 deg mean,
    11.7 deg max). Nothing in the reward set priced foot orientation at all, so
    the landing angle was free.

    The window is not a detail, it is what makes the term correct: over the whole
    contact phase the foot legitimately pitches 41.7 deg (left) / 38.4 deg (right)
    on average -- up to 67 deg at toe-off -- so 17/18 left strides and 18/18 right
    strides exceed any usable deadband at some point in stance. Pricing the whole
    stance would therefore tax every normal step, good foot included. Priced at
    the landing only, it targets exactly the reported defect: a foot that touches
    down askew pays each frame it stays askew inside the window, so a foot that
    lands badly and then settles flat pays only briefly.

    The angle is measured on the foot *body*, not the ankle joint. The joint-space
    proxy was rejected by measurement: the left foot's ankle pitch (mean -0.365
    rad) is *larger* than the right's (+0.264) yet only the left lands flat, and
    the right's own landings split into a flat mode (1-5 deg, ankle pitched to
    +0.5..+0.68) and a tilted mode (20-41 deg, ankle near neutral) -- the whole
    leg chain sets the foot's angle. ``local_up`` is the sole normal in the foot's
    own frame; on this asset local +Z points up (the left foot's flat landings
    measure ``up = (0.007, -0.001, 1.000)``), so tilt is ``acos`` of the world-Z
    component of the rotated axis. Non-negative: the weight carries the sign.

    ``deadband`` follows the same rule as the hip terms -- calibrate to the leg
    that already lands correctly: 0.23 rad (13.2 deg) leaves the left's measured
    11.7 deg worst case free with margin, while the right's tilted landings pay
    every frame of the window.
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
    
    # Get contact times for the feet bodies only
    # The contact_forces sensor tracks all robot bodies; we slice by feet body_ids
    sensor = env.scene.sensors[left_sensor_cfg.name]  # same sensor for both
    contact_time_all = sensor.data.current_contact_time  # (num_envs, num_bodies)
    contact_time = contact_time_all[:, feet_cfg.body_ids]  # (num_envs, 2)
    
    landing = (contact_time > 0.0) & (contact_time <= landing_window)
    return (excess.square() * landing).sum(dim=1)


def course_no_progress(    env,
    asset_name: str = "robot",
    axis: int = 0,
    min_displacement: float = 2.0,
    min_time: float = 15.0,
    spawn: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Penalized reset when the robot hasn't displaced along the course.

    Unlike a free time-out, this termination carries the fall penalty, so
    freezing in place costs instead of paying. Displacement is measured from
    the spawn point (env origin + spawn offset), since scattered starts sit
    metres from the origin itself.
    """
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
    """Promote a full upright traversal; demote unsuccessful completed episodes.

    The initial reset has no rollout and must not alter terrain levels. A fall
    on the same frame as crossing the goal is not a success. No reward, asset,
    observation, actuator, or PPO parameter is modified by this curriculum.
    """
    completed = course_complete(env, goal_distance, lateral_limit, minimum_height, tilt_limit, asset_name)
    has_rollout = env.episode_length_buf[env_ids] > 0
    move_up = completed[env_ids] & ~env.termination_manager.terminated[env_ids] & has_rollout
    move_down = ~move_up & has_rollout
    terrain = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up=move_up, move_down=move_down)
    return terrain.terrain_levels.float().mean()
