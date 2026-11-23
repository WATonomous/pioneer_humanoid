"""Gait-clock terms: the legs follow a fixed-period clock half
a cycle apart, each foot is meant to stand for the first STANCE of its cycle and to swing SWING_HEIGHT high."""
import math

import torch
from isaaclab.managers import SceneEntityCfg

PERIOD = 1.0  # s, about a human stride cycle
STANCE = 0.55  # fraction of a leg's cycle on the ground
SWING_HEIGHT = 0.10  # sole above the ground mid-swing, m
SOLE_BELOW_FOOT = 0.059  # foot body origin above the sole, standing


def leg_phase(env, period: float = PERIOD) -> torch.Tensor:
    """(num_envs, 2) phase in [0, 1) of the left and the right leg.

    ``period`` is the stride cycle in seconds. The default 1 s suits a 0.26 m/s
    walk on the straight course's 0.29 m treads; a staircase with 0.55 m of
    going per tread needs a much slower cycle at the same speed, and the phase
    is also what the scripted arms and the contact-matching reward ride on.
    """
    phase = (env.episode_length_buf * env.step_dt) % period / period
    return torch.stack([phase, (phase + 0.5) % 1.0], dim=-1)


def gait_clock(env, period: float = PERIOD) -> torch.Tensor:
    """Observation: sin and cos of the left leg's phase."""
    phase = leg_phase(env, period)[:, 0]
    return torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=-1)


def _in_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0


def shank_backward_rotation_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["l_shank_foot", "r_shank_foot"],
                                               preserve_order=True),
) -> torch.Tensor:
    """Backward rotation of a *shank* whose foot is airborne, squared and summed.

    This is the kick itself, in the frame the eye sees it. Measured with
    kick_probe.py (385 swings aligned at lift-off), over the first 0.28 s of
    swing: thigh pitch -11.6 deg -> +0.9 deg (the hip flexes only 12.5 deg)
    while shank pitch goes -22.9 deg -> -39.5 deg, i.e. the shank rotates 16.6
    deg *backwards in world space*. The foot still ends up forward because the
    thigh moves, which is why pricing the foot's position missed it and why
    pricing the knee angle failed: the fold is geometrically required to clear
    the riser from a foot that leaves the tread 0.24 m behind the body, so the
    policy simply paid the penalty instead (knee fold unchanged at -0.73 rad at
    34% of income).

    What a smooth step-up does instead is lead with the thigh: as the knee
    folds, the hip flexes by at least as much, so the shank does not swing back.
    This prices the shank's own angular velocity, so that is the only way to
    satisfy it. Sign: ``corr(omega_lat, d(shank pitch)/dt) = -0.92`` measured,
    so positive lateral angular velocity is backward rotation.
    """
    asset = env.scene[asset_cfg.name]
    airborne = ~_in_contact(env, sensor_cfg)
    omega = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    yaw = asset.data.heading_w
    lat_x, lat_y = -torch.sin(yaw), torch.cos(yaw)
    omega_lat = omega[..., 0] * lat_x.unsqueeze(-1) + omega[..., 1] * lat_y.unsqueeze(-1)
    back = omega_lat.clamp_min(0.0)
    return (back.square() * airborne.float()).sum(dim=1)


def knee_flexion_airborne_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Knee_L", "Knee_R"], preserve_order=True),
    max_flexion: float = 0.45,
) -> torch.Tensor:
    """Knee flexion past ``max_flexion`` while that leg's foot is airborne.

    The visible "back kick": after toe-off the knee folds from -0.23 rad to
    -0.73 rad over 0.24 s and stays there, i.e. the shank rotates ~29 deg
    backwards. Moving the *foot* forward does not remove it -- the hip can carry
    the foot forward while the knee folds -- so this prices the joint angle
    itself. Gated on airborne, so a leg that pre-folds while still loaded (which
    is what a real step-up does: the heel comes up before the toe leaves) is
    free, and the policy is pushed to clear the riser with the hip instead of
    snapping the shank up behind the body.

    The knee's convention is negative-flexed (clip -1.10..-0.05 rad), so flexion
    is ``-q`` and the excess is charged squared past ``max_flexion``.
    """
    asset = env.scene[asset_cfg.name]
    airborne = ~_in_contact(env, sensor_cfg)
    flex = (-asset.data.joint_pos[:, asset_cfg.joint_ids] - max_flexion).clamp_min(0.0)
    return (flex.square() * airborne.float()).sum(dim=1)


def feet_swing_behind_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["l_foot", "r_foot"], preserve_order=True),
    allowance: float = 0.10,
) -> torch.Tensor:
    """How far each airborne foot sits *behind* the root, past an allowance.

    The kick, measured (kick_probe.py, 374 swings aligned at lift-off): at
    lift-off the foot is 0.24 m behind the root, it drifts to 0.27 m over the
    next 0.24 s while the knee folds to -0.74 rad (the shank rotates ~31 deg
    backwards -- the visible kick), and only then does it come forward.
    ``feet_swing_height_relative`` prices only *height*, so folding the knee is
    the cheapest way to clear the riser: the foot rises without the hip ever
    carrying it forward.

    This charges the squared forward/aft excess of an airborne foot behind the
    root. It is positional, so unlike a velocity penalty it cannot be satisfied
    by moving slowly, and it is gated on airborne so the stance foot -- which is
    legitimately behind the body as the robot advances over it -- is free.
    ``allowance`` keeps a normal trailing position free and leaves the foot room
    to swing forward.
    """
    asset = env.scene[asset_cfg.name]
    airborne = ~_in_contact(env, sensor_cfg)
    feet = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    root = asset.data.root_pos_w[:, :2].unsqueeze(1)
    yaw = asset.data.heading_w
    rel = feet - root
    fwd = torch.cos(yaw).unsqueeze(-1) * rel[..., 0] + torch.sin(yaw).unsqueeze(-1) * rel[..., 1]
    behind = (-fwd - allowance).clamp_min(0.0)
    return (behind.square() * airborne.float()).sum(dim=1)


def feet_swing_backward_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["l_foot", "r_foot"], preserve_order=True),
) -> torch.Tensor:
    """Rearward body-frame speed of each *airborne* foot, squared and summed.

    Measured on the spiral gait (kick_probe.py, 381 swings aligned at lift-off):
    the foot leaves the tread 0.24 m behind the root, drifts a further 3 cm
    backwards over the next 0.26 s while the knee folds to -0.75 rad, and only
    then reaches forward. That is the visible "back kick". Nothing in the reward
    prices the fore/aft direction of swing -- ``feet_swing_height_relative``
    prices only the sole's height -- so the cheapest way to clear the riser is
    to tuck the leg up behind the body, which is what the policy learned.

    The term is the squared *backward* component of an airborne foot's velocity
    in the robot's frame, so forward motion costs nothing (a normal swing, which
    starts behind the body and travels forward, pays zero) and only the kick is
    charged. It is gated on airborne so a planted foot, which is legitimately
    moving backwards relative to the body as the robot advances over it, is free.
    """
    asset = env.scene[asset_cfg.name]
    airborne = ~_in_contact(env, sensor_cfg)
    vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    yaw = asset.data.heading_w
    # body +X in world is (cos yaw, sin yaw)
    fwd = torch.cos(yaw).unsqueeze(-1) * vel[..., 0] + torch.sin(yaw).unsqueeze(-1) * vel[..., 1]
    back = (-fwd).clamp_min(0.0)
    return (back.square() * airborne.float()).sum(dim=1)


def contact_matches_phase(env, sensor_cfg: SceneEntityCfg, period: float = PERIOD) -> torch.Tensor:
    """Number of feet (0-2) whose ground contact matches the clock's stance or swing."""
    return (_in_contact(env, sensor_cfg) == (leg_phase(env, period) < STANCE)).float().sum(dim=1)


def contact_mismatch_l2(env, sensor_cfg: SceneEntityCfg, period: float = PERIOD) -> torch.Tensor:
    """Squared number of feet that are *not* in the clock's stance/swing state.

    Zero when both feet match the clock, 4 when both are wrong. A gait prior:
    with a clock period set to the cadence the terrain implies, a swing takes
    the whole cycle fraction instead of a 0.2 s flick, which is what turns a
    stub-and-shuffle into a step onto the next tread. Written as a penalty so
    that standing still (which matches nothing for long) cannot farm it.
    """
    matches = contact_matches_phase(env, sensor_cfg, period)
    return (2.0 - matches).square()


def feet_swing_height(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared error of each airborne sole's height from SWING_HEIGHT."""
    sole_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2] - SOLE_BELOW_FOOT
    airborne = ~_in_contact(env, sensor_cfg)
    return ((sole_z - SWING_HEIGHT).square() * airborne).sum(dim=1)


def feet_swing_height_relative(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                               target: float = SWING_HEIGHT) -> torch.Tensor:
    """feet_swing_height for uneven ground: the airborne foot's height above the lower foot, not above z = 0.

    ``target`` is the mid-swing height above the *stance* foot. A staircase
    taller than the flat course's 0.10 m has to raise it to its own clearance
    band, or the term pays the policy to under-lift and stub the risers.
    """
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    above = foot_z - foot_z.min(dim=1, keepdim=True).values
    airborne = ~_in_contact(env, sensor_cfg)
    return ((above - target).square() * airborne).sum(dim=1)


def stance_foot_velocity(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared velocity of the feet that are on the ground (slipping or scuffing)."""
    vel = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return (vel.square().sum(dim=-1) * _in_contact(env, sensor_cfg)).sum(dim=1)


def joint_pos_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared joint positions (hip roll and yaw kept near zero)."""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids].square().sum(dim=1)


def stance_foot_tilt(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared tilt of the feet that are on the ground (gravity's xy in the foot frame): land and stand on a flat sole,
    not on its edge."""
    from isaaclab.utils.math import quat_apply_inverse

    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, quat.shape[1], -1)
    tilt = quat_apply_inverse(quat.reshape(-1, 4), gravity.reshape(-1, 3)).view(quat.shape[0], -1, 3)[..., :2]
    return (tilt.square().sum(dim=-1) * _in_contact(env, sensor_cfg)).sum(dim=1)


def stance_foot_tilt_hinge(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                           allowance_deg: float = 10.0) -> torch.Tensor:
    """Sole tilt beyond ``allowance_deg`` on loaded feet, in units of 10 deg, squared.

    Measured (step_probe.py, 435 real swings): the spiral policy's back foot is
    already tilted 48 deg from flat 0.2 s before lift-off and 57 deg at
    lift-off, i.e. it rolls onto the ball of the foot and pushes off from there;
    the straight deliverable goes 8 -> 35 deg over the same window. A hinge, so a
    flat or nearly flat sole costs nothing and only the heel-lift is charged;
    ``stance_foot_tilt`` (squared, no allowance) would also tax every normal
    heel-strike roll. Requires sensor and asset cfg to list the same feet in the
    same order (both preserve_order=True): the contact sensor's body ids are NOT
    the robot's body ids.
    """
    from isaaclab.utils.math import quat_apply_inverse

    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, quat.shape[1], -1)
    g = quat_apply_inverse(quat.reshape(-1, 4), gravity.reshape(-1, 3)).view(quat.shape[0], -1, 3)
    tilt_deg = torch.rad2deg(torch.asin(g[..., :2].norm(dim=-1).clamp(max=1.0)))
    excess = ((tilt_deg - allowance_deg) / 10.0).clamp_min(0.0)
    return (excess.square() * _in_contact(env, sensor_cfg)).sum(dim=1)


def stance_knee(env, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared distance of each knee from `target` while its leg is in the clock's stance: a near-straight supporting
    leg, as people walk, instead of a crouch. asset_cfg.joint_ids are the [left, right] knees."""
    knee = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return ((knee - target).square() * (leg_phase(env) < STANCE)).sum(dim=1)


def physics_blowup(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Termination: a robot whose state is no longer finite or whose base moves faster than 20 m/s (a PhysX glitch). Ends
    the episode before its numbers reach the critic."""
    data = env.scene[asset_cfg.name].data
    finite = torch.isfinite(data.joint_vel).all(dim=1) & torch.isfinite(data.root_lin_vel_w).all(dim=1)
    return ~finite | (data.root_lin_vel_w.norm(dim=1) > 20.0)


def finite(data: torch.Tensor, limit: float = 100.0) -> torch.Tensor:
    """Observation modifier: NaN -> 0, +-inf -> +-limit."""
    return torch.nan_to_num(data, nan=0.0, posinf=limit, neginf=-limit)


def _finite_rewards() -> None:
    """A rare PhysX glitch (a robot whose state turns NaN) gives NaN rewards for that step, and one NaN reward turns the
    whole PPO update into NaNs (training died at random iterations). Zero non-finite rewards; physics_blowup then resets
    the robot."""
    from isaaclab.managers import RewardManager

    if getattr(RewardManager.compute, "_finite", False):
        return
    compute = RewardManager.compute

    def finite_compute(self, dt: float) -> torch.Tensor:
        return torch.nan_to_num(compute(self, dt), nan=0.0, posinf=0.0, neginf=0.0)

    finite_compute._finite = True
    RewardManager.compute = finite_compute


_finite_rewards()



def _joint_action_idx(env, names) -> torch.Tensor:
    """Indices of ``names`` in the joint-position action vector (cached on the env)."""
    key = "_gait_idx_" + "_".join(names)
    cached = getattr(env, key, None)
    if cached is None:
        term = env.action_manager.get_term("joint_pos")
        order = list(term._joint_names)
        cached = torch.tensor([order.index(n) for n in names], device=env.device)
        setattr(env, key, cached)
    return cached


def swing_start_forward(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["l_foot", "r_foot"], preserve_order=True),
    window: float = 0.2,
    v_min: float = -0.3,
    v_max: float = 0.6,
) -> torch.Tensor:
    """Forward world-frame foot speed in the first ``window`` s after lift-off.

    Measured on the 0.951 spiral model (kick_probe.py, real swings only): at
    +0.2 s the foot has moved forward on average (+0.13 m) but 55 % of swings
    went *backwards* first (worst decile -7 cm), because the foot leaves the
    tread 0.26 m behind the body and the knee folds before the hip leads. The
    existing terms price position and shank rotation, neither of which is the
    thing the eye sees: the foot's own motion in the world. This pays for that
    motion directly, only while the swing is young (``current_air_time`` below
    ``window``), clamped so it cannot be farmed by flinging the foot.
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    t_air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    young = (t_air > 0.0) & (t_air < window)
    vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    yaw = asset.data.heading_w
    fwd = torch.cos(yaw).unsqueeze(-1) * vel[..., 0] + torch.sin(yaw).unsqueeze(-1) * vel[..., 1]
    return (fwd.clamp(v_min, v_max) * young.float()).sum(dim=1)


def knee_swing_target_rate_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    knee_actions=("Knee_L", "Knee_R"),
) -> torch.Tensor:
    """Squared *change of the knee's commanded angle* (rad) while its foot is airborne.

    In radians, not raw action units, so it is independent of the knee scale.
    Gated to swing so the stance knee (which has to react to the landing) is
    free. Targets the stutter in the knee's rise (p95 second difference 3.6x the
    mean) and the relay behaviour (knee target on a clip value 45 % of steps).
    """
    term = env.action_manager.get_term("joint_pos")
    idx = _joint_action_idx(env, knee_actions)
    scale = torch.as_tensor(term._scale, device=env.device, dtype=torch.float32)
    if scale.dim() == 2:
        scale = scale[:, idx]
    elif scale.dim() == 1:
        scale = scale[idx]
    delta = (env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]) * scale
    airborne = ~_in_contact(env, sensor_cfg)
    return (delta.square() * airborne.float()).sum(dim=1)


def action_saturation_l2(env, threshold: float = 3.0, joint_names=None) -> torch.Tensor:
    """Squared raw-action magnitude *beyond* ``threshold`` (zero inside it).

    Why this exists. The joint-position action is ``target = offset + scale * a``
    and the target is clipped (knee) or simply lies outside the joint's range
    (every other leg joint). Past that point the reward is *flat* in ``a``, so
    the Gaussian policy's mean gets no gradient pulling it back and drifts
    outward until the ``clip_actions`` rail: measured on the spiral lineage,
    mean |a| is 7-10 on every leg joint (straight-stairs deliverable: 0.2-2),
    targets sit outside the joint range 50-87 % of steps, Hip_F is at its effort
    limit 40-48 % of steps, and the knee target is on a clip value 45 % of
    steps. That is bang-bang control: the joints slam their position stops,
    which shows up as ~270 rad/s^2 leg-acceleration spikes and 2-4 kN foot
    forces on a vertically gentle landing, and why every action-*rate* penalty
    and slew cap left the spikes untouched.

    ``action_l2`` cannot fix it without taxing legitimate motion everywhere; a
    hinge is free for any command inside the usable window and steep outside.
    """
    act = env.action_manager.action
    if joint_names:
        act = act[:, _joint_action_idx(env, tuple(joint_names))]
    excess = torch.clamp(act.abs() - threshold, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def feet_flight(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """1 when *both* feet are off the ground, 0 while at least one is loaded.

    A staircase is a support-transfer task: a flight phase is a fall that
    happens to land higher up. The spiral policy drifted into one anyway
    (measured 4.3 footfalls/s, 0.27 duty, ~73% of the time airborne, 44 deg
    torso pitch), because the per-riser bonus pays for height and a hop is the
    cheapest way to get it. Used with a negative weight; the term is zero in
    every supported step, so it does not prefer standing still (which earns no
    climb reward).
    """
    return (~_in_contact(env, sensor_cfg)).all(dim=1).float()
