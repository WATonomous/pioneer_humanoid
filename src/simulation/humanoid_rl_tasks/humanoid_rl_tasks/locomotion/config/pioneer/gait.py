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
    """(num_envs, 2) phase in [0, 1) of the left and the right leg; ``period`` is the stride cycle in seconds."""
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
    """Squared backward shank rotation rate of airborne feet (the kick as seen by eye; prices the rotation itself)."""
    asset = env.scene[asset_cfg.name]
    airborne = ~_in_contact(env, sensor_cfg)
    omega = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    yaw = asset.data.heading_w
    lat_x, lat_y = -torch.sin(yaw), torch.cos(yaw)
    omega_lat = omega[..., 0] * lat_x.unsqueeze(-1) + omega[..., 1] * lat_y.unsqueeze(-1)
    back = omega_lat.clamp_min(0.0)
    return (back.square() * airborne.float()).sum(dim=1)


def feet_swing_behind_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["l_foot", "r_foot"], preserve_order=True),
    allowance: float = 0.10,
) -> torch.Tensor:
    """Squared distance an airborne foot hangs behind the body beyond ``allowance`` (the rear kick the foot-speed term cannot see)."""
    asset = env.scene[asset_cfg.name]
    airborne = ~_in_contact(env, sensor_cfg)
    feet = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    root = asset.data.root_pos_w[:, :2].unsqueeze(1)
    yaw = asset.data.heading_w
    rel = feet - root
    fwd = torch.cos(yaw).unsqueeze(-1) * rel[..., 0] + torch.sin(yaw).unsqueeze(-1) * rel[..., 1]
    behind = (-fwd - allowance).clamp_min(0.0)
    return (behind.square() * airborne.float()).sum(dim=1)


def contact_matches_phase(env, sensor_cfg: SceneEntityCfg, period: float = PERIOD) -> torch.Tensor:
    """Number of feet (0-2) whose ground contact matches the clock's stance or swing."""
    return (_in_contact(env, sensor_cfg) == (leg_phase(env, period) < STANCE)).float().sum(dim=1)


def contact_mismatch_l2(env, sensor_cfg: SceneEntityCfg, period: float = PERIOD) -> torch.Tensor:
    """Count of feet whose contact state disagrees with the gait clock's stance/swing phase (4 = both wrong)."""
    matches = contact_matches_phase(env, sensor_cfg, period)
    return (2.0 - matches).square()


def feet_swing_height(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared error of each airborne sole's height from SWING_HEIGHT."""
    sole_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2] - SOLE_BELOW_FOOT
    airborne = ~_in_contact(env, sensor_cfg)
    return ((sole_z - SWING_HEIGHT).square() * airborne).sum(dim=1)


def feet_swing_height_relative(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                               target: float = SWING_HEIGHT) -> torch.Tensor:
    """feet_swing_height for uneven ground: the airborne foot's height above the lower foot; ``target`` is the mid-swing height above the stance foot."""
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
    """Sole tilt beyond ``allowance_deg`` on loaded feet, in units of 10 deg, squared (prices heel lift / ball-of-foot push-off, not the normal heel-strike roll)."""
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
    """Reward forward foot speed in the first 0.2 s of swing, so the foot leaves forward rather than kicking back."""
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
    """Squared rate of change of the knee's commanded angle while the foot swings."""
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
    """Hinge on the raw action beyond ``threshold``: the reward is flat in the action past the clip rail, so the policy gets no gradient there."""
    act = env.action_manager.action
    if joint_names:
        act = act[:, _joint_action_idx(env, tuple(joint_names))]
    excess = torch.clamp(act.abs() - threshold, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def feet_flight(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalty on a flight phase (both feet off the ground), which turns a walk into a hop."""
    return (~_in_contact(env, sensor_cfg)).all(dim=1).float()
