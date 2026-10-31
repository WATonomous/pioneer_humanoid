"""Gait-clock terms: the legs follow a fixed-period clock half
a cycle apart, each foot is meant to stand for the first STANCE of its cycle and to swing SWING_HEIGHT high."""
import math

import torch
from isaaclab.managers import SceneEntityCfg

PERIOD = 1.0  # s, about a human stride cycle
STANCE = 0.55  # fraction of a leg's cycle on the ground
SWING_HEIGHT = 0.10  # sole above the ground mid-swing, m
SOLE_BELOW_FOOT = 0.059  # foot body origin above the sole, standing


def leg_phase(env) -> torch.Tensor:
    """(num_envs, 2) phase in [0, 1) of the left and the right leg."""
    phase = (env.episode_length_buf * env.step_dt) % PERIOD / PERIOD
    return torch.stack([phase, (phase + 0.5) % 1.0], dim=-1)


def gait_clock(env) -> torch.Tensor:
    """Observation: sin and cos of the left leg's phase."""
    phase = leg_phase(env)[:, 0]
    return torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=-1)


def _in_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0


def contact_matches_phase(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Number of feet (0-2) whose ground contact matches the clock's stance or swing."""
    return (_in_contact(env, sensor_cfg) == (leg_phase(env) < STANCE)).float().sum(dim=1)


def feet_swing_height(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared error of each airborne sole's height from SWING_HEIGHT."""
    sole_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2] - SOLE_BELOW_FOOT
    airborne = ~_in_contact(env, sensor_cfg)
    return ((sole_z - SWING_HEIGHT).square() * airborne).sum(dim=1)


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

