"""Ports of Pioneer Isaac flat-locomotion terms to mjlab tensor APIs.

Source: humanoid_rl_tasks/locomotion/mdp/rewards.py. Reward dt scaling is
performed by mjlab, as in Isaac Lab. Metrics below are unweighted physical units.
"""

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from .model import PARAMS


def velocity_yaw(env):
    d = env.scene["robot"].data
    return quat_apply_inverse(yaw_quat(d.root_link_quat_w), d.root_link_lin_vel_w)


def tracking_error(env):
    return torch.linalg.vector_norm(
        env.command_manager.get_command("base_velocity")[:, :2]
        - velocity_yaw(env)[:, :2],
        dim=1,
    )


def track_linear(env, std=0.5):
    return torch.exp(-tracking_error(env).square() / std**2)


def yaw_error(env):
    return (
        env.command_manager.get_command("base_velocity")[:, 2]
        - env.scene["robot"].data.root_link_ang_vel_w[:, 2]
    ).abs()


def track_yaw(env, std=0.5):
    return torch.exp(-yaw_error(env).square() / std**2)


def air_time(env, threshold=0.2):
    d = env.scene["feet"].data
    contact = d.current_contact_time > 0
    mode_time = torch.where(contact, d.current_contact_time, d.current_air_time)
    single = contact.sum(dim=1) == 1
    reward = (
        torch.where(single[:, None], mode_time, 0)
        .min(dim=1)
        .values.clamp(max=threshold)
    )
    moving = env.command_manager.get_command("base_velocity")[:, :2].norm(dim=1) > 0.1
    return reward * moving


def feet_slide(env, asset_cfg: SceneEntityCfg):
    contacts = env.scene["feet"].data.force.norm(dim=-1) > 1.0
    v = env.scene["robot"].data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    return (v.norm(dim=-1) * contacts).sum(dim=1)


def feet_crossing(env, asset_cfg: SceneEntityCfg, margin=0.08):
    d = env.scene["robot"].data
    rel = d.body_link_pos_w[:, asset_cfg.body_ids] - d.root_link_pos_w[:, None]
    quat = yaw_quat(d.root_link_quat_w)[:, None].expand(-1, 2, -1)
    pos = quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).reshape(-1, 2, 3)
    return (margin - (pos[:, 0, 1] - pos[:, 1, 1])).clamp(min=0).square()


def vertical_velocity(env):
    return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()


def hip_deviation(env, asset_cfg: SceneEntityCfg):
    d = env.scene["robot"].data
    return (
        (
            d.joint_pos[:, asset_cfg.joint_ids]
            - d.default_joint_pos[:, asset_cfg.joint_ids]
        )
        .abs()
        .sum(dim=1)
    )


def speed_fraction(env):
    robot = env.scene["robot"]
    limits = torch.tensor(
        [PARAMS[n.rsplit("_", 1)[0]][3] for n in robot.joint_names], device=env.device
    )
    return (robot.data.joint_vel.abs() / limits).amax(dim=1)


def overspeed(env):
    return (speed_fraction(env) - 1).clamp(min=0).square()


def effort_fraction(env):
    robot = env.scene["robot"]
    limits = torch.tensor(
        [PARAMS[n.rsplit("_", 1)[0]][2] for n in robot.joint_names], device=env.device
    )
    return (robot.data.actuator_force.abs() / limits).amax(dim=1)


def forward_speed(env):
    return velocity_yaw(env)[:, 0]


def body_ground(env):
    return env.scene["body_ground"].data.found.any(dim=1)
