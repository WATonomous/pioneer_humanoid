"""Pioneer gait rewards, adapted from the repository walking recipe; no experiment variants."""

import math

import mujoco
import torch
from mjlab.envs.mdp.rewards import (
    action_rate_l2,
    flat_orientation_l2,
    is_alive,
    is_terminated,
    joint_acc_l2,
    joint_pos_limits,
    joint_torques_l2,
    joint_vel_l2,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse as quat_apply_inverse
from mjlab.utils.lab_api.math import yaw_quat as yaw_quat


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


PERIOD = 1.0


def leg_phase(env):
    phase = (env.episode_length_buf * env.step_dt / PERIOD).remainder(1.0)
    return torch.stack((phase, (phase + 0.5).remainder(1.0)), dim=-1)


STANCE = 0.55


def moving(env):
    command = env.command_manager.get_command("base_velocity")
    return (command[:, :2].norm(dim=1) > 0.05) | (command[:, 2].abs() > 0.05)


def contact_matches_phase(env):
    contact = env.scene["feet"].data.force[..., 2] > 1.0
    desired = (leg_phase(env) < STANCE) | ~moving(env)[:, None]
    return (contact == desired).float().sum(dim=1)


def sole_clearance(poses, half_sizes, floor_z):
    up = torch.zeros_like(poses[..., :3])
    up[..., 2] = 1.0
    local_up = quat_apply_inverse(poses[..., 3:7], up)
    extent = (local_up.abs() * half_sizes).sum(dim=-1)
    return poses[..., 2] - extent - floor_z[:, None]


def clearance_score(height, air_time, contact_time, speed, settings):
    ratio = (speed / settings.reference_speed).clamp(0.5, 1.5)
    duration = settings.swing_duration / ratio.sqrt()
    phase = (air_time / duration[:, None]).clamp(0.0, 1.0)
    target = settings.clearance_height * ratio[:, None] * torch.sin(torch.pi * phase)
    tracking = torch.exp(-((height - target) / settings.clearance_std).square())
    contact = contact_time > 0.0
    swing = ~contact & (air_time > 0.0) & (air_time < duration[:, None])
    moving = (speed > 0.1) & (contact.sum(dim=1) == 1)
    score = (tracking * swing).sum(dim=1) / swing.sum(dim=1).clamp(min=1)
    return score * moving


class SwingClearance:
    def __init__(self, cfg, env):
        asset = cfg.params["asset_cfg"]
        robot = env.scene[asset.name]
        ids = robot.indexing.geom_ids[asset.geom_ids].cpu().numpy()
        model = env.sim.mj_model
        if not (model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_BOX).all():
            raise ValueError("Swing clearance requires box-shaped foot collisions")
        self.half_sizes = torch.as_tensor(
            model.geom_size[ids].copy(), device=env.device, dtype=torch.float32
        )

    def __call__(self, env, asset_cfg, settings):
        poses = env.scene[asset_cfg.name].data.geom_pose_w[:, asset_cfg.geom_ids]
        height = sole_clearance(poses, self.half_sizes, env.scene.env_origins[:, 2])
        feet = env.scene["feet"].data
        speed = env.command_manager.get_command("base_velocity")[:, :2].norm(dim=1)
        return clearance_score(
            height, feet.current_air_time, feet.current_contact_time, speed, settings
        )


SWING_HEIGHT = 0.1


class FootCost(SwingClearance):
    def __call__(self, env, asset_cfg, kind):
        robot = env.scene[asset_cfg.name].data
        contact = env.scene["feet"].data.force[..., 2] > 1.0
        if kind == "height":
            poses = robot.geom_pose_w[:, asset_cfg.geom_ids]
            height = sole_clearance(poses, self.half_sizes, env.scene.env_origins[:, 2])
            return ((height - SWING_HEIGHT).square() * ~contact).sum(dim=1) * moving(
                env
            )
        if kind == "slip":
            velocity = robot.body_link_lin_vel_w[:, asset_cfg.body_ids]
            return (velocity.square().sum(dim=-1) * contact).sum(dim=1)
        if kind == "tilt":
            quat = robot.body_link_quat_w[:, asset_cfg.body_ids]
            up = torch.zeros_like(quat[..., :3])
            up[..., 2] = 1.0
            tilt = quat_apply_inverse(quat, up)[..., :2].square().sum(dim=-1)
            return (tilt * contact).sum(dim=1)
        raise ValueError(f"Unknown foot cost: {kind}")


def feet_crossing(env, asset_cfg: SceneEntityCfg, margin=0.08):
    d = env.scene["robot"].data
    rel = d.body_link_pos_w[:, asset_cfg.body_ids] - d.root_link_pos_w[:, None]
    quat = yaw_quat(d.root_link_quat_w)[:, None].expand(-1, 2, -1)
    pos = quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).reshape(-1, 2, 3)
    return (margin - (pos[:, 0, 1] - pos[:, 1, 1])).clamp(min=0).square()


HEIGHT_TARGET = 0.836065


def base_height(env):
    z = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (z - HEIGHT_TARGET).square()


def hip_position(env, asset_cfg):
    return (
        env.scene[asset_cfg.name]
        .data.joint_pos[:, asset_cfg.joint_ids]
        .square()
        .sum(dim=1)
    )


def vertical_velocity(env):
    return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()


def angular_velocity_xy(env):
    return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(dim=1)


def stance_knee(env, asset_cfg):
    knee = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return ((knee + 0.15).square() * (leg_phase(env) < STANCE)).sum(dim=1) * moving(env)


ARM_PARAMS = {
    "l_shoulder_pitch": (60.0, 6.0, 53.0, 6.0),
    "r_shoulder_pitch": (60.0, 6.0, 53.0, 6.0),
    "l_elbow": (30.0, 3.0, 22.0, 6.0),
    "r_elbow": (30.0, 3.0, 22.0, 6.0),
}


PARAMS = {
    "Hip_F": (100.0, 8.0, 222.0, 3.6652, 0.25, 0.08),
    "Hip_A": (80.0, 5.0, 120.0, 20.944, 0.06, 0.04),
    "Hip_R": (60.0, 4.0, 60.0, 20.42, 0.15, 0.0),
    "Knee": (100.0, 8.0, 222.0, 3.6652, 0.3, -0.22),
    "Ankle_P": (80.0, 6.0, 60.0, 20.42, 0.12, -0.14),
    "Ankle_R": (80.0, 6.0, 60.0, 20.42, 0.08, 0.04),
}


def speed_fraction(env):
    robot = env.scene["robot"]
    limits = torch.tensor(
        [
            ARM_PARAMS[n][3] if n in ARM_PARAMS else PARAMS[n.rsplit("_", 1)[0]][3]
            for n in robot.joint_names
        ],
        device=env.device,
    )
    return (robot.data.joint_vel.abs() / limits).amax(dim=1)


def overspeed(env):
    return (speed_fraction(env) - 1).clamp(min=0).square()


def gait_clock(env):
    angle = 2 * math.pi * leg_phase(env)[:, 0]
    return torch.stack((angle.sin(), angle.cos()), dim=-1)


def arm_targets(phase, commands):
    amplitude = 0.35 * (commands[:, 0].abs() / 0.8).clamp(max=1.0)
    swing = amplitude * torch.sin(2 * math.pi * (phase - STANCE / 2 + 0.25))
    return torch.stack(
        (-swing, swing, torch.full_like(swing, -0.3), torch.full_like(swing, -0.3)),
        dim=-1,
    )


def make_rewards():
    """Return independent reward terms for the current gait recipe."""
    return {
        "track_linear": RewardTermCfg(func=track_linear, weight=1.0),
        "track_yaw": RewardTermCfg(func=track_yaw, weight=0.5),
        "fall": RewardTermCfg(func=is_terminated, weight=-200.0),
        "contact": RewardTermCfg(func=contact_matches_phase, weight=0.18),
        "swing_height": RewardTermCfg(
            func=FootCost,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=("l_foot", "r_foot"),
                    geom_names=("l_foot_collision_0", "r_foot_collision_0"),
                    preserve_order=True,
                ),
                "kind": "height",
            },
            weight=-20.0,
        ),
        "contact_no_vel": RewardTermCfg(
            func=FootCost,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=("l_foot", "r_foot"),
                    geom_names=("l_foot_collision_0", "r_foot_collision_0"),
                    preserve_order=True,
                ),
                "kind": "slip",
            },
            weight=-0.2,
        ),
        "feet_flat": RewardTermCfg(
            func=FootCost,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=("l_foot", "r_foot"),
                    geom_names=("l_foot_collision_0", "r_foot_collision_0"),
                    preserve_order=True,
                ),
                "kind": "tilt",
            },
            weight=-2.0,
        ),
        "feet_crossing": RewardTermCfg(
            func=feet_crossing,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=("l_foot", "r_foot"),
                    geom_names=("l_foot_collision_0", "r_foot_collision_0"),
                    preserve_order=True,
                ),
                "margin": 0.2,
            },
            weight=-40.0,
        ),
        "base_height": RewardTermCfg(func=base_height, weight=-10.0),
        "alive": RewardTermCfg(func=is_alive, weight=0.15),
        "hip_pos": RewardTermCfg(
            func=hip_position,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot", joint_names=("Hip_A_.*", "Hip_R_.*")
                )
            },
            weight=-1.0,
        ),
        "dof_vel": RewardTermCfg(
            func=joint_vel_l2,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=(
                        "Hip_F_L",
                        "Hip_A_L",
                        "Hip_R_L",
                        "Knee_L",
                        "Ankle_P_L",
                        "Ankle_R_L",
                        "Hip_F_R",
                        "Hip_A_R",
                        "Hip_R_R",
                        "Knee_R",
                        "Ankle_P_R",
                        "Ankle_R_R",
                    ),
                    preserve_order=True,
                )
            },
            weight=-0.001,
        ),
        "upright": RewardTermCfg(func=flat_orientation_l2, weight=-1.0),
        "vertical_velocity": RewardTermCfg(func=vertical_velocity, weight=-2.0),
        "ang_vel_xy": RewardTermCfg(func=angular_velocity_xy, weight=-0.05),
        "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
        "joint_acceleration": RewardTermCfg(
            func=joint_acc_l2,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot", joint_names=("Hip_.*", "Knee_.*")
                )
            },
            weight=-2.5e-07,
        ),
        "torques": RewardTermCfg(
            func=joint_torques_l2,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot", joint_names=("Hip_.*", "Knee_.*")
                )
            },
            weight=-2e-06,
        ),
        "joint_limits": RewardTermCfg(func=joint_pos_limits, weight=-5.0),
        "stance_knee": RewardTermCfg(
            func=stance_knee,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot", joint_names=("Knee_L", "Knee_R"), preserve_order=True
                )
            },
            weight=-1.0,
        ),
        "overspeed": RewardTermCfg(func=overspeed, weight=-2.0),
    }
