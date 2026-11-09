"""Reward terms for pushing the block up the ramp onto the box's interior floor.

All positions are expressed in the per-env frame (world minus ``env_origins``);
the push direction is the env-frame ``+x`` axis (the box is spawned with its
ramp's up-direction aligned to ``+x``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .utils import BLOCK_HALF_SIZE, block_center_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def block_to_target(
    env: ManagerBasedRLEnv,
    temp: float,
    target: tuple[float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """``exp(-temp * d)``, where d is the xy distance from the block center to ``target``."""
    object: RigidObject = env.scene[object_cfg.name]
    center = block_center_w(object) - env.scene.env_origins
    target_t = torch.tensor(target, device=env.device)
    distance = torch.norm(center[:, :2] - target_t, dim=1)
    return torch.exp(-temp * distance)

def push_point_w(
    env: ManagerBasedRLEnv,
    push_offset: float,
    target: tuple[float, float],
    angle_gain: float,
    max_angle_deg: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """World-frame point ``push_offset`` m from the block center (at its height) that the fingertips should sit at.
    N is the angle from the centerline (+x) to the block->``target`` direction. The push comes along ``angle_gain * N``
    (clamped to ``max_angle_deg``), so a block off the centerline is pushed back toward it harder than the straight
    line to the target. N = 0 puts the point straight behind the block."""
    object: RigidObject = env.scene[object_cfg.name]
    center = block_center_w(object)
    target_w = env.scene.env_origins[:, :2] + torch.tensor(target, device=env.device)
    to_target = target_w - center[:, :2]
    n = torch.atan2(to_target[:, 1], to_target[:, 0].clamp(min=1e-3))
    limit = math.radians(max_angle_deg)
    angle = (angle_gain * n).clamp(-limit, limit)
    push_point = center
    push_point[:, 0] -= push_offset * torch.cos(angle)
    push_point[:, 1] -= push_offset * torch.sin(angle)
    return push_point


def ee_behind_block(
    env: ManagerBasedRLEnv,
    temp: float,
    push_offset: float,
    target: tuple[float, float],
    angle_gain: float,
    max_angle_deg: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """``exp(-temp * d)``, where d is the 3D distance from the end-effector to ``push_point_w``."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    push_point = push_point_w(env, push_offset, target, angle_gain, max_angle_deg, object_cfg)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(ee_w - push_point, dim=1)
    return torch.exp(-temp * distance)

def block_on_floor(
    env: ManagerBasedRLEnv,
    x_min: float,
    x_max: float,
    y_half: float,
    floor_z: float,
    z_tol: float,
    rest_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Success flag: 1.0 on steps where the block is on the interior floor, else 0.0.
    All four must hold: center inside the x/y footprint, at floor height (within ``z_tol``),
    and moving slower than ``rest_speed``. Sparse, so it stays 0 until the task is solved."""
    object: RigidObject = env.scene[object_cfg.name]
    center = block_center_w(object) - env.scene.env_origins
    in_x = (center[:, 0] > x_min) & (center[:, 0] < x_max)
    in_y = center[:, 1].abs() < y_half
    on_floor = (center[:, 2] - (floor_z + BLOCK_HALF_SIZE)).abs() < z_tol
    at_rest = torch.norm(object.data.root_lin_vel_w, dim=1) < rest_speed
    return (in_x & in_y & on_floor & at_rest).float()

def action_smoothness(env: ManagerBasedRLEnv, temp: float) -> torch.Tensor:
    """``1 - exp(-temp * d)``, where d is the norm of this step's action minus the last step's.
    0 for a perfectly smooth policy, rising toward 1 (bounded) as the action jumps around.
    Meant for a negative weight, so it penalizes jitter without paying the policy for standing still."""
    action_change = torch.norm(env.action_manager.action - env.action_manager.prev_action, dim=1)
    return 1.0 - torch.exp(-temp * action_change)
