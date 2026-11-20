"""Event terms for the push-block task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .utils import BLOCK_HALF_SIZE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["hold_default_joint_targets", "reset_block_center_uniform"]


def reset_block_center_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_exclusion: dict[str, float] | None = None,
    max_resample_iters: int = 8,
):
    """Reset the block to a random *center* pose, uniformly within the ranges.

    Like :func:`reset_root_state_uniform`, but samples the block's **geometric
    center** rather than its root. ``block.usd`` keeps the OBJ's corner origin
    (the rigid-body root sits at the mesh corner), so rotating the root about a
    sampled yaw would swing the center on a ~0.036 m circle and couple
    orientation into position. Here we instead sample the center offset and the
    yaw independently, then back-compute the corner root pose so the center
    lands exactly where sampled at any yaw.

    ``pose_range`` keys (``x``, ``y``, ``z``, ``roll``, ``pitch``, ``yaw``) are
    offsets applied to the block's *default center*; missing keys default to 0.

    ``box_exclusion`` (optional) rejects sampled centers that fall inside the
    box's own footprint so a widened (curriculum) offset range never spawns the
    block inside the walls / on the elevated interior floor. Keys (env frame):
    ``x_min``, ``x_max``, ``y_abs``. Rejected envs are re-sampled up to
    ``max_resample_iters`` times, then any still-inside fall back to the default
    center (offset 0), which is on the ramp approach and always valid -- so the
    reset can never stall.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # default root (corner) state and orientation
    root_states = asset.data.default_root_state[env_ids].clone()
    default_corner = root_states[:, 0:3]
    default_quat = root_states[:, 3:7]

    half_vec = torch.full((len(env_ids), 3), BLOCK_HALF_SIZE, device=asset.device)
    # default center = corner + rotated half-diagonal (see utils.block_center_w)
    default_center = default_corner + math_utils.quat_apply(default_quat, half_vec)

    # sample center-pose offsets
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    target_center = default_center + rand_samples[:, 0:3]

    # reject-and-resample any center that lands inside the box footprint
    if box_exclusion is not None:
        x_min = box_exclusion["x_min"]
        x_max = box_exclusion["x_max"]
        y_abs = box_exclusion["y_abs"]

        def _inside(centers: torch.Tensor) -> torch.Tensor:
            return (centers[:, 0] > x_min) & (centers[:, 0] < x_max) & (centers[:, 1].abs() < y_abs)

        inside = _inside(target_center)
        for _ in range(max_resample_iters):
            if not bool(inside.any()):
                break
            n = int(inside.sum())
            resample = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)
            rand_samples[inside] = resample
            target_center[inside] = default_center[inside] + resample[:, 0:3]
            inside = _inside(target_center)
        # fallback: force any still-inside envs to the (always-valid) default center
        if bool(inside.any()):
            rand_samples[inside, 0:3] = 0.0
            target_center[inside] = default_center[inside]

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(default_quat, orientations_delta)
    # back out the corner root position that puts the center at target_center
    corner = target_center - math_utils.quat_apply(orientations, half_vec)
    positions = corner + env.scene.env_origins[env_ids]

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def hold_default_joint_targets(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """Set the PD targets of joints no action term drives to their default positions (else they track 0)."""
    robot = env.scene[asset_cfg.name]
    targets = robot.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    robot.set_joint_position_target(targets, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
