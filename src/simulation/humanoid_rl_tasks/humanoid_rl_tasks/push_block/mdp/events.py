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

__all__ = ["reset_block_center_uniform", "check_gripper_tip_offset"]


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


_TIP_TOLERANCE_M = 0.002
_tip_local_cache: dict[str, tuple[float, float, float]] = {}


def _mesh_tip_local(robot, link_name: str) -> tuple[float, float, float]:
    """Finger tip in the finger link's own frame: outward-Y extreme of its mesh bounding box, X/Z at the center."""
    import omni.usd
    from pxr import Usd, UsdGeom

    root = robot.cfg.prim_path.replace("env_.*", "env_0")
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(f"{root}/{link_name}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    box = cache.ComputeRelativeBound(prim, prim).ComputeAlignedRange()
    lo, hi = box.GetMin(), box.GetMax()
    y = hi[1] if abs(hi[1]) > abs(lo[1]) else lo[1]
    return (0.5 * (lo[0] + hi[0]), y, 0.5 * (lo[2] + hi[2]))


def check_gripper_tip_offset(env: ManagerBasedEnv, env_ids: torch.Tensor | None):
    """Print one red line if the fingertip midpoint, measured from the finger meshes, is more than 2 mm
    from ``LEFT_GRIPPER_TIP_OFFSET``. The line contains the value to paste and the file it lives in."""
    from pioneer_humanoid import arm_params
    from pioneer_humanoid.bimanual_arm import LEFT_EE_BODY, LEFT_FINGER_TIP_BODIES

    robot = env.scene["robot"]
    pos, quat = robot.data.body_pos_w[0], robot.data.body_quat_w[0]
    try:
        tips_w = []
        for name in LEFT_FINGER_TIP_BODIES:
            if name not in _tip_local_cache:
                _tip_local_cache[name] = _mesh_tip_local(robot, name)
            body_id = robot.find_bodies(name)[0][0]
            local = torch.tensor(_tip_local_cache[name], device=pos.device, dtype=pos.dtype)
            tips_w.append(pos[body_id] + math_utils.quat_apply(quat[body_id], local))
    except Exception as err:
        print(f"\033[91m[gripper tip] cannot measure the fingertips from the meshes: {err}\033[0m", flush=True)
        return
    wrist_id = robot.find_bodies(LEFT_EE_BODY)[0][0]
    measured = math_utils.quat_apply(math_utils.quat_inv(quat[wrist_id]), 0.5 * (tips_w[0] + tips_w[1]) - pos[wrist_id])
    expected = torch.tensor(arm_params.LEFT_GRIPPER_TIP_OFFSET, device=pos.device, dtype=pos.dtype)
    error = torch.norm(measured - expected).item()
    if error > _TIP_TOLERANCE_M:
        x, y, z = measured.tolist()
        print(
            f"\033[91m[gripper tip] off by {error * 1000:.1f} mm: set LEFT_GRIPPER_TIP_OFFSET = ({x:.5f}, {y:.5f}, {z:.5f}) "
            f"in {arm_params.__file__}\033[0m",
            flush=True,
        )
