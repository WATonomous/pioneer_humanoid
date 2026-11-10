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

__all__ = ["reset_block_center_uniform", "randomize_tray_pose", "check_gripper_tip_offset"]


def reset_block_center_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_exclusion: dict[str, float] | None = None,
    table_bounds: dict[str, float] | None = None,
    table_margin: float = 0.01,
    tray_pivot: tuple[float, float] = (0.0, 0.0),
    max_resample_iters: int = 12,
):
    """Reset the block to a random *center* pose, uniformly within the ranges.

    Like :func:`reset_root_state_uniform`, but samples the block's **geometric
    center** rather than its root. ``block.usd`` keeps the OBJ's corner origin
    (the rigid-body root sits at the mesh corner), so rotating the root about a
    sampled yaw would swing the center on a ~0.036 m circle and couple
    orientation into position. Here we instead sample the center offset and the
    yaw independently, then back-compute the corner root pose so the center
    lands exactly where sampled at any yaw.

    The offsets are sampled in the tray's NOMINAL frame, then carried with the tray: if ``randomize_tray_pose`` has moved the
    tray this reset (``env._tray_pose`` = dx, dy, dyaw per env), the sampled center and yaw are rotated about ``tray_pivot``
    and shifted by the same amount, so the block keeps its place relative to the tray.

    ``table_bounds`` (keys ``x_min``, ``x_max``, ``y_abs``, env frame; ``workcell.TABLE_SURFACE``) keeps the block's whole
    rotated footprint, plus ``table_margin``, on the real table surface after the tray shift: a sample whose footprint
    reaches past it is re-sampled like a box-exclusion hit.

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

    tray = env.__dict__.get("_tray_pose")
    tray = tray[env_ids] if tray is not None else torch.zeros(len(env_ids), 3, device=asset.device)
    pivot = torch.tensor(tray_pivot, device=asset.device)
    cos_t, sin_t = torch.cos(tray[:, 2]), torch.sin(tray[:, 2])

    def _carried(centers: torch.Tensor, yaws: torch.Tensor):
        """Centers and yaws as they end up once carried with the moved tray."""
        rel = centers[:, :2] - pivot
        xy = pivot + torch.stack([cos_t * rel[:, 0] - sin_t * rel[:, 1], sin_t * rel[:, 0] + cos_t * rel[:, 1]], dim=-1) + tray[:, :2]
        return xy, yaws + tray[:, 2]

    def _invalid(centers: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        bad = torch.zeros(len(centers), dtype=torch.bool, device=asset.device)
        if box_exclusion is not None:   # in the tray's own (nominal) frame
            bad |= (centers[:, 0] > box_exclusion["x_min"]) & (centers[:, 0] < box_exclusion["x_max"]) & (centers[:, 1].abs() < box_exclusion["y_abs"])
        if table_bounds is not None:    # on the table, after the tray has carried the block
            xy, yaw = _carried(centers, samples[:, 5])
            reach = BLOCK_HALF_SIZE * (torch.cos(yaw).abs() + torch.sin(yaw).abs()) + table_margin   # half-width of the rotated footprint
            bad |= (xy[:, 0] - reach < table_bounds["x_min"]) | (xy[:, 0] + reach > table_bounds["x_max"])
            bad |= xy[:, 1].abs() + reach > table_bounds["y_abs"]
        return bad

    # reject-and-resample any sample inside the tray footprint or reaching off the table
    invalid = _invalid(target_center, rand_samples)
    for _ in range(max_resample_iters):
        if not bool(invalid.any()):
            break
        n = int(invalid.sum())
        resample = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)
        rand_samples[invalid] = resample
        target_center[invalid] = default_center[invalid] + resample[:, 0:3]
        invalid = _invalid(target_center, rand_samples)
    # fallback: any still-invalid env gets the default center (offset 0, yaw 0), which is on the ramp approach, well inside the table
    if bool(invalid.any()):
        rand_samples[invalid, 0:3] = 0.0
        rand_samples[invalid, 5] = 0.0
        target_center[invalid] = default_center[invalid]

    # carry the block with the tray
    carried_xy, carried_yaw = _carried(target_center, rand_samples[:, 5])
    target_center = torch.cat([carried_xy, target_center[:, 2:3]], dim=-1)
    rand_samples[:, 5] = carried_yaw

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


def randomize_tray_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pos_range: float = 0.0,
    yaw_deg: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    pivot: tuple[float, float] = (0.0, 0.0),
):
    """Shift the tray by up to ``+-pos_range`` m in x and y and turn it by up to ``+-yaw_deg`` about ``pivot`` (its footprint centre).

    The tray is a kinematic rigid body, so this writes its pose through the physics API: both its collider and its mesh move.
    The shift is stored in ``env._tray_pose`` (dx, dy, dyaw per env) for ``reset_block_center_uniform`` to carry the block with
    it. Must run after the scene reset and before the block reset; with the default ranges of 0 it leaves the tray where it is.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    n = len(env_ids)
    d = math_utils.sample_uniform(-pos_range, pos_range, (n, 2), device=asset.device)
    dyaw = torch.deg2rad(math_utils.sample_uniform(-yaw_deg, yaw_deg, (n,), device=asset.device))
    buf = env.__dict__.setdefault("_tray_pose", torch.zeros(env.num_envs, 3, device=asset.device))
    buf[env_ids] = torch.cat([d, dyaw[:, None]], dim=-1)

    root = asset.data.default_root_state[env_ids].clone()   # env frame
    c, s = torch.cos(dyaw), torch.sin(dyaw)
    piv = torch.tensor(pivot, device=asset.device)
    rel = root[:, :2] - piv
    xy = piv + torch.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], dim=-1) + d
    zeros = torch.zeros(n, device=asset.device)
    quat = math_utils.quat_mul(math_utils.quat_from_euler_xyz(zeros, zeros, dyaw), root[:, 3:7])
    pose = torch.cat([xy, root[:, 2:3]], dim=-1) + env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(torch.cat([pose, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=asset.device), env_ids=env_ids)


_TIP_TOLERANCE_M = 0.002
_tip_local_cache: dict[str, tuple[float, float, float]] = {}


def _mesh_tip_local(robot, link_name: str) -> tuple[float, float, float]:
    """Finger tip in the finger link's own frame: the centroid of its mesh points within 3 mm of the distal end (lowest local z)."""
    import numpy as np
    import omni.usd
    from pxr import Usd, UsdGeom

    root = robot.cfg.prim_path.replace("env_.*", "env_0")
    link = omni.usd.get_context().get_stage().GetPrimAtPath(f"{root}/{link_name}")
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_to_link = xforms.GetLocalToWorldTransform(link).GetInverse()
    points = []
    for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):   # the visuals are instanced
        if prim.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(prim).GetPurposeAttr().Get() != UsdGeom.Tokens.guide:
            raw = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            to_link = np.array(xforms.GetLocalToWorldTransform(prim) * world_to_link)   # row-vector convention
            points.append((np.c_[raw, np.ones(len(raw))] @ to_link)[:, :3])
    points = np.vstack(points)
    distal = points[points[:, 2] < points[:, 2].min() + 0.003]
    return tuple(float(v) for v in distal.mean(axis=0))


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
