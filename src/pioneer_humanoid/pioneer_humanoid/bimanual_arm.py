"""
Canonical articulation config for the pioneer_bimanual_arm.

Robot model: assets/pioneer_bimanual_arm (pioneer_bimanual_arm.usd),
formerly wato_arm_v2/armWithStand.usd. This replaces the old bimanual_arm_cfg.py,
which used a reversed L/R convention, inverted gripper dicts, X-axis fingertip
geometry (wrong for this asset), and a rest pose tuned for a superseded CAD
export.

Naming: LEFT_* = the L-suffixed chain (joint1L..joint6l) = physical LEFT arm,
matching the CAD. RIGHT_* = unsuffixed (joint1..joint6). An earlier revision
had these reversed; don't reintroduce that.

Notes on the values below:
  - _DEFAULT_JOINT_POS is this URDF's zero pose plus elbow flex (see below), a
    correct-by-construction rest pose -- not a hand-tuned one.
  - LEFT_/RIGHT_FINGER_DISTAL_TIP_LOCAL are measured against this asset's mesh,
    along its joint7/8 travel axis (Y for this export).
  - _WRIST_ORIENT_OFFSET_* in run_quest_bimanual_teleop.py are identity and need
    live in-headset retuning for this asset's joint6 axis.
  - actuator gains and the +/-0.05 m gripper stroke are motor/mechanism-derived
    but not independently verified.

Motor specs: https://watonomous.github.io/humanoid-docs/mechanical/index.html
  Shoulder joints 1-2  AK10-9 V3.0  18 Nm rated / 53 Nm peak
  Elbow joints 3-5     AK80-9 V3.0   9 Nm rated / 22 Nm peak
  Wrist joint 6        GL40 KV70     0.25 Nm rated / 0.73 Nm peak
  Gripper              GL40 KV70     one rotary motor -> linkage -> two prismatic
                       finger joints (joint7+joint8). We drive the pair with
                       synchronized targets and high stiffness/damping so the
                       fingers stay locked during arm motion. effort_limit_sim on
                       a prismatic DOF is a force cap (N), not torque -- tune by
                       grasp behaviour, not the 0.25 Nm rating.

Actuator groups were tuned for holding a static pose -- recheck stiffness/damping
if an arm feels sluggish tracking a moving IK target.
"""
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg

from .arm_params import (  # noqa: F401 -- re-exported for existing call sites
    ACTUATOR_GROUPS,
    CAMERA_NAMES,
    CAMERAS,
    DEFAULT_JOINT_POS,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_CLOSED,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_OPEN,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_CLOSED,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_OPEN,
    vertical_aperture,
)
from .urdf_joint_limits import JOINT_POS_LIMITS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARM_ROOT = str(_REPO_ROOT / "assets" / "pioneer_bimanual_arm")
_ARM_USD_PATH = str(Path(_ARM_ROOT) / "usd" / "pioneer_bimanual_arm.usd")


# --- Joint limits -----------------------------------------------------------
# Re-exported from urdf_joint_limits (parses the URDF) so `from pioneer_humanoid.bimanual_arm import
# JOINT_POS_LIMITS` call sites keep working. apply_joint_limits() and
# patch_joint_pos_limits_on_prim() below write it onto the spawned prims, so this is what the
# joints actually enforce at runtime. Replaced an unverified +/-2pi (no-limit) placeholder.

# --- Default (spawn) pose, joint names, gripper endpoints and actuator gains: arm_params.py
# (simulator-neutral, shared with mujoco_bimanual_arm.py).
#
# NOTE: InitialStateCfg.joint_pos only sets robot.data.default_joint_pos; PhysX still spawns at
# the USD zero pose and the implicit actuators barely move it. run_simulator writes the state
# explicitly each step -- an offline check must do the same or it measures the arms hanging down.

# Jacobian anchor is the wrist link; IK pose target is the fingertip center (see below).
LEFT_EE_BODY = "link6l"
RIGHT_EE_BODY = "link6"
LEFT_FINGER_TIP_BODIES = ("link7l", "link8l")
RIGHT_FINGER_TIP_BODIES = ("link7", "link8")
# Distal mesh point of each finger, in that link's OWN object-space frame: the outward-Y extreme
# (larger-magnitude Y bbox bound) with X/Z at the bbox centre, measured via
# UsdGeom.BBoxCache.ComputeRelativeBound against this asset's mesh. Same "outward extreme along
# the travel axis" methodology bimanual_arm_cfg.py used, but along Y -- bimanual's joint7/8 axis
# is X, so its numbers are wrong here (see module docstring).
#
# LINK-LOCAL, so they do not survive a re-export that moves the link origins; re-measure the same
# way if the arm asset changes. Sanity check: composed with the joint7/8 origins these give a
# 0.1625m open fingertip gap on both grippers, matching the empirically measured 0.161m below.
LEFT_FINGER_DISTAL_TIP_LOCAL = {
    "link7l": (0.0, 0.151416, -0.045369),
    "link8l": (0.0, -0.114416, -0.039825),
}
RIGHT_FINGER_DISTAL_TIP_LOCAL = {
    "link7": (0.0, -0.151416, -0.045369),
    "link8": (0.0, 0.114416, -0.039825),
}


def _joint_limit_key(name: str) -> str | None:
    """Map a USD/articulation joint name to a JOINT_POS_LIMITS key."""
    if name in JOINT_POS_LIMITS:
        return name
    if len(name) > 1 and name[-1] in ("l", "L"):
        alt = name[:-1] + ("L" if name[-1] == "l" else "l")
        if alt in JOINT_POS_LIMITS:
            return alt
    return None


def apply_joint_limits(robot) -> None:
    """Apply Physics Inspector joint limits to the articulation."""
    limits = robot.data.joint_pos_limits.clone()
    updated = []

    for joint_idx, joint_name in enumerate(robot.data.joint_names):
        limit_key = _joint_limit_key(joint_name)
        if limit_key is None:
            continue
        lo, hi = JOINT_POS_LIMITS[limit_key]
        limits[:, joint_idx, 0] = lo
        limits[:, joint_idx, 1] = hi
        updated.append(joint_name)

    if updated:
        robot.write_joint_position_limit_to_sim(limits, warn_limit_violation=False)
        print(f"[INFO] Applied joint limits for {len(updated)} joints.")


def resolve_body_ids(robot, names: tuple[str, ...] | list[str]) -> list[int]:
    """Map body names to articulation body indices."""
    body_names = list(robot.data.body_names)
    name_to_id = {name: idx for idx, name in enumerate(body_names)}
    missing = [name for name in names if name not in name_to_id]
    if missing:
        raise KeyError(f"Body names {missing} not found in {body_names}")
    return [name_to_id[name] for name in names]


def compute_gripper_tip_pos_w(
    robot, finger_body_ids: list[int],
    tip_bodies: tuple[str, ...] = LEFT_FINGER_TIP_BODIES,
    tip_local: dict = LEFT_FINGER_DISTAL_TIP_LOCAL,
):
    """World-frame midpoint between the distal tips of the two gripper fingers."""
    import torch
    from isaaclab.utils.math import quat_apply

    dtype = robot.data.body_pos_w.dtype
    device = robot.data.body_pos_w.device
    tips = []
    for body_name, body_id in zip(tip_bodies, finger_body_ids):
        local = torch.tensor([tip_local[body_name]], device=device, dtype=dtype)
        body_pos = robot.data.body_pos_w[:, body_id]
        body_quat = robot.data.body_quat_w[:, body_id]
        tips.append(body_pos + quat_apply(body_quat, local))
    return (tips[0] + tips[1]) * 0.5


def compute_gripper_tip_pose_w(
    robot, wrist_body_id: int, finger_body_ids: list[int],
    tip_bodies: tuple[str, ...] = LEFT_FINGER_TIP_BODIES,
    tip_local: dict = LEFT_FINGER_DISTAL_TIP_LOCAL,
):
    """Gripper-tip center pose in world frame (position from fingers, orientation from wrist)."""
    tip_pos_w = compute_gripper_tip_pos_w(robot, finger_body_ids, tip_bodies, tip_local)
    tip_quat_w = robot.data.body_quat_w[:, wrist_body_id]
    return tip_pos_w, tip_quat_w


def compute_gripper_tip_pose_b(
    robot, root_pose_w, wrist_body_id: int, finger_body_ids: list[int],
    tip_bodies: tuple[str, ...] = LEFT_FINGER_TIP_BODIES,
    tip_local: dict = LEFT_FINGER_DISTAL_TIP_LOCAL,
):
    """Gripper-tip center pose in the robot root frame."""
    from isaaclab.utils.math import subtract_frame_transforms

    tip_pos_w, tip_quat_w = compute_gripper_tip_pose_w(
        robot, wrist_body_id, finger_body_ids, tip_bodies, tip_local
    )
    return subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], tip_pos_w, tip_quat_w
    )


def jacobian_world_to_root(robot, jacobian_w):
    """Rotate PhysX world-frame Jacobian into the articulation root frame."""
    import torch
    from isaaclab.utils.math import matrix_from_quat, quat_inv

    base_rot = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, :3, :] = torch.bmm(base_rot, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(base_rot, jacobian_b[:, 3:, :])
    return jacobian_b


def adjust_jacobian_for_gripper_tip(jacobian_b, wrist_pos_b, tip_pos_b):
    """Map link6l root-frame Jacobian to the fingertip center."""
    import torch
    from isaaclab.utils.math import skew_symmetric_matrix

    offset_b = tip_pos_b - wrist_pos_b
    tip_jacobian = jacobian_b.clone()
    tip_jacobian[:, 0:3, :] += torch.bmm(-skew_symmetric_matrix(offset_b), jacobian_b[:, 3:, :])
    return tip_jacobian


def compute_tip_ik_jacobian(robot, jacobian_w, wrist_pos_b, tip_pos_b):
    """World-frame link6l Jacobian -> root frame -> fingertip center."""
    return adjust_jacobian_for_gripper_tip(
        jacobian_world_to_root(robot, jacobian_w), wrist_pos_b, tip_pos_b
    )


def resolve_joint_name(robot, name: str) -> str:
    """Match a config joint name to the articulation's actual joint name."""
    names = list(robot.data.joint_names)
    if name in names:
        return name
    # Physics Inspector may show joint2L while URDF uses joint2l
    if len(name) > 1 and name[-1] in ("l", "L"):
        alt = name[:-1] + ("L" if name[-1] == "l" else "l")
        if alt in names:
            return alt
    raise KeyError(f"Joint '{name}' not found in {names}")


def patch_joint_pos_limits_on_prim(prim_path: str) -> None:
    """Write Physics Inspector limits onto spawned USD joints before articulation init."""
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Usd, UsdPhysics

    stage = stage_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return

    updated = []
    for prim in Usd.PrimRange(root):
        limit_key = _joint_limit_key(prim.GetName())
        if limit_key is None:
            continue
        lo, hi = JOINT_POS_LIMITS[limit_key]
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            # USD RevoluteJoint limits are in degrees; JOINT_POS_LIMITS uses radians.
            lo, hi = math.degrees(lo), math.degrees(hi)
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint = UsdPhysics.PrismaticJoint(prim)
        else:
            continue
        joint.GetLowerLimitAttr().Set(lo)
        joint.GetUpperLimitAttr().Set(hi)
        updated.append(prim.GetName())

    if updated:
        print(f"[INFO] Patched USD joint limits for {len(updated)} joints under {prim_path}.")


def _spawn_bimanual_arm_from_usd(prim_path, cfg, translation=None, orientation=None):
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    patch_joint_pos_limits_on_prim(prim_path)
    return prim


from isaaclab.sim.utils import clone  # noqa: E402

spawn_bimanual_arm_from_usd = clone(_spawn_bimanual_arm_from_usd)
# functools.wraps (via clone()) copies the inner __name__ onto this wrapper. IsaacLab config
# (de)serialization round-trips a callable as "module:__name__", so it would resolve back to
# the unwrapped inner function, losing the {ENV_REGEX_NS} resolution and crashing scene
# creation. Restore module-name == __name__ so the round-trip lands on this wrapper.
# Metadata-only; direct callers are unaffected.
spawn_bimanual_arm_from_usd.__name__ = "spawn_bimanual_arm_from_usd"
spawn_bimanual_arm_from_usd.__qualname__ = "spawn_bimanual_arm_from_usd"


BIMANUAL_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_bimanual_arm_from_usd,
        usd_path=_ARM_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # OFF is what makes grasping work: each finger's CAD collider is a convex hull
            # bulging into the jaw gap, so ON the two hulls hold the jaw wider than the box
            # and only one finger contacts it. Re-enable once the colliders are real primitives.
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=DEFAULT_JOINT_POS),
    actuators={
        name: ImplicitActuatorCfg(
            joint_names_expr=group["joints"],
            stiffness=group["stiffness"],
            damping=group["damping"],
            effort_limit_sim=group["effort_limit"],
            velocity_limit_sim=group["velocity_limit"],
        )
        for name, group in ACTUATOR_GROUPS.items()
    },
)


def make_camera_cfg(name: str, height: int, width: int) -> CameraCfg:
    """Isaac CameraCfg for robot camera ``name`` (arm_params.CAMERAS) at height x width.

    Vertical aperture follows the image aspect. Requires --enable_cameras. Prim paths assume the
    robot is spawned at ``{ENV_REGEX_NS}/Robot``; cameras stay out of the arm asset.
    """
    cam = CAMERAS[name]
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + f"{cam.body}/{cam.prim}",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam.focal,
            horizontal_aperture=cam.h_aperture,
            vertical_aperture=vertical_aperture(name, height, width),
            clipping_range=cam.clip,
        ),
        offset=CameraCfg.OffsetCfg(pos=cam.pos, rot=cam.rot, convention="opengl"),
        height=height,
        width=width,
        update_period=0.0,
        data_types=["rgb"],
    )
