"""
armWithStand (wato_arm_v2) articulation config for teleoperation.

Robot model: Humanoid_Wato/wato_arm_v2 (armWithStand.usd).

CORRECTION (see git history for the original claim this replaced): despite
matching link/joint NAMES with wato_bimanual_arm, this is a DIFFERENT CAD
export with DIFFERENT per-joint rotation axes and origins -- confirmed by
diffing both URDFs' <joint><axis>/<origin> tags directly. e.g. joint3's axis
is (0,-1,0) in bimanual vs (0,0,-1) here; joint4 is (-1,0,0) vs (0,-1,0);
joint5 is (0,-1,0) vs (0,0,1); joint6 is (1,0,0) vs (0,-1,0). Every revolute
joint from 3 onward rotates about a different local axis than its bimanual
namesake. Do NOT assume any constant below transfers from bimanual_arm_cfg.py
just because the joint/link names match.

What this means for the values below:
  - _DEFAULT_JOINT_POS: reset to the URDF's own zero pose (0.0 rad on every
    revolute joint) instead of copying bimanual's tuned degree values --
    applying angles tuned for bimanual's axes to armWithStand's differently-
    oriented joints produced a visibly twisted/unnatural rest pose (this was
    the root cause of "joints moved in different directions than my real
    hand" reported during Quest teleop testing: the DLS IK solver itself
    uses the actual PhysX Jacobian, so it's axis-agnostic and was solving
    correctly, but starting from a contorted configuration makes the
    resulting motion look wrong). Zero is not necessarily the most ergonomic
    rest pose, but it's a defined, correct-by-construction one -- retune
    live via Physics Inspector once you have GUI access, same as bimanual's
    values originally were.
  - RIGHT_/LEFT_FINGER_DISTAL_TIP_LOCAL: re-measured directly from
    armWithStand's own link7/8/7l/8l mesh geometry (UsdGeom.BBoxCache
    ComputeRelativeBound, object-space) along its ACTUAL prismatic travel
    axis for joint7/8 (Y here, vs X in bimanual) -- see the "outward-Y
    extreme, X/Z bbox-center" values below. Bimanual's X-based numbers are
    wrong for this asset's joint7/8 axis.
  - JOINT_POS_LIMITS / actuator gains / GRIPPER_OPEN-CLOSED: UNCHANGED from
    bimanual_arm_cfg.py, i.e. still not independently verified. The
    revolute ±2π placeholder is universal (not orientation-dependent) so
    it's safe as-is; the gripper's ±0.05m stroke and actuator stiffness/
    damping/effort numbers are motor/mechanism-derived (same GL40 hardware
    presumed), not orientation-derived, so they're a reasonable carry-over
    but still worth confirming against the real mechanism.
  - _WRIST_ORIENT_OFFSET_LEFT/RIGHT in run_quest_armv2_teleop.py
    (not in this file) were reset to identity for the same reason as
    _DEFAULT_JOINT_POS -- bimanual's offsets were empirically tuned against
    its own joint6 axis and don't transfer. Needs live retuning once wrist
    rotation is tested in-headset (see that script's module docstring).

Joint limits/poses: Isaac Sim Physics Inspector (/World/bimanual_arm/root_joint)
Motor specs: https://watonomous.github.io/humanoid-docs/mechanical/index.html

  Shoulder joints 1-2  AK10-9 V3.0  18 Nm rated / 53 Nm peak
  Elbow joints 3-5     AK80-9 V3.0   9 Nm rated / 22 Nm peak
  Wrist joint 6        GL40 KV70     0.25 Nm rated / 0.73 Nm peak
  Gripper              GL40 KV70     0.25 Nm rated / 0.73 Nm peak
                       (rotary motor + linkage → prismatic finger travel in URDF)

NOTE: the "L"-suffixed URDF chain (joint1L..joint6l) is the robot's REAL,
CAN-actuated arm -- confirmed against real hardware. The mechanical/CAD
naming ("L" for what it called "left") was WRONG: this chain is physically
the robot's RIGHT arm. Corrected throughout this file; the unsuffixed chain
(joint1..joint6) is the LEFT arm (mirrored/held, not CAN-actuated).

In this module's own keyboard-teleop script, only the right arm is actuated;
the left arm is held at the Physics Inspector default pose below. Note that
run_quest_bimanual_teleop.py also imports BIMANUAL_ARM_CFG from here and
drives BOTH arms via Quest hand tracking — the "left_arm" ImplicitActuatorCfg
gains below were tuned for a held-still pose, not for continuous IK tracking,
so re-check them if the left arm feels sluggish or unresponsive under Quest
teleop.

Gripper actuation note
----------------------
On hardware, ONE GL40 rotary motor closes/opens the gripper through a
mechanical linkage (screw / hinge stack). The URDF instead exposes TWO
independent prismatic joints (joint7 + joint8, joint7l + joint8l).

In sim we:
  1. Drive both prismatic joints with synchronized position targets (open/closed pair).
  2. Use HIGH stiffness/damping so fingers stay locked during arm motion.
  3. Do NOT copy 0.25 Nm directly — effort_limit_sim on prismatic DOFs is a
     linear force cap (Newtons), not motor torque. Tune by grasp/hold behaviour.
"""
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_ARM_V2_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "Humanoid_Wato", "wato_arm_v2"))
_ARM_USD_PATH = os.path.join(_ARM_V2_ROOT, "urdf", "armWithStand", "armWithStand.usd")


def _deg(degrees: float) -> float:
    return degrees * math.pi / 180.0


# --- Joint limits from Physics Inspector ------------------------------------
_REVOLUTE_LIMIT = (-2 * math.pi, 2 * math.pi)
_JOINT7_LIMIT = (-0.05, 0.0)
_JOINT8_LIMIT = (0.0, 0.05)

JOINT_POS_LIMITS = {
    "joint1": _REVOLUTE_LIMIT,
    "joint2": _REVOLUTE_LIMIT,
    "joint3": _REVOLUTE_LIMIT,
    "joint4": _REVOLUTE_LIMIT,
    "joint5": _REVOLUTE_LIMIT,
    "joint6": _REVOLUTE_LIMIT,
    "joint7": _JOINT7_LIMIT,
    "joint8": _JOINT8_LIMIT,
    "joint1L": _REVOLUTE_LIMIT,
    "joint2l": _REVOLUTE_LIMIT,
    "joint3l": _REVOLUTE_LIMIT,
    "joint4l": _REVOLUTE_LIMIT,
    "joint5l": _REVOLUTE_LIMIT,
    "joint6l": _REVOLUTE_LIMIT,
    "joint7l": _JOINT7_LIMIT,
    "joint8l": _JOINT8_LIMIT,
}

# --- Default poses: URDF zero. Tried two alternatives live and both were
# rejected: (1) IK-solved "reach forward and up" target didn't converge
# cleanly (stuck ~0.2m error, joint angles past a full rotation); (2)
# directly reusing bimanual's tuned degree values looked plausibly bent in
# a narrow test crop but was confirmed wrong ("not naturally forward") once
# actually viewed live -- expected, since armWithStand's joint axes differ
# per-joint from bimanual's (see module docstring), so bimanual's angles
# don't correspond to the same physical configuration here. Back to zero
# as the known, correct-by-construction baseline. Retune live via Physics
# Inspector for a real ergonomic pose if wanted.
_DEFAULT_JOINT_POS = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.0,
    "joint8": 0.0,
    # Right arm — teleoperated
    "joint1L": 0.0,
    "joint2l": 0.0,
    "joint3l": 0.0,
    "joint4l": 0.0,
    "joint5l": 0.0,
    "joint6l": 0.0,
    "joint7l": 0.0,
    "joint8l": 0.0,
}

LEFT_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
RIGHT_ARM_JOINTS = ["joint1L", "joint2l", "joint3l", "joint4l", "joint5l", "joint6l"]
RIGHT_GRIPPER_JOINTS = ["joint7l", "joint8l"]
# Jacobian anchor is the wrist link; IK pose target is the fingertip center (see below).
RIGHT_EE_BODY = "link6l"
LEFT_EE_BODY = "link6"
RIGHT_FINGER_TIP_BODIES = ("link7l", "link8l")
LEFT_FINGER_TIP_BODIES = ("link7", "link8")
# Distal mesh points in each finger link's OWN object-space frame, measured
# via UsdGeom.BBoxCache.ComputeRelativeBound(link_prim, link_prim) against
# armWithStand's actual mesh geometry -- NOT bimanual's numbers (that asset's
# joint7/8 prismatic axis is X; this asset's is Y, see module docstring).
# Each finger's own outward-Y extreme (larger-magnitude Y bbox bound), X/Z
# bbox-center, same "outward extreme along the travel axis" methodology
# bimanual_arm_cfg.py used for its own (X-axis) fingers.
RIGHT_FINGER_DISTAL_TIP_LOCAL = {
    "link7l": (0.0, 0.0296, -0.0398),
    "link8l": (0.0, -0.0296, -0.0398),
}
LEFT_FINGER_DISTAL_TIP_LOCAL = {
    "link7": (0.0, -0.0296, -0.0398),
    "link8": (0.0, 0.0296, -0.0398),
}

# Gripper finger targets (joint7l: [-0.05, 0], joint8l: [0, 0.05]).
# VERIFIED empirically (not copied from bimanual) via headless sim: measured
# the actual fingertip-to-fingertip gap at both endpoints of the configured
# range. (joint7l=-0.05, joint8l=+0.05) -> gap=0.074m (closed).
# (joint7l=0, joint8l=0) -> gap=0.159m (open). This is the OPPOSITE labeling
# bimanual_arm_cfg.py uses for its own joint7l/8l -- expected, since this
# asset's joint7/8 prismatic axis is Y, bimanual's is X (see module
# docstring). Synchronized pair mimics single GL40 motor driving both
# fingers via linkage.
GRIPPER_OPEN = {"joint7l": 0.0, "joint8l": 0.0}
GRIPPER_CLOSED = {"joint7l": -0.05, "joint8l": 0.05}

# Prismatic gripper PD — tuned for hold during arm motion (not from motor datasheet).
# If fingers bounce when the shoulder moves, raise stiffness; if jittery, raise damping.
_GRIPPER_STIFFNESS = 400.0
_GRIPPER_DAMPING = 40.0
_GRIPPER_EFFORT_LIMIT = 30.0  # N (sim linear-force cap; tune empirically)
_GRIPPER_VELOCITY_LIMIT = 0.2  # m/s


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
    tip_bodies: tuple[str, ...] = RIGHT_FINGER_TIP_BODIES,
    tip_local: dict = RIGHT_FINGER_DISTAL_TIP_LOCAL,
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
    tip_bodies: tuple[str, ...] = RIGHT_FINGER_TIP_BODIES,
    tip_local: dict = RIGHT_FINGER_DISTAL_TIP_LOCAL,
):
    """Gripper-tip center pose in world frame (position from fingers, orientation from wrist)."""
    tip_pos_w = compute_gripper_tip_pos_w(robot, finger_body_ids, tip_bodies, tip_local)
    tip_quat_w = robot.data.body_quat_w[:, wrist_body_id]
    return tip_pos_w, tip_quat_w


def compute_gripper_tip_pose_b(
    robot, root_pose_w, wrist_body_id: int, finger_body_ids: list[int],
    tip_bodies: tuple[str, ...] = RIGHT_FINGER_TIP_BODIES,
    tip_local: dict = RIGHT_FINGER_DISTAL_TIP_LOCAL,
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
# clone() uses functools.wraps, which copies the INNER function's __name__
# ("_spawn_bimanual_arm_from_usd") onto this wrapped object. IsaacLab's config
# (de)serialization (callable_to_string / string_to_callable, used by the Hydra
# path in rsl_rl train.py) round-trips a callable as "module:__name__" — so the
# wrapper would serialize as ".._spawn_bimanual_arm_from_usd" and resolve BACK to
# the UNWRAPPED inner function, dropping the {ENV_REGEX_NS} path resolution the
# clone wrapper provides and crashing scene creation ("Path must be an absolute
# path: <>"). Restore the naming invariant the standard @clone decorator keeps
# (module-level name == __name__) so the round-trip resolves to THIS wrapper.
# Metadata-only: direct callers (teleop / pick_place / IL) are unaffected.
spawn_bimanual_arm_from_usd.__name__ = "spawn_bimanual_arm_from_usd"
spawn_bimanual_arm_from_usd.__qualname__ = "spawn_bimanual_arm_from_usd"


ARM_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_bimanual_arm_from_usd,
        usd_path=_ARM_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=_DEFAULT_JOINT_POS),
    actuators={
        # AK10-9 V3.0 — shoulder joints 1-2. effort_limit_sim uses the motor's
        # peak rating (53 Nm), not rated (18 Nm): at rated torque the shoulder
        # actuator saturates against gravity load when the arm is extended
        # (largest moment arm), causing the commanded IK target to visibly
        # undershoot on upward reach specifically — this is a simulation-only
        # cap (not applied to real hardware limits), so raising it is safe here.
        # NOTE: do NOT raise effort_limit_sim past the real peak rating (53 Nm)
        # to fix responsiveness — that would let the arm move in ways the real
        # motor can't, which matters for dataset collection (sim-to-real
        # mismatch). Stiffness/damping (below) is the safe knob instead: it
        # only changes how aggressively the PD controller commands torque
        # *within* that same real torque budget. Live testing showed the
        # shoulder visibly lagging the faster-responding elbow/wrist during
        # large reaches (diagnostic target_err in run_quest_bimanual_teleop.py
        # stayed elevated during active movement) — raised ~50% from the
        # original (which was tuned for holding a static pose, not tracking a
        # continuously moving IK target) to close that gap. Watch for shoulder
        # oscillation/overshoot if pushed further.
        "right_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint1L", "joint2l"],
            stiffness=2270.0,
            damping=180.0,
            effort_limit_sim=53.0,
            velocity_limit_sim=6.0,
        ),
        # AK80-9 V3.0 — elbow joints 3-5. Same rated (9 Nm) vs peak (22 Nm)
        # saturation concern as the shoulder, above. Raised alongside the
        # shoulder bump (same ~25% factor) — "forearm moves slower than it
        # should" reports track these joints directly, and this is the same
        # safe stiffness/damping knob (torque budget cap unchanged at 22 Nm).
        "right_elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint3l", "joint4l", "joint5l"],
            stiffness=1550.0,
            damping=110.0,
            effort_limit_sim=22.0,
            velocity_limit_sim=6.0,
        ),
        # GL40 KV70 — wrist joint 6
        "right_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint6l"],
            stiffness=341.0,
            damping=18.0,
            effort_limit_sim=0.25,
            velocity_limit_sim=6.0,
        ),
        # GL40 KV70 rotary → linkage → two prismatic fingers (see module docstring)
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint7l", "joint8l"],
            stiffness=_GRIPPER_STIFFNESS,
            damping=_GRIPPER_DAMPING,
            effort_limit_sim=_GRIPPER_EFFORT_LIMIT,
            velocity_limit_sim=_GRIPPER_VELOCITY_LIMIT,
        ),
        # Left arm — mirrors right_shoulder/right_elbow/right_wrist above.
        # Previously one coarse "left_arm" group at uniform stiffness=1000/
        # damping=100/effort=18 for all 6 joints, tuned for holding a static
        # pose (this file's own keyboard-teleop script never actuates the
        # left arm). run_quest_bimanual_teleop.py drives both arms via Quest
        # hand tracking, so the left arm needs the same per-joint-type torque
        # headroom as the right, especially the shoulder on upward reach (see
        # note above).
        "left_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2"],
            stiffness=2270.0,
            damping=180.0,
            effort_limit_sim=53.0,
            velocity_limit_sim=6.0,
        ),
        "left_elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint3", "joint4", "joint5"],
            stiffness=1550.0,
            damping=110.0,
            effort_limit_sim=22.0,
            velocity_limit_sim=6.0,
        ),
        "left_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint6"],
            stiffness=341.0,
            damping=18.0,
            effort_limit_sim=0.25,
            velocity_limit_sim=6.0,
        ),
        # Left gripper — same coupled-prismatic hold as right
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint7", "joint8"],
            stiffness=_GRIPPER_STIFFNESS,
            damping=_GRIPPER_DAMPING,
            effort_limit_sim=_GRIPPER_EFFORT_LIMIT,
            velocity_limit_sim=_GRIPPER_VELOCITY_LIMIT,
        ),
    },
)
