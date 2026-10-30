"""Simulator-neutral pioneer_bimanual_arm parameters: joints, default pose, gripper, actuator gains, cameras.

Plain Python (no Isaac / MuJoCo imports) so both bimanual_arm.py (Isaac Lab) and
mujoco_bimanual_arm.py (MuJoCo) build from the same numbers. Naming: LEFT_* = the L-suffixed
chain (joint1L..joint6l) = physical LEFT arm; RIGHT_* = unsuffixed.
"""
import math
from typing import NamedTuple



def _deg(degrees: float) -> float:
    return degrees * math.pi / 180.0


# --- Default (spawn / home) pose: URDF zero (arms hanging), except the elbows flexed to +/-90 deg.
#
# Also the leader-teleop home: forearm forward, above the table, so returning to it never sweeps
# through the table. URDF zero = the real arm's and the leader's calibrated hanging pose.
#
# At URDF zero both arms hang straight down at the elbow EXTENSION SINGULARITY (manipulability
# ~2e-06, cond(J) ~2560), with the least-controllable direction almost exactly the +Z that
# _HOME_TIP_Z_OFFSET then commands -- so the first IK step is huge, hikes the shoulder, and
# picks an elbow-bend direction at random. Flexing the elbows raises manipulability ~10,000x
# and starts the arm in the flexion branch.
#
# Signs are OPPOSITE (joint4/joint4l axes are (0,-1,0)/(0,1,0)); both put the forearm forward
# toward +X. Opposite-and-equal is what makes the pose mirror-symmetric. 90 holds the forearm
# level (fingertip ~10cm higher than the previous 75); don't go below ~60 (tip nears the table).
# Both inside the URDF limits (42.5 deg margin to joint4l's 132.5).
DEFAULT_JOINT_POS = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": _deg(90.0),
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.0,
    "joint8": 0.0,
    # Left arm (the L-suffixed chain)
    "joint1L": 0.0,
    "joint2l": 0.0,
    "joint3l": 0.0,
    "joint4l": _deg(-90.0),
    "joint5l": 0.0,
    "joint6l": 0.0,
    "joint7l": 0.0,
    "joint8l": 0.0,
}

LEFT_ARM_JOINTS = ["joint1L", "joint2l", "joint3l", "joint4l", "joint5l", "joint6l"]
RIGHT_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
LEFT_GRIPPER_JOINTS = ["joint7l", "joint8l"]
RIGHT_GRIPPER_JOINTS = ["joint7", "joint8"]

# Gripper finger targets, verified empirically in headless Isaac sim: fingertip gap 0.074m
# closed / ~0.16m open at these endpoints. This asset's joint7/8 axis is Y.
LEFT_GRIPPER_OPEN = {"joint7l": 0.0, "joint8l": 0.0}
LEFT_GRIPPER_CLOSED = {"joint7l": -0.05, "joint8l": 0.05}
RIGHT_GRIPPER_OPEN = {"joint7": 0.0, "joint8": 0.0}
RIGHT_GRIPPER_CLOSED = {"joint7": -0.05, "joint8": 0.05}

# Midpoint of the two fingertips in the wrist (link6l) frame, m, gripper at 0.0. Checked at startup by
# check_gripper_tip_offset (humanoid_rl_tasks push_block/mdp/events.py), which prints the value to paste if it drifts.
LEFT_GRIPPER_TIP_OFFSET = (0.0, 0.053, -0.15638)

# Prismatic gripper PD — tuned for hold during arm motion (not from motor datasheet).
# If fingers bounce when the shoulder moves, raise stiffness; if jittery, raise damping.
_GRIPPER = dict(stiffness=400.0, damping=40.0, effort_limit=30.0, velocity_limit=0.2)  # N, m/s

# Sim-only joint speed cap shared by every revolute arm joint (rad/s); the gripper has its own in _GRIPPER.
ARM_JOINT_VELOCITY_LIMIT = _deg(60)

# Joint PD groups: stiffness (Nm/rad or N/m), damping, effort_limit (Nm or N), velocity_limit.
# effort_limit is a sim-only cap at the motor's PEAK torque (rated saturates against gravity
# with the arm extended). Do NOT raise it past the real peak (sim-to-real mismatch for dataset
# collection) -- use stiffness/damping, which command torque harder within the same budget.
ACTUATOR_GROUPS = {
    # AK10-9 V3.0 — shoulder joints 1-2 (18 Nm rated / 53 peak). Stiffness/damping raised ~50%
    # from the original static-hold tuning to stop the shoulder lagging on large reaches; watch
    # for overshoot if pushed further.
    "left_shoulder": dict(joints=["joint1L", "joint2l"], stiffness=2270.0, damping=180.0, effort_limit=53.0, velocity_limit=ARM_JOINT_VELOCITY_LIMIT),
    # AK80-9 V3.0 — elbow joints 3-5 (9 rated / 22 peak); raised with the shoulder for "forearm too slow".
    "left_elbow": dict(joints=["joint3l", "joint4l", "joint5l"], stiffness=1550.0, damping=110.0, effort_limit=22.0, velocity_limit=ARM_JOINT_VELOCITY_LIMIT),
    # GL40 KV70 — wrist joint 6 (0.25 rated / 0.73 peak). Rated saturated instantly in Isaac;
    # don't exceed 0.73 -- raise stiffness instead.
    "left_wrist": dict(joints=["joint6l"], stiffness=341.0, damping=18.0, effort_limit=0.73, velocity_limit=ARM_JOINT_VELOCITY_LIMIT),
    # GL40 KV70 rotary -> linkage -> two prismatic fingers, driven with synchronized targets.
    # effort_limit on a prismatic DOF is a force cap (N) -- tune by grasp, not the motor rating.
    "left_gripper": dict(joints=["joint7l", "joint8l"], **_GRIPPER),
    # Right arm (unsuffixed) mirrors the left: both arms are teleoperated and need equal headroom.
    "right_shoulder": dict(joints=["joint1", "joint2"], stiffness=2270.0, damping=180.0, effort_limit=53.0, velocity_limit=ARM_JOINT_VELOCITY_LIMIT),
    "right_elbow": dict(joints=["joint3", "joint4", "joint5"], stiffness=1550.0, damping=110.0, effort_limit=22.0, velocity_limit=ARM_JOINT_VELOCITY_LIMIT),
    "right_wrist": dict(joints=["joint6"], stiffness=341.0, damping=18.0, effort_limit=0.73, velocity_limit=ARM_JOINT_VELOCITY_LIMIT),
    "right_gripper": dict(joints=["joint7", "joint8"], **_GRIPPER),
}


# --- Cameras (poses and lenses from PR #296's camera USD) -------------------------------------
# Poses are relative to the parent link, OpenGL convention (camera looks along -Z, +Y up), which
# both simulators use.


class CameraMount(NamedTuple):
    body: str                 # parent link
    prim: str                 # Isaac prim name under the link
    pos: tuple                # m, link frame
    rot: tuple                # wxyz, link frame, OpenGL convention
    focal: float              # mm
    h_aperture: float         # mm
    clip: tuple               # near, far (m)
    aspect: float | None      # sensor w/h the image must keep, or None


# ego          RealSense D455 colour at 640x480 (4:3), 50 deg down. ESTIMATE ~80 x 65 deg: native
#              1280x800 lens with the sides cropped to 4:3. Replace with the real camera_info.
# wrist_left   on the camera bracket plate of link6l: centre of its 4-hole pattern (24 x 25 mm) on face 1 plus 0.1 mm,
#              from meshes/link6l.STL. The plate is tilted 20 deg, so the optical axis is the plate normal: pitch 20 deg,
#              roll 270. Lens: 120 deg DIAGONAL FOV on the 4:3 sensor (about 108 x 92 deg), a deliberate choice.
#              The real camera calibrates to ~74 deg horizontal on its raw 1920x1080 frame, and real frames are remapped
#              to the pinhole in calib/wrist_cam_pinhole_640x480.yaml (47.75 x 36.73 deg), so sim and real no longer
#              share a camera here: change the remap target (perception/camera_calibration) before sim2real.
# wrist_right  mirror of wrist_left (old pose and lens, unused)
WRIST_DIAG_FOV_DEG = 120.0
WRIST_FOCAL_MM = math.hypot(20.955, 20.955 * 3 / 4) / (2.0 * math.tan(math.radians(WRIST_DIAG_FOV_DEG / 2.0)))
CAMERAS = {
    "ego": CameraMount(
        "base_link", "cam_ego",
        (0.08421, -0.00008, 0.26038), (0.664463, 0.241845, -0.241845, -0.664463),
        1.93, 3.896 * (800 * 4 / 3) / 1280, (0.01, 100.0), 4 / 3,
    ),
    "wrist_left": CameraMount(
        "link6l", "cam_wrist_left",
        (0.062138, 0.053, -0.068179), (0.696364, -0.122788, 0.122788, -0.696364),
        WRIST_FOCAL_MM, 20.955, (0.01, 100.0), 4 / 3,
    ),
    "wrist_right": CameraMount(
        "link6", "cam_wrist_right",
        (0.06179, -0.04397, -0.07077), (0.696364, -0.122788, 0.122788, -0.696364),
        WRIST_FOCAL_MM, 20.955, (0.05, 5.0), 4 / 3,
    ),
}
CAMERA_NAMES = tuple(CAMERAS)


def vertical_aperture(name: str, height: int, width: int) -> float:
    """Vertical aperture (mm) at height x width: follows the image aspect, no stretch.

    A camera modelled on a real sensor must keep that sensor's aspect, so sim sees what the real one sees."""
    cam = CAMERAS[name]
    if cam.aspect is not None and abs(width / height - cam.aspect) > 0.02 * cam.aspect:
        raise ValueError(f"camera {name}: {width}x{height} must match its sensor aspect {cam.aspect:.3g} (w/h)")
    return cam.h_aperture * height / width


def vertical_fov_deg(name: str, height: int, width: int) -> float:
    return math.degrees(2.0 * math.atan(vertical_aperture(name, height, width) / (2.0 * CAMERAS[name].focal)))
