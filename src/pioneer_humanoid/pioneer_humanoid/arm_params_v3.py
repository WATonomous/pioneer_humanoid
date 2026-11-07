"""pioneer_bimanual_arm v3 parameters: the 7-DOF arms (wrist pitch + yaw on CubeMars AK45-10, gripper on an AK40-10).

Same structure as arm_params.py (v2: one GL40 wrist joint, GL40 gripper), for
assets/pioneer_bimanual_arm/urdf/pioneer_bimanual_arm_v3.urdf (README_v3.md there). Simulator-neutral plain Python.

Joints per arm (right unsuffixed, left as the team's names joint1L, joint2l, ...):
  joint1-2  shoulder         AK10-9 V3.0   (unchanged)
  joint3-5  upper arm, elbow, forearm roll  AK80-9 V3.0 (unchanged; joint5's origin now on the AK80 output face)
  joint6    wrist pitch      AK45-10 V3.0  axis -y (right) / +y (left), +-90 deg   (was the GL40 wrist)
  joint7    wrist yaw        AK45-10 V3.0  axis +x, +-40 deg                        (NEW)
  joint8-9  the two claws    AK40-10 V3.0  one motor -> crank -> two prismatic claws, 0..41 mm each (were joint7-8)
joint5, joint6 and joint7 meet at one point (the wrist centre, 130 mm below the AK80 output): a spherical wrist.
"""
from .arm_params import CameraMount, WRIST_FOCAL_MM, _deg  # noqa: F401 -- same camera model as v2

# --- Default (home) pose: as v2 (elbows flexed +/-90 deg), the new wrist joints at zero
DEFAULT_JOINT_POS = {
    "joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": _deg(90.0), "joint5": 0.0,
    "joint6": 0.0, "joint7": 0.0, "joint8": 0.0, "joint9": 0.0,
    "joint1L": 0.0, "joint2l": 0.0, "joint3l": 0.0, "joint4l": _deg(-90.0), "joint5l": 0.0,
    "joint6l": 0.0, "joint7l": 0.0, "joint8l": 0.0, "joint9l": 0.0,
}

LEFT_ARM_JOINTS = ["joint1L", "joint2l", "joint3l", "joint4l", "joint5l", "joint6l", "joint7l"]
RIGHT_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
LEFT_GRIPPER_JOINTS = ["joint8l", "joint9l"]
RIGHT_GRIPPER_JOINTS = ["joint8", "joint9"]

# Claw targets: open, and closed with the claws touching (the URDF limits; same stroke and signs as v2's joint7 / 8)
LEFT_GRIPPER_OPEN = {"joint8l": 0.0, "joint9l": 0.0}
LEFT_GRIPPER_CLOSED = {"joint8l": -0.041, "joint9l": 0.041}
RIGHT_GRIPPER_OPEN = {"joint8": 0.0, "joint9": 0.0}
RIGHT_GRIPPER_CLOSED = {"joint8": -0.041, "joint9": 0.041}

# Gripper: the AK40-10's 4.1 Nm peak through the crank (v2's numbers already assumed an AK40-10)
_GRIPPER = dict(stiffness=400.0, damping=40.0, effort_limit=34.0, velocity_limit=0.2)  # N, m/s

# AK45-10 V3.0 (CubeMars product page): 2.5 Nm rated / 7 Nm peak, 180 rpm no-load = 18.8 rad/s, 10:1.
# Stiffness / damping scaled from v2's wrist group by the torque ratio (7 / 4.1); not tuned on hardware.
_WRIST = dict(stiffness=580.0, damping=30.0, effort_limit=7.0, velocity_limit=6.0)

ACTUATOR_GROUPS = {
    "left_shoulder": dict(joints=["joint1L", "joint2l"], stiffness=2270.0, damping=180.0, effort_limit=53.0, velocity_limit=6.0),
    "left_elbow": dict(joints=["joint3l", "joint4l", "joint5l"], stiffness=1550.0, damping=110.0, effort_limit=22.0, velocity_limit=6.0),
    "left_wrist": dict(joints=["joint6l", "joint7l"], **_WRIST),
    "left_gripper": dict(joints=["joint8l", "joint9l"], **_GRIPPER),
    "right_shoulder": dict(joints=["joint1", "joint2"], stiffness=2270.0, damping=180.0, effort_limit=53.0, velocity_limit=6.0),
    "right_elbow": dict(joints=["joint3", "joint4", "joint5"], stiffness=1550.0, damping=110.0, effort_limit=22.0, velocity_limit=6.0),
    "right_wrist": dict(joints=["joint6", "joint7"], **_WRIST),
    "right_gripper": dict(joints=["joint8", "joint9"], **_GRIPPER),
}

# One motor drives both claws: they always move evenly (joint9 = -joint8)
FINGER_PAIRS = (("joint8", "joint9"), ("joint8l", "joint9l"))
FINGER_BODIES = ("link8", "link9", "link8l", "link9l")
EE_BODIES = {"left": "link7l", "right": "link7"}           # the hand link (wrist frame + gripper), origin at the wrist centre

# Wrist cameras: on the wrist frame's bracket, the same pose relative to the claws as v2's (lens front, link7 frame;
# v2 had them on link6 at (0.06179, -0.05097 / 0.05997, -0.07977)); same lens and orientation.
_CAM_ROT = (0.696364, -0.122788, 0.122788, -0.696364)
CAMERAS_WRIST = {
    "wrist_left": CameraMount("link7l", "cam_wrist_left", (0.06435, 0.04390, -0.09992), _CAM_ROT,
                              WRIST_FOCAL_MM, 20.955, (0.05, 5.0), 4 / 3),
    "wrist_right": CameraMount("link7", "cam_wrist_right", (0.06435, -0.04390, -0.09992), _CAM_ROT,
                               WRIST_FOCAL_MM, 20.955, (0.05, 5.0), 4 / 3),
}


def wrist_centre_offset() -> float:
    """Forearm-roll output face -> wrist centre (m), along the forearm."""
    return 0.130
