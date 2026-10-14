"""Quest teleop cameras: shared Pioneer ego camera and VR wrist cameras.

Prim paths assume the robot is spawned at ``{ENV_REGEX_NS}/Robot``.
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from pioneer_humanoid.bimanual_arm import make_camera_cfg

# VR wrist-camera lens. The ego camera uses the shared Pioneer lens.
DATA_CAM_LENS = sim_utils.PinholeCameraCfg(
    focal_length=7.336, horizontal_aperture=20.955, vertical_aperture=15.2908,
    clipping_range=(0.01, 100.0),
)

# wrist_cam aiming. The two angles are independent: roll spins the image, pitch aims the camera.
WRIST_CAM_ROLL_DEG = 270.0   # rotates the image counter-clockwise; 90 / 180 / 270
WRIST_CAM_PITCH_DEG = 30.0   # angles the camera down toward the gripper

# wrist_cam mount, relative to link6l's origin. Gripper-relative: +X up, +Y left, -Z forward
# (follows from the roll/pitch above). It sits inside the wrist housing, so large moves bury it
# in the mesh; the Quest script's _SHOW_WRIST_CAM_MARKER flag draws a sphere here to check.
WRIST_CAM_POS = (0.06168, 0.053, -0.06995)


def _quat_mul_wxyz(a: tuple, b: tuple) -> tuple:
    """Hamilton product of two (w, x, y, z) quaternions as plain tuples.

    isaaclab.utils.math.quat_mul wants torch tensors; these are module constants."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


# Pitch must be composed OUTSIDE the roll. The other order rolls the pitch axis too, so changing
# the roll would silently change which way the camera tilts.
def _wrist_cam_rot(roll_deg: float) -> tuple:
    """Wrist-camera orientation for a given roll, at the shared WRIST_CAM_PITCH_DEG."""
    return _quat_mul_wxyz(
        (math.cos(math.radians(WRIST_CAM_PITCH_DEG) / 2.0), 0.0,
         math.sin(math.radians(WRIST_CAM_PITCH_DEG) / 2.0), 0.0),
        (math.cos(math.radians(roll_deg) / 2.0), 0.0, 0.0,
         math.sin(math.radians(roll_deg) / 2.0)),
    )


WRIST_CAM_ROT = _wrist_cam_rot(WRIST_CAM_ROLL_DEG)


def make_ego_cam_cfg() -> CameraCfg:
    """Fresh CameraCfg for ego_cam. Requires launching with --enable_cameras.

    A new instance per call, not a shared constant: InteractiveScene rewrites prim_path in place
    when it resolves {ENV_REGEX_NS}."""
    return make_camera_cfg("ego", height=480, width=640)


# Right wrist cam is the left one mirrored through the robot's XZ plane: position (x,y,z)->
# (x,-y,z), roll -> 180 - roll. The +180 is not cosmetic: a pure frame mirror aims correctly
# but flips image-up (local +Y -> -Y), rendering the gripper upside down; the extra 180 roll
# about the view axis restores it. Pitch is unchanged (already about the mirror-plane normal).
#
# Asset asymmetry, compensated: link6's origin sits 9.0mm further out in Y than the exact mirror
# of link6l's (CAD frame placement; the fingertips themselves mirror perfectly). Without adding
# it back, the right feed sat too far outboard in the headset.
_LINK6_ORIGIN_Y_ASYMMETRY_M = 0.008997  # link6l.y + link6.y, measured in-sim at the rest pose


def make_wrist_cam_cfg(body: str = "link6l", name: str = "wrist_cam", mirror: bool = False) -> CameraCfg:
    """Fresh CameraCfg for a wrist camera -- see make_ego_cam_cfg for why it is a factory.

    Defaults reproduce the original single link6l camera exactly, so existing call sites are
    unchanged. Pass ``body="link6", mirror=True`` for the opposite arm's wrist."""
    mirrored_pos = (WRIST_CAM_POS[0],
                    -WRIST_CAM_POS[1] + _LINK6_ORIGIN_Y_ASYMMETRY_M,
                    WRIST_CAM_POS[2])
    pos = mirrored_pos if mirror else WRIST_CAM_POS
    rot = _wrist_cam_rot(180.0 - WRIST_CAM_ROLL_DEG) if mirror else WRIST_CAM_ROT
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + f"{body}/{name}",
        spawn=DATA_CAM_LENS,
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
        height=480, width=640,
        update_period=0.0,
        data_types=["rgb"],
    )
