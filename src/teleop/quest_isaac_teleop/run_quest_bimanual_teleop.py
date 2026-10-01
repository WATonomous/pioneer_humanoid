"""Quest VR -> Isaac Sim teleoperation for the pioneer_bimanual_arm (pioneer_bimanual_arm.usd).

Runs inside the simulation_isaac container. Both arms are driven by Quest hand tracking:
the left Quest wrist drives the left arm, the right drives the right, and pinching
thumb+index closes that arm's gripper.

Pipeline
--------
    Quest browser (WebXR) --WSS--> quest_teleop_node --/quest_teleop--> this script
    this script --DLS IK--> Isaac Sim --JPEG--> webxr_server --HTTP poll--> headset

Control
-------
IK: plain (unweighted) damped-least-squares, one solver per arm so neither arm's homing
state leaks into the other's. Each arm tracks its gripper's FINGERTIP-TIP midpoint (not
the wrist link); orientation comes from the wrist link. Damping lambda_val is 0.1
(0.175 right) -- see _DLS_LAMBDA.

Coordinate mapping: WebXR is Y-up (X=right, Y=up, -Z=forward); the robot base is Z-up and
yawed 180deg. Both hand POSITION and wrist ORIENTATION are mapped CAMERA-RELATIVE (see
_QUEST_TO_CAM_LOCAL / _CAM_LOCAL_*), not through a fixed world rotation, so "push away
from you" follows the camera's actual boresight.

Naming: the "L"-suffixed chain (joint1L..joint6l, link6l) is the physical LEFT arm and is
driven by the LEFT Quest wrist -- both senses agree. Unsuffixed = RIGHT. pioneer_humanoid.bimanual_arm's
LEFT_*/RIGHT_* are imported as-is.

Cameras
-------
    eye pair (RSD455)  stereo headset view, attached at runtime -> pov_left/right.jpg
    wrist_cam          link6l (LEFT Quest hand) -- headset HUD panel + recording
    wrist_cam_right    link6  (RIGHT Quest hand) -- headset HUD panel only
    ego_cam            recording only; left out of the scene entirely without --record

Performance
-----------
See main() for the measured cost model and the fps/RTF tuning table. Three knobs:
_POV_CAPTURE_EVERY_N_STEPS (render cadence), _PHYSICS_DT, _PACE_TO_REALTIME.

Keys (Isaac Sim window focused): R recalibrate, T reset scene, S save episode,
D discard episode (S/D only with --record).

Usage
-----
    ./run_quest_bimanual_teleop.sh [--record] [--publish-real-left-arm]
"""

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

# ── path setup (must be before AppLauncher so PYTHONPATH is correct) ─────────
_THIS_DIR = Path(__file__).resolve().parent
_AUTONOMY = _THIS_DIR.parents[1]  # src/teleop/quest_isaac_teleop -> src
_SIM_DIR = _AUTONOMY / "simulation"
# pioneer_humanoid package (canonical arm config). Editable-installed in the image; this fallback
# keeps a bare bind-mounted checkout working.
sys.path.insert(0, str(_AUTONOMY / "pioneer_humanoid"))
# src/teleop/ for the shared teleop_cameras module
sys.path.insert(0, str(_AUTONOMY / "teleop"))
_IL_PKG = _AUTONOMY / "il"


def _ensure_il_on_path() -> None:
    if str(_IL_PKG) not in sys.path:
        sys.path.insert(0, str(_IL_PKG))


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Quest pioneer_bimanual_arm teleop (both arms: DLS fingertip IK)")
parser.add_argument("--gain", type=float, default=1.0,
                    help="Motion gain: metres of EE motion per metre of real wrist motion")
parser.add_argument("--record", action="store_true",
                    help="Record demonstrations (requires: pip install -e src/il[record]). Records the "
                         "L-suffixed (link6l) arm only -- that's the arm ego_cam/wrist_cam are mounted for.")
parser.add_argument("--schema", type=str,
                    default=str(_IL_PKG / "config" / "dataset_schema_wato_arm_v2_push_box.yaml"),
                    help="Path to dataset_schema.yaml (only used with --record)")
parser.add_argument("--sink", type=str, default="lerobot",
                    help="Output sinks when --record: lerobot, hdf5, or lerobot,hdf5")
parser.add_argument("--dataset_root", type=str, default=None,
                    help="Override record.root from schema")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--task_description", type=str, default="pick up box and place it in container")
parser.add_argument("--publish-real-left-arm", action="store_true",
                    help="Also publish the left arm's DLS-solved joint targets to /arm/joint_targets for "
                         "joint_command_node to drive the REAL physical left arm over CAN. Off by default -- "
                         "sim-only unless explicitly requested. RIGHT ARM HAS NO REAL-HARDWARE CAN MAPPING YET "
                         "(hardware_mapping.yaml only has a left: section) -- this flag does not and cannot "
                         "touch the right arm. Requires can_node + joint_command_node already running, the "
                         "CANable adapter connected, e-stop armed, and the real arm manually positioned near "
                         "the sim's rest pose (see PUBLISH_START_DELAY below).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports (require Omniverse runtime) ───────────────────────────
import rclpy  # noqa: E402
import torch  # noqa: E402
from rclpy.node import Node  # noqa: E402
from common_msgs.msg import QuestHandPose  # noqa: E402

import carb  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    quat_apply,
    quat_apply_inverse,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    quat_slerp,
    subtract_frame_transforms,
)

# No aliasing: LEFT_* is the L-suffixed chain (physical left = left Quest wrist), RIGHT_* the
# unsuffixed one. This block used to swap them to undo a reversed upstream; don't bring that back.
from teleop_cameras import (  # noqa: E402
    WRIST_CAM_POS,
    make_ego_cam_cfg,
    make_wrist_cam_cfg,
)
from pioneer_humanoid.bimanual_arm import (  # noqa: E402
    BIMANUAL_ARM_CFG,
    LEFT_ARM_JOINTS,
    LEFT_EE_BODY,
    LEFT_FINGER_TIP_BODIES,
    LEFT_FINGER_DISTAL_TIP_LOCAL,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_OPEN,
    LEFT_GRIPPER_CLOSED,
    RIGHT_ARM_JOINTS,
    RIGHT_EE_BODY,
    RIGHT_FINGER_TIP_BODIES,
    RIGHT_FINGER_DISTAL_TIP_LOCAL,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_OPEN,
    RIGHT_GRIPPER_CLOSED,
    apply_joint_limits,
    compute_gripper_tip_pose_b,
    compute_tip_ik_jacobian,
    resolve_body_ids,
    resolve_joint_name,
)

# ── constants ─────────────────────────────────────────────────────────────────

# RETIRED. Fixed WebXR->world rotation, superseded by _QUEST_TO_CAM_LOCAL for both position and
# orientation. Kept only because quest_to_world_quat is still constructed from it below.
_QUEST_TO_WORLD = torch.tensor(
    [[-1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)
# Maps a WebXR-local rotation (X=right, Y=up, -Z=forward) into the camera mount's local frame:
# cam X(fwd)=-quest_z, Y(left)=-quest_x, Z(up)=quest_y. Then rotated to world by camera_tilt_quat.
# Position and orientation both use this camera-relative construction; a fixed world rotation
# made sideways hand motion drive the arm forward. Flip an entry only for a handedness issue.
_QUEST_TO_CAM_LOCAL = torch.tensor(
    [[0.0, 0.0, -1.0],
     [-1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)
_AXIS_SIGN_LEFT = torch.tensor([1.0, 1.0, 1.0])  # [right, up, forward]
_AXIS_SIGN_RIGHT = torch.tensor([1.0, 1.0, 1.0])  # [right, up, forward]
_CAM_LOCAL_FORWARD = torch.tensor([1.0, 0.0, 0.0])  # rsd455 local +X = boresight
_CAM_LOCAL_UP = torch.tensor([0.0, 0.0, 1.0])  # rsd455 local +Z = up
_CAM_LOCAL_RIGHT = torch.tensor([0.0, -1.0, 0.0])  # matches _EYE_LOCAL_RIGHT below

_GAIN_FAR = 1.0  # 1:1 hand->EE. 2.0 was tried for more reach; it made tracking feel imprecise.
_GAIN_RAMP_START_M = 0.15
_GAIN_RAMP_END_M = 0.35

# Shoulder-to-fingertip is ~0.70-0.75m (link-length estimate), so 0.5m stays under the
# kinematic ceiling but close enough that DLS conditioning degrades near it -- see
# _DLS_LAMBDA_RIGHT. Lower it if the right arm locks up mid-reach.
_MAX_REACH_M = 0.5

# Both 0: fix framing on the camera/enclosure side, not by offsetting the arm's target from its
# rest pose. Z was +0.2 to lift the tip clear of the table under the old all-zeros spawn pose;
# the arm config's flexed-elbow spawn already rests 0.27m above the table, so it's
# redundant -- and it also pulled the hands closer to the eye cameras, ~doubling the arm mesh
# in frame and adding ~30ms/cycle of render+encode.
_HOME_TIP_X_OFFSET = 0.0
_HOME_TIP_Z_OFFSET = 0.0

# Real-hardware bridge timing (--publish-real-left-arm), mirroring task_space_ik.py.
# joint_command_node now seeds its rate-limiter from live motor feedback on the first
# ArmPose, so even that first command is velocity/delta-limited from the real arm's actual
# pose (no snap). The delay is operator prep time -- a hand on the e-stop before motion.
_REAL_ARM_PUBLISH_PERIOD_S = 0.02  # 20ms = 50Hz, matches joint_command_node's control_rate_hz
_REAL_ARM_PUBLISH_START_DELAY_S = 5.0

# Per-arm rotation offset conjugated onto the wrist orientation delta; identity = no
# correction. The old bimanual_arm_cfg's non-identity value does NOT transfer (this asset's
# joint6l rotates about Y, the old one about X). Retune per the README's "Wrist orientation" section
# if rotation drives the EE wrong -- but a "rotates the opposite way" feel is a chirality
# issue needing a sign flip on the raw quaternion, not a rotation offset here.
_WRIST_ORIENT_OFFSET_LEFT = torch.tensor([1.0, 0.0, 0.0, 0.0])
_WRIST_ORIENT_OFFSET_RIGHT = torch.tensor([1.0, 0.0, 0.0, 0.0])

_THUMB_TIP_IDX = 4
_INDEX_TIP_IDX = 9
_PINCH_CLOSE_M = 0.035  # was 0.030, nudged up per live feedback (easier to trigger close)
_PINCH_OPEN_M = 0.050

# Lowered from 0.2/0.35: those were raised for conditioning near full extension, which was an
# artifact of the old all-zeros spawn pose. the arm config's flexed spawn fixes the
# conditioning, and lower damping cuts wrist-twist leakage into the shoulder (~35% -> ~18% at
# 0.2 -> 0.1). Raise back toward 0.2/0.35 if the arm feels jittery near singularities.
# Per-arm ratio kept, not unified -- only the left chain was re-measured.
_DLS_LAMBDA = 0.1
_DLS_LAMBDA_RIGHT = 0.175

# Ceiling + threshold for Chiaverini adaptive damping (_adaptive_dls_lambda): lambda ramps from
# the floor above toward lambda_max as manipulability drops below epsilon.
# CURRENTLY INACTIVE -- solve_and_apply was reverted to fixed lambda on live feedback that
# adaptive damping made the IK feel wrong. Kept wired up for a retry; the [Quest][ikdiag] print
# exists to gather the manipulability data needed to tune epsilon properly first.
_DLS_LAMBDA_MAX = 0.6
_DLS_LAMBDA_MAX_RIGHT = 0.9
_DLS_MANIPULABILITY_EPSILON = 0.01

# Joint-space output smoothing, modelled on joint_command_core.cpp's active default (velocity +
# delta clamp; NOT its trapezoidal ramp or low-pass, both inactive on hardware). safety_limits
# .yaml's per-joint overrides are deliberately NOT applied -- the arm config's joint
# grouping (2-DOF shoulder/3-DOF elbow) does not match hardware_mapping.yaml's ArmPose split.
# All three are UNUSED as of the revert to unsmoothed IK output; kept for a retry.
_JOINT_VELOCITY_MAX_RAD_S = 1.7453292519943295  # 100 deg/s speed ceiling
_JOINT_DELTA_MAX_RAD = 0.2617993877991494  # 15 deg/step
_SMOOTH_TIME_S = 0.08  # _smooth_damp: time to close ~90% of the gap. Lower = snappier.

_ENCLOSURE_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), emissive_color=(1.0, 1.0, 1.0))
# Enclosure footprint. X spans the mount/stand bar back to 0.75m, which contains the arms' full
# reach instead of letting them poke through; the Y walls and ceiling close the sides and top.
_ENCLOSURE_X_MIN = -0.45
_ENCLOSURE_X_MAX = 0.75
_ENCLOSURE_Y_MIN = -0.9
_ENCLOSURE_Y_MAX = 1.0
# The stand puts the arm's highest point ~1.457m above the table; 1.8m leaves ~0.34m headroom.
_ENCLOSURE_TOP_Z = 1.8
_ENCLOSURE_Y_CTR = (_ENCLOSURE_Y_MIN + _ENCLOSURE_Y_MAX) / 2
_ENCLOSURE_Y_SPAN = _ENCLOSURE_Y_MAX - _ENCLOSURE_Y_MIN
_ENCLOSURE_X_SPAN = _ENCLOSURE_X_MAX - _ENCLOSURE_X_MIN
_ENCLOSURE_X_CTR = (_ENCLOSURE_X_MIN + _ENCLOSURE_X_MAX) / 2
# Which X face carries the closed backdrop. _ENCLOSURE_X_MAX puts it in the +X direction the
# robot faces (6cm past the table's front edge), so the head cameras and recorded POV see white;
# the open face is at _ENCLOSURE_X_MIN, behind the robot where the operator stands.
_ENCLOSURE_BACK_X = _ENCLOSURE_X_MAX


# table.usd was converted from a SolidWorks STEP export (units: inches, hence the scale). The
# STEP source is not tracked here; re-export from CAD if the geometry changes. Its origin sits
# at the table-TOP surface, so no Z offset is needed. Real size ~1.39 x 0.75 x 0.62m.
_TABLE_USD_PATH = str(
    _SIM_DIR.parent.parent / "assets" / "props" / "table.usd"
)
_TABLE_SCALE = (0.0254, 0.0254, 0.0254)
# GUI-verified pose. The rotation is needed because the STEP conversion's native axes do not
# come out Z-up-front (Orient XYZ = 90, 90, 0 deg in USD rotateXYZ convention).
_TABLE_POS = (0.69, 0.00612, 0.33)
_TABLE_ROT = (0.5000000000000001, 0.5, 0.5, 0.49999999999999994)  # wxyz

# Box (to grasp) and container (to place it in). Both assets have their local origin at a
# bottom corner, so matching X/Y/Z rests the box flush on the container floor.
_BOX_USD_PATH = str(
    _SIM_DIR.parent.parent / "assets" / "props" / "block.usd"
)  # 5.08cm cube -- box.usd (25x25x3cm flat pad) is too flat/wide for this gripper to grasp
_CONTAINER_USD_PATH = str(
    _SIM_DIR.parent.parent / "assets" / "lerobot" / "so101_vial_task" / "usd" / "tray.usda"
)
_CONTAINER_POS = (0.3, -0.07041, 0.70917)  # +33cm net with the table
_CONTAINER_ROT = (0.7071067811865476, 0.0, 0.0, 0.7071067811865475)  # wxyz
_BOX_POS = (0.2, 0.2, 0.70917)  # moved 0.1m closer in X, per live feedback (was 0.3)

# Stereo pair: two RealSense D455s on base_link giving real depth via two eye textures (not a
# mirrored monocular feed), fixed at _HEAD_VIEWPOINT_HOME_POS/QUAT. Head tracking is off --
# head_pose is still received but unused for positioning; see _HEAD_TRACKING_LIVE.
# rsd455 local axes: +X = boresight, +Z = up, so -Y = right (the IPD offset axis).
_RSD455_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Sensors/Intel/RealSense/rsd455.usd"
)
# Head cam pose, relative to base_link: a level downward glance facing +X (the confirmed
# "front"). Built from forward/up vectors rather than composed rotations, which introduced roll
# in an earlier attempt. Aiming it at the rest-pose fingertips instead was tried and is wrong --
# it forces a ~92deg near-vertical pitch, since the zero-pose hands droop below shoulder height.
# A far-side (180deg yaw) viewpoint was also tried and rejected.
_HEAD_VIEWPOINT_HOME_POS = (0.0905602619765378, 0.04, 0.23809789390566405)
_HEAD_VIEWPOINT_HOME_QUAT = (0.9063077870366499, 0.0, 0.42261826174069944, 0.0)  # ~50deg pitch, no roll
_EYE_LOCAL_RIGHT = torch.tensor([0.0, -1.0, 0.0])
_EYE_IPD_M = 0.063
_EYE_LOCAL_FORWARD = torch.tensor([1.0, 0.0, 0.0])
# ego_cam is pinned to the left eye's pose but nudged clear of the RSD455 housing mesh, which
# otherwise blacks out half the frame: forward along the gaze, plus a straight base-frame +Z
# (the forward offset alone was not enough -- it needed to clear geometry above it, not ahead).
_EGO_CAM_FORWARD_OFFSET_M = 0.25
_EGO_CAM_UP_OFFSET_M = 0.1
# When True, ego_cam is never repositioned, so a manual GUI drag sticks. Probing aid only.
_EGO_CAM_FREE_MOVE_DEBUG = False

# Head tracking is not live -- head_viewpoint_pos_b/quat_b are set once at startup and never
# mutated (see run_simulator). With this False the camera mounts are positioned once before the
# sim loop instead of re-authored every physics step; flip to True the moment the head pose is
# actually driven from the Quest, and per-step tracking resumes with no other change.
_HEAD_TRACKING_LIVE = False
# ego_cam is a bare USD Camera prim (looks down local -Z, up +Y), NOT an rsd455 payload
# (+X forward, +Z up, -Y right). _HEAD_VIEWPOINT_HOME_QUAT is calibrated for the rsd455
# convention, so it aims ego_cam wrong despite being the identical quaternion value. This is the
# basis change between them -- compose as quat_mul(head_orient_b, this), not the reverse.
_EGO_CAM_CONVENTION_FIX_QUAT = (-0.5, -0.5, 0.5, 0.5)
# Extra +20deg downward pitch on ego_cam only (70deg total); the eye views keep their own
# orientation. Applied as the OUTERMOST rotation: quat_mul(this, head_orient * convention_fix).
_EGO_CAM_EXTRA_TILT_QUAT = (0.984807753012208, 0.0, 0.17364817766693033, 0.0)
# Sub-path to the actual renderable Camera prim inside the rsd455 payload --
# same as the SO101 vial task's camera_external_D455 (task_env_cfg.py).
_RSD455_CAMERA_SUBPATH = "rsd455/RSD455/Camera_OmniVision_OV9782_Right"
# Widen the RSD455's baked-in ~90.5deg FOV to ~120deg. The head is too close to the arm to see
# its full range by pulling the camera back (stand geometry blocks the view first), so widen the
# lens instead. Solved from fov = 2*atan(aperture / (2*focalLength)), aperture unchanged.
_RSD455_WIDENED_FOCAL_LENGTH = 1.1246782979523935  # ~120deg horizontal FOV

# Rubber-on-plastic-ish grip friction, applied to both the box and the gripper fingers so the
# effective coefficient is the same whichever way PhysX combines the two materials.
_BOX_STATIC_FRICTION = 1.5
_BOX_DYNAMIC_FRICTION = 1.2
_GRIPPER_STATIC_FRICTION = 1.5
_GRIPPER_DYNAMIC_FRICTION = 1.2


def _set_rigid_body_friction(
    asset, static_friction: float, dynamic_friction: float, body_names: tuple[str, ...] | None = None,
) -> None:
    """Set static/dynamic friction on PhysX's material buffer via root_physx_view.

    MUST go through get/set_material_properties(). root_physx_view fills its material buffer at
    cook time (sim.reset()) and never re-reads USD afterwards, so binding a UsdShade.Material at
    runtime silently does nothing -- that cost weeks of "friction has no effect" debugging. Same
    mechanism isaaclab.envs.mdp.events.randomize_rigid_body_material uses.

    body_names=None writes every shape (fine for the single-body box); naming links restricts the
    write so the rest of the articulation keeps its own material."""
    view = asset.root_physx_view
    materials = view.get_material_properties()  # (num_instances, max_shapes, 3)
    env_ids = torch.arange(materials.shape[0])
    touched_shapes = 0

    if body_names is None:
        materials[:, :, 0] = static_friction
        materials[:, :, 1] = dynamic_friction
        touched_shapes = materials.shape[1]
    else:
        # The shapes dimension is flattened across ALL bodies in link_paths order, and
        # Articulation exposes no per-body shape-range lookup, so each body's slice has to be
        # found by summing shape counts. Same workaround randomize_rigid_body_material uses.
        shape_start = 0
        for link_path in view.link_paths[0]:
            link_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # noqa: SLF001
            num_shapes = link_view.max_shapes
            if link_path.split("/")[-1] in body_names:
                materials[:, shape_start:shape_start + num_shapes, 0] = static_friction
                materials[:, shape_start:shape_start + num_shapes, 1] = dynamic_friction
                touched_shapes += num_shapes
            shape_start += num_shapes

    view.set_material_properties(materials, env_ids)

    # Read back rather than trust the write -- the old USD path failed silently for weeks.
    # Zero matched shapes means body_names does not match link_paths naming.
    readback = view.get_material_properties()
    print(f"[Quest][friction] {asset.cfg.prim_path}: wrote {touched_shapes}/{materials.shape[1]} shapes "
          f"-> static={static_friction}, dynamic={dynamic_friction} | "
          f"PhysX now reports static={readback[0, :, 0].tolist()} dynamic={readback[0, :, 1].tolist()}", flush=True)
    if touched_shapes == 0:
        print(f"[Quest][friction] WARNING: matched NO shapes for {body_names}. "
              f"Actual link_paths: {[p.split('/')[-1] for p in view.link_paths[0]]}", flush=True)


def _widen_camera_fov(camera_prim_path: str, focal_length: float) -> None:
    """Override a Camera prim's focalLength, waiting for it to load first.

    The rsd455 payload takes a few frames to compose after AddPayload(); writing before it loads
    silently no-ops rather than failing, hence the retry loop."""
    stage = omni.usd.get_context().get_stage()
    app = omni.kit.app.get_app_interface()
    for _ in range(100):
        cam_prim = stage.GetPrimAtPath(camera_prim_path)
        if cam_prim.IsValid() and cam_prim.IsA(UsdGeom.Camera):
            UsdGeom.Camera(cam_prim).GetFocalLengthAttr().Set(focal_length)
            print(f"[Quest] Widened FOV on {camera_prim_path} (focalLength={focal_length})", flush=True)
            return
        app.update()
    print(f"[Quest] WARNING: could not widen FOV on {camera_prim_path} (camera prim never became valid)", flush=True)


def _read_camera_fov(camera_prim_path: str) -> tuple[float, float, float] | None:
    """Read a Camera prim's focalLength/apertures; same load-wait as _widen_camera_fov.

    Call BEFORE _widen_camera_fov, to capture the RSD455's native FOV before it is overwritten."""
    stage = omni.usd.get_context().get_stage()
    app = omni.kit.app.get_app_interface()
    for _ in range(100):
        cam_prim = stage.GetPrimAtPath(camera_prim_path)
        if cam_prim.IsValid() and cam_prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(cam_prim)
            return (
                cam.GetFocalLengthAttr().Get(),
                cam.GetHorizontalApertureAttr().Get(),
                cam.GetVerticalApertureAttr().Get(),
            )
        app.update()
    print(f"[Quest] WARNING: could not read FOV on {camera_prim_path} (camera prim never became valid)", flush=True)
    return None


def _set_camera_fov(camera_prim_path: str, focal_length: float, h_aperture: float, v_aperture: float) -> None:
    """Set a Camera prim's focalLength/apertures. No load-wait -- ego_cam is baked into the USD,
    unlike the late-composing rsd455 payload."""
    stage = omni.usd.get_context().get_stage()
    cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim_path))
    cam.GetFocalLengthAttr().Set(focal_length)
    cam.GetHorizontalApertureAttr().Set(h_aperture)
    cam.GetVerticalApertureAttr().Set(v_aperture)


def _project_world_point_to_uv(camera_prim: Usd.Prim, world_xyz: tuple) -> tuple[float, float, bool]:
    """World point -> that camera's normalized image UV (0..1, top-left origin).

    Uses the camera's REAL resolved transform: GetCamera() composes the full prim hierarchy,
    including the rsd455 payload's internal camera-to-mount offset, so a projected point lands on
    the pixel it actually occupies in pov_*.jpg. Drawing at the browser's own WebXR hand position
    instead is a different coordinate space, and put the marker at the operator's physical wrist.

    Returns (u, v, visible); skip drawing when visible is False (behind camera / outside frustum)."""
    cam = UsdGeom.Camera(camera_prim).GetCamera(Usd.TimeCode.Default())
    view = cam.frustum.ComputeViewMatrix()
    proj = cam.frustum.ComputeProjectionMatrix()
    clip = Gf.Vec4d(world_xyz[0], world_xyz[1], world_xyz[2], 1.0) * view * proj
    if clip[3] <= 1e-6:
        return 0.5, 0.5, False
    ndc_x = clip[0] / clip[3]
    ndc_y = clip[1] / clip[3]
    u = (ndc_x + 1.0) / 2.0
    v = (1.0 - ndc_y) / 2.0
    visible = -1.0 <= ndc_x <= 1.0 and -1.0 <= ndc_y <= 1.0
    return u, v, visible


def _attach_rsd455_camera(parent_prim_path: str, mount_name: str, translate: tuple, orient_wxyz: tuple) -> str:
    """Attach an rsd455 payload as a `mount_name` child Xform. Returns the mount prim path.

    create=True is used only here; see _set_mount_pose for why it must not be repeated."""
    mount_path = f"{parent_prim_path}/{mount_name}"
    _set_mount_pose(mount_path, translate, orient_wxyz, create=True)
    stage = omni.usd.get_context().get_stage()
    rsd_prim = stage.DefinePrim(f"{mount_path}/rsd455")
    rsd_prim.GetPayloads().AddPayload(_RSD455_USD_URL)
    return mount_path


def _set_mount_pose(mount_path: str, translate: tuple, orient_wxyz: tuple, create: bool = False) -> None:
    """Set (or create) a camera mount Xform's translate/orient.

    AddTranslateOp()/AddOrientOp() are NOT idempotent in this USD version -- re-adding raises
    'xformOp already exists in xformOpOrder'. So ops are added once at create=True; later calls
    fetch the existing ops and .Set() them."""
    stage = omni.usd.get_context().get_stage()
    if create:
        prim = UsdGeom.Xform.Define(stage, mount_path)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    else:
        prim = stage.GetPrimAtPath(mount_path)
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        translate_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        orient_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
        if translate_op is None or orient_op is None:
            # ego_cam is spawner-created, and a spawned prim is not guaranteed to carry both
            # named ops. Rebuild once; later calls take the fast path above.
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
            orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

    translate_op.Set(Gf.Vec3d(*translate))
    w, x, y, z = orient_wxyz
    orient_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def _spawn_wrist_cam_marker(prim_path: str, pos: tuple, radius: float = 0.01) -> None:
    """Draw a plain USD sphere at pos, parented to prim_path's parent.

    Must be created here rather than as a scene AssetBaseCfg: prims nested inside the robot spawn
    too early that way, before its referenced USD layer has composed."""
    stage = omni.usd.get_context().get_stage()
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.GetRadiusAttr().Set(radius)
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])
    xformable = UsdGeom.Xformable(sphere.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))


def _open_pov_camera(camera_prim_path: str, label: str, width: int = 480, height: int = 360):
    """Wrap an existing camera prim as a standalone Camera sensor, read via .data.output["rgb"].

    REPLACES a Kit ViewportWindow + capture_viewport_to_file approach that cost 41-68ms per pair
    and silently ignored the requested width/height (viewport window size, render-texture size
    and file-capture resolution are three different properties). CameraCfg width/height are
    always honoured. Capture dropped to ~5ms.

    Caller must .update(dt) before each read -- this is not a scene entity, so scene.update()
    never ticks it (the eye mounts are created at runtime, after scene construction).

    The _initialize_callback call is not optional: SensorBase allocates its internal state only
    in _initialize_impl(), invoked solely by a timeline PLAY-event callback. Sensors built before
    sim.reset() get that for free; this one cannot, since its prim does not exist until
    _attach_rsd455_camera runs. Firing the callback directly does what the missed event would."""
    try:
        camera = Camera(CameraCfg(
            prim_path=camera_prim_path, spawn=None, height=height, width=width,
            update_period=0.0, data_types=["rgb"],
        ))
        camera._initialize_callback(None)  # noqa: SLF001 -- see docstring
        print(f"[Quest] {label} camera ready at {width}x{height} ({camera_prim_path})", flush=True)
        return camera
    except Exception as exc:  # noqa: BLE001 -- best-effort convenience feature
        print(f"[Quest] Could not create POV camera '{label}': {exc}", flush=True)
        return None


def _save_frame_atomic(frame, file_path, quality: int = 80) -> None:
    """Encode an HxWx3 uint8 array to file_path via a temp file + atomic rename.

    Three details, each of which caused a real failure:
      - rename: writing straight to the served path lets a poll fetch a half-written file.
        os.replace is atomic, so a reader sees the old frame or the new one, never a torn one.
      - explicit format=: PIL infers the encoder from the extension and cannot map ".tmp".
      - PID in the temp name: two live sim instances otherwise race on one path, and the loser
        dies on a missing temp file.
    Nothing propagates -- a filesystem error must degrade the preview, never kill teleop."""
    from PIL import Image

    file_path = Path(file_path)
    tmp_path = file_path.with_name(f"{file_path.name}.{os.getpid()}.tmp")
    fmt = "JPEG" if file_path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
    try:
        Image.fromarray(frame).save(str(tmp_path), format=fmt, quality=quality)
        os.replace(tmp_path, file_path)
    except Exception as exc:  # noqa: BLE001 -- see docstring: never kill teleop over a frame
        print(f"[Quest] WARNING: could not write {file_path.name}: {exc}", flush=True)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _camera_rgb_frame(camera):
    """HxWx3 uint8 RGB from a Camera sensor's output tensor, alpha dropped if present."""
    frame = camera.data.output["rgb"][0].detach().cpu().numpy()
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    return frame.astype("uint8")


def _write_pov_jpeg(camera, file_path) -> None:
    """Write a standalone Camera's current RGB frame to file_path as an atomically-replaced
    JPEG -- same tensor-read approach as _write_wrist_cam_hud_frames, for the two eye cameras."""
    _save_frame_atomic(_camera_rgb_frame(camera), file_path)


# Eye frames for the headset. They live in the WebXR static dir so webxr_server.py's stock
# SimpleHTTPRequestHandler serves them with no server changes. Written every
# _POV_CAPTURE_EVERY_N_STEPS steps -- the same constant that gates `render=`, so a capture
# always lands on freshly-rendered pixels. Sim-time render rate is 1/(n*dt); the headset sees
# that scaled by RTF, which is why the fps diagnostic prints both.
_POV_STATIC_DIR = _SIM_DIR.parent / "teleop" / "quest_teleop" / "static"
_POV_FRAME_PATH_LEFT = _POV_STATIC_DIR / "pov_left.jpg"
_POV_FRAME_PATH_RIGHT = _POV_STATIC_DIR / "pov_right.jpg"
# THE render/capture cadence -- single source of truth. main()'s SimulationCfg reads this, and
# so does the `render=` gate in run_simulator's loop, so there is nothing to keep in sync by hand.
# Raising it trades frame rate for real-time factor: fps = 1/(n*_PHYSICS_DT) once RTF hits 1.0,
# while each extra physics step is far cheaper than the render it defers. See main() for the
# measured cost model and the fps/RTF table.
_POV_CAPTURE_EVERY_N_STEPS = 5
_PHYSICS_DT = 0.02  # seconds of simulated time per physics step (50Hz)

# Dataset fps, derived: the render rate is the only rate at which recorded pixels can change.
# Declared independently (schema said 30, sim rendered 10) it duplicated ~25% of frames --
# measured 217/858 ego transitions with zero pixel change. Recorder gets rate_limit=False so
# the render gate in run_simulator is the sole decimator; never set this by hand.
_RECORD_FPS = round(1.0 / (_POV_CAPTURE_EVERY_N_STEPS * _PHYSICS_DT))
# Feeds Kit's rendering_dt (= _PHYSICS_DT * this). NOT the render gate -- that's
# _POV_CAPTURE_EVERY_N_STEPS via sim.step(render=...). Don't set it equal to the gate: rendering_dt
# is an RTX shading input (temporal accumulation, motion blur), so a larger value costs real GPU
# time (2 -> 5 took each render ~36ms -> ~97ms). It only needs to stay small.
_KIT_RENDERING_INTERVAL = 2

# Wall-clock pacer. Without it the sim advances as fast as it computes, so whenever there is
# spare headroom RTF drifts above 1.0 and the arm moves FASTER than the operator's hand. Worse,
# it oscillates around 1.0 as load varies, and a fluctuating time distortion is harder to
# compensate for than a constant one. With this on, RTF is capped at 1.0 and surplus headroom
# becomes slack instead of speed-up. It can never make the sim slower than it already is -- with
# no headroom it sleeps zero and the loop behaves exactly as before.
_PACE_TO_REALTIME = True

# Deficit the pacer will carry, in render cycles. Work here is bursty by construction -- the
# render step costs ~40ms against a 20ms budget while the other steps cost ~3ms -- so the pacer
# MUST carry that deficit across the cycle and repay it on the cheap steps. A pacer that reset
# its deadline on every overrun would sleep on the cheap steps, never recover the render
# overrun, and end up SLOWER than no pacer at all. Debt past this bound is written off instead
# of sprinted away: after a real stall, catching up would mean a burst of RTF >> 1.0, which is
# exactly what this exists to prevent.
_PACER_MAX_DEBT_CYCLES = 1.0
# [Quest][axisdbg]: two prints per physics step while a hand moves (~200 stdout writes/sec
# through the docker pipe -- costs real loop time). On only when re-deriving the axis mapping.
_SHOW_AXIS_DEBUG = False
# Wrist-cam HUD overlays, written from the existing sensor tensors (the ones recording reads)
# rather than second viewport windows, which would re-add render cost that was removed.
# left = the L-suffixed chain (link6l), right = the unsuffixed chain (link6) -- the robot's
# actual left and right, and also which Quest wrist drives each.
_WRIST_CAM_FRAME_PATH_LEFT = _POV_STATIC_DIR / "wrist_cam_left.jpg"
_WRIST_CAM_FRAME_PATH_RIGHT = _POV_STATIC_DIR / "wrist_cam_right.jpg"
# Per-eye IK-target marker screen position, projected through the REAL eye cameras so the
# browser draws it where it actually appears in the image. Using the browser's own WebXR hand
# position instead put the marker at the operator's physical wrist. Temp-file-then-renamed.
_MARKER_UV_PATH = _POV_STATIC_DIR / "marker_uv.json"

# Draws a 2cm green sphere at teleop_cameras.WRIST_CAM_POS so the camera's mount point is
# visible in the viewport while tuning it. Visual only -- no collision or physics. Turn it off
# before recording: it sits inside link6l and can appear in ego_cam/wrist_cam frames.
_SHOW_WRIST_CAM_MARKER = False


# ── helpers ───────────────────────────────────────────────────────────────────

@configclass
class ArmV2SceneCfg(InteractiveSceneCfg):
    """Bare arm stand + the white lightbox enclosure walls/table (visual only, no collision on the
    walls) -- same geometry as PushBlockSceneCfg in
    humanoid_rl_tasks/push_block/scene.py. Always on (no CLI flag)."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    # The stand's bottom sits 1.1997m below base_link, so lift by that to rest the feet on the
    # floor (z=0), not on the collision-safety ground plane at z=-1.05. The table (top ~0.7m)
    # is a separate work surface; the column passes up through it.
    robot = BIMANUAL_ARM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=BIMANUAL_ARM_CFG.init_state.replace(pos=(0.0, 0.0, 1.1997)),
        # Self-collisions are OFF in BIMANUAL_ARM_CFG itself (see the note there); no override needed.
    )

    # Data-collection / HUD cameras, not the headset display (the RSD455 pair, attached at
    # runtime). wrist_cam is on link6l (LEFT arm), wrist_cam_right its mirror on link6. Pose and
    # lens live in src/teleop/teleop_cameras.py -- adjust there, not here. wrist_cam keeps its
    # unsuffixed name because _capture_record_images, the dataset schema, and the keyboard
    # teleop scenes all bind that key.
    ego_cam = make_ego_cam_cfg()
    wrist_cam = make_wrist_cam_cfg()
    wrist_cam_right = make_wrist_cam_cfg(body="link6", name="wrist_cam_right", mirror=True)

    enclosure_back: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/EnclosureBack",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_ENCLOSURE_BACK_X, _ENCLOSURE_Y_CTR, _ENCLOSURE_TOP_Z / 2)),
        spawn=sim_utils.CuboidCfg(size=(0.003, _ENCLOSURE_Y_SPAN, _ENCLOSURE_TOP_Z), visual_material=_ENCLOSURE_MATERIAL),
    )
    enclosure_left: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/EnclosureLeft",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_ENCLOSURE_X_CTR, _ENCLOSURE_Y_MIN, _ENCLOSURE_TOP_Z / 2)),
        spawn=sim_utils.CuboidCfg(size=(_ENCLOSURE_X_SPAN, 0.003, _ENCLOSURE_TOP_Z), visual_material=_ENCLOSURE_MATERIAL),
    )
    enclosure_right: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/EnclosureRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_ENCLOSURE_X_CTR, _ENCLOSURE_Y_MAX, _ENCLOSURE_TOP_Z / 2)),
        spawn=sim_utils.CuboidCfg(size=(_ENCLOSURE_X_SPAN, 0.003, _ENCLOSURE_TOP_Z), visual_material=_ENCLOSURE_MATERIAL),
    )
    enclosure_top: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/EnclosureTop",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_ENCLOSURE_X_CTR, _ENCLOSURE_Y_CTR, _ENCLOSURE_TOP_Z)),
        spawn=sim_utils.CuboidCfg(size=(_ENCLOSURE_X_SPAN, _ENCLOSURE_Y_SPAN, 0.003), visual_material=_ENCLOSURE_MATERIAL),
    )
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_TABLE_POS, rot=_TABLE_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=_TABLE_USD_PATH,
            scale=_TABLE_SCALE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
    # Graspable box -- dynamic rigid body, same rigid/mass/collision pattern
    # as pick_place_env_cfg.py's _cuboid_object_cfg, but referencing the
    # imported block.usd mesh instead of a procedural CuboidCfg.
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_BOX_POS),
        spawn=sim_utils.UsdFileCfg(
            usd_path=_BOX_USD_PATH,
            scale=(0.9, 0.9, 0.9),  # slightly smaller than the native 5.08cm cube, per live feedback
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
    # Container -- kinematic (doesn't get knocked around, just needs to
    # collide with the box), same pattern as pick_place_env_cfg.py's
    # _tray_cfg.
    container: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Container",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CONTAINER_POS, rot=_CONTAINER_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=_CONTAINER_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


class QuestRosReceiver(Node):
    """ROS 2 subscriber that caches the latest /quest_teleop message."""

    def __init__(self) -> None:
        super().__init__("quest_ik_listener")
        self._latest: QuestHandPose | None = None
        self._lock = threading.Lock()
        self.create_subscription(QuestHandPose, "/quest_teleop", self._cb, 1)
        self.get_logger().info("Subscribed to /quest_teleop")

    def _cb(self, msg: QuestHandPose) -> None:
        with self._lock:
            self._latest = msg

    def poll(self) -> QuestHandPose | None:
        with self._lock:
            return self._latest


def _joint_ids(robot, names: list[str]) -> list[int]:
    name_to_id = {n: i for i, n in enumerate(robot.data.joint_names)}
    return [name_to_id[resolve_joint_name(robot, n)] for n in names]


def _make_entity_cfg(scene, joint_names: list[str], ee_body: str) -> SceneEntityCfg:
    cfg = SceneEntityCfg("robot", joint_names=joint_names, body_names=[ee_body])
    cfg.resolve(scene)
    return cfg


def _wrist_xyz(wrist) -> torch.Tensor:
    p = wrist.position
    return torch.tensor([p.x, p.y, p.z], dtype=torch.float32)


def _wrist_quat_wxyz(wrist) -> torch.Tensor:
    o = wrist.orientation
    return torch.tensor([o.w, o.x, o.y, o.z], dtype=torch.float32)


def _ramped_gain(disp_m: torch.Tensor, gain_near: float) -> torch.Tensor:
    t = ((disp_m - _GAIN_RAMP_START_M) / (_GAIN_RAMP_END_M - _GAIN_RAMP_START_M)).clamp(0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return gain_near + (_GAIN_FAR - gain_near) * t


def _is_tracked(pos: torch.Tensor, quat_wxyz: torch.Tensor, eps: float = 1e-6) -> bool:
    """True unless this is WebXR's "untracked hand" sentinel: zero position
    + identity orientation."""
    return not (
        torch.allclose(pos, torch.zeros_like(pos), atol=eps)
        and torch.allclose(quat_wxyz, torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat_wxyz.device), atol=eps)
    )


def _pinch_dist(hand_joints: list) -> float:
    if len(hand_joints) != 75:
        return float("inf")
    ti, ii = _THUMB_TIP_IDX * 3, _INDEX_TIP_IDX * 3
    thumb = torch.tensor(hand_joints[ti:ti + 3])
    index = torch.tensor(hand_joints[ii:ii + 3])
    return (thumb - index).norm().item()


def _publish_real_left_arm_pose(pub, clock_node, joint_pos_des_rad) -> None:
    """Build and publish an ArmPose for the real left arm from a DLS
    solution. Field layout/units copied EXACTLY from
    src/teleop/task_space_controller/task_space_ik.py's
    publish_joint_pos -- same LEFT_ARM_JOINTS order (joint1L, joint2l,
    joint3l, joint4l, joint5l, joint6l) maps 1:1 to
    shoulder(flexion,abduction,rotation) / elbow(flexion,forearm_rotation) /
    wrist(extension), degrees (hardware_mapping.yaml + the CAN PositionDeg
    signal expect degrees, joint_pos_des is radians)."""
    import math

    from common_msgs.msg import ArmPose as WatoArmPose, JointState as WatoJointState

    q = [math.degrees(v) for v in joint_pos_des_rad[0].tolist()]

    msg = WatoArmPose()
    msg.header.stamp = clock_node.get_clock().now().to_msg()

    shoulder = WatoJointState()
    shoulder.position = [q[0], q[1], q[2]]
    msg.shoulder = shoulder

    elbow = WatoJointState()
    elbow.position = [q[3], q[4]]
    msg.elbow = elbow

    wrist = WatoJointState()
    wrist.position = [q[5]]
    msg.wrist = wrist

    pub.publish(msg)


def _clamp_step(target: torch.Tensor, previous: torch.Tensor, delta_max) -> torch.Tensor:
    """Direct port of joint_command_core.cpp's clampStep -- a generic
    rate-limit-toward-target. delta_max may be a scalar or a tensor matching
    target's shape."""
    return previous + torch.clamp(target - previous, -delta_max, delta_max)


def _adaptive_dls_lambda(
    jacobian: torch.Tensor, lambda_min: float, lambda_max: float, epsilon: float,
) -> tuple[float, float]:
    """Chiaverini adaptive damping for DLS IK. CURRENTLY UNCALLED -- see _DLS_LAMBDA_MAX.

    manipulability = sqrt(det(J J^T)), the standard distance-from-singularity measure (0 =
    singular). Ramps lambda continuously from lambda_min at/above epsilon toward lambda_max as
    manipulability -> 0, so damping never jumps frame-to-frame. Returns (lambda, manipulability)."""
    jjt = jacobian @ jacobian.transpose(-2, -1)
    manipulability = torch.sqrt(torch.clamp(torch.linalg.det(jjt), min=0.0)).item()
    if manipulability >= epsilon:
        return lambda_min, manipulability
    ratio = manipulability / epsilon
    lambda_val = math.sqrt(lambda_min**2 + (1.0 - ratio**2) * (lambda_max**2 - lambda_min**2))
    return lambda_val, manipulability


class _OneEuroFilter:
    """1 Euro Filter (Casiez et al., CHI 2012) on the RAW Quest wrist POSITION, before it
    drives the IK target.

    Smooths heavily when nearly still (kills tracking jitter) and relaxes during fast motion
    (adds ~no lag to deliberate movement) -- a fixed-cutoff low-pass must pick one tradeoff.
    Separate concern from _smooth_damp, which smooths joint COMMANDS after the solve."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff  # Hz -- baseline smoothing strength when nearly still
        self.beta = beta  # how aggressively speed relaxes the cutoff (higher = less lag, more residual jitter during fast motion)
        self.d_cutoff = d_cutoff  # Hz -- smooths the velocity estimate itself, so cutoff adaptation isn't itself noisy
        self.x_prev: torch.Tensor | None = None
        self.dx_prev: torch.Tensor | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        if self.x_prev is None:
            self.x_prev = x.clone()
            self.dx_prev = torch.zeros_like(x)
            return x.clone()
        dt = max(dt, 1e-4)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        edx = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * edx.abs()
        a = self._alpha(cutoff, dt)
        x_filtered = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_filtered
        self.dx_prev = edx
        return x_filtered

    def reset(self) -> None:
        """Called on recalibration (R key) / scene reset (T key) so a teleport-style jump in
        raw position doesn't get smoothed into a slow drift -- the next filter() call reseeds
        state exactly like the first-ever call."""
        self.x_prev = None
        self.dx_prev = None


class _OneEuroQuatFilter:
    """Same as _OneEuroFilter but for wrist ORIENTATION, blending via quat_slerp.

    Orientation was unfiltered for a long time while position was not, which is what "wrist
    rotation isn't smooth at all" turned out to be. SLERP rather than a per-component low-pass,
    which would need renormalising and ignores the unit-quaternion manifold."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff  # Hz -- baseline smoothing strength when nearly still
        self.beta = beta  # how aggressively angular speed relaxes the cutoff
        self.d_cutoff = d_cutoff  # Hz -- smooths the angular-velocity estimate itself
        self.q_prev: torch.Tensor | None = None
        self.angvel_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, q: torch.Tensor, dt: float) -> torch.Tensor:
        if self.q_prev is None:
            self.q_prev = q.clone()
            self.angvel_prev = 0.0
            return q.clone()
        dt = max(dt, 1e-4)
        # Angular distance between the last filtered quat and this new raw sample, as the
        # "speed" signal driving the adaptive cutoff (same role dx plays in _OneEuroFilter).
        dot = torch.clamp(torch.dot(self.q_prev, q).abs(), -1.0, 1.0)
        angle = 2.0 * torch.acos(dot).item()
        angvel = angle / dt
        a_d = self._alpha(self.d_cutoff, dt)
        smoothed_angvel = a_d * angvel + (1.0 - a_d) * self.angvel_prev
        cutoff = self.min_cutoff + self.beta * smoothed_angvel
        a = self._alpha(cutoff, dt)
        q_filtered = quat_slerp(self.q_prev, q, a)
        self.q_prev = q_filtered
        self.angvel_prev = smoothed_angvel
        return q_filtered

    def reset(self) -> None:
        """See _OneEuroFilter.reset's docstring -- same reasoning, called from the same places."""
        self.q_prev = None
        self.angvel_prev = 0.0


def _smooth_damp(
    current: torch.Tensor, current_vel: torch.Tensor, target: torch.Tensor,
    smooth_time: float, max_speed: float, dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Critically-damped spring smoothing (Unity's Mathf.SmoothDamp; Game Programming Gems 4).

    Tracks position AND velocity so deceleration eases in as the target is approached, instead of
    the bang-bang profile a hard rate clamp produces. max_speed is still enforced, just smoothly.
    Returns (new_position, new_velocity); the caller must carry the velocity into the next call.
    Currently used only for the grippers -- the arm joints were reverted to unsmoothed IK."""
    smooth_time = max(1e-4, smooth_time)
    omega = 2.0 / smooth_time

    x = omega * dt
    exp_val = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)  # stable exp(-omega*dt) approx, same as Unity's impl

    max_change = abs(max_speed) * smooth_time
    change = torch.clamp(current - target, -max_change, max_change)
    clamped_target = current - change

    temp = (current_vel + omega * change) * dt
    new_vel = (current_vel - omega * temp) * exp_val
    new_pos = clamped_target + (change + temp) * exp_val

    # Prevent overshoot: if we'd cross past the (pre-speed-clamp) target, snap to it instead.
    orig_to_current = target - current
    out_to_orig = new_pos - target
    overshot = (orig_to_current * out_to_orig) > 0
    new_pos = torch.where(overshot, target, new_pos)
    new_vel = torch.where(overshot, (target - current) / dt, new_vel)

    return new_pos, new_vel


def _step_velocity_and_delta_clamp(
    target: torch.Tensor, prev_pos: torch.Tensor, velocity_max: float, delta_max: float, dt: float,
) -> torch.Tensor:
    """Mirrors joint_command_core.cpp's non-trapezoidal moderation path:
    velocity-limit clamp (velocity_max converted to a per-frame step via dt)
    then a fixed delta-limit clamp, applied in that order."""
    velocity_step = abs(velocity_max) * dt
    stepped = _clamp_step(target, prev_pos, velocity_step)
    stepped = _clamp_step(stepped, prev_pos, abs(delta_max))
    return stepped


def _ee_pose_in_base(robot, body_id: int):
    root_pose_w = robot.data.root_state_w[:, 0:7]
    ee_pose_w = robot.data.body_state_w[:, body_id, 0:7]
    return subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
    )


class _ArmDlsController:
    """A DifferentialIKController plus the per-arm state to drive it from Quest wrist data:
    entity/jacobian indices, fingertip geometry, homing state, current target.

    One instance per arm rather than one controller with num_envs=2, so neither arm's
    homing/recalibration state can leak into the other's."""

    def __init__(self, scene, robot, device, arm_joint_names, ee_body, finger_tip_bodies, finger_distal_local,
                 lambda_val=_DLS_LAMBDA, lambda_max=_DLS_LAMBDA_MAX):
        self.finger_tip_bodies = finger_tip_bodies
        self.finger_distal_local = finger_distal_local

        entity_cfg = _make_entity_cfg(scene, arm_joint_names, ee_body)
        self.body_id = entity_cfg.body_ids[0]
        self.arm_ids = entity_cfg.joint_ids
        self.ee_jacobi_idx = self.body_id - 1 if robot.is_fixed_base else self.body_id
        self.finger_body_ids = resolve_body_ids(robot, finger_tip_bodies)

        cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
            ik_params={"lambda_val": lambda_val},
        )
        self.arm_joint_names = [robot.data.joint_names[i] for i in self.arm_ids]
        self.controller = DifferentialIKController(
            cfg, num_envs=scene.num_envs, device=device,
        )
        self.controller.reset(env_ids=torch.arange(scene.num_envs, device=device))
        # Floor/ceiling for adaptive damping. Inert while _adaptive_dls_lambda is uncalled.
        self.lambda_min = lambda_val
        self.lambda_max = lambda_max
        self.last_manipulability: float | None = None

        self.quest_home_xyz: torch.Tensor | None = None
        self.quest_home_quat: torch.Tensor | None = None
        self.home_tip_pos_b: torch.Tensor | None = None
        self.home_tip_quat_b: torch.Tensor | None = None
        self.wrist_orient_offset: torch.Tensor | None = None
        # Latest DLS solution (radians), read by the real-hardware bridge. Deliberately the RAW
        # solution -- the real arm smooths independently in joint_command_core.cpp, so smoothing
        # here too would only add bridge latency.
        self.last_joint_pos_des: torch.Tensor | None = None
        # _smooth_damp state. Unused since the revert to unsmoothed IK output; kept for a retry.
        self.smoothed_pos: torch.Tensor | None = None
        self.smoothed_vel: torch.Tensor | None = None
        # Input-side jitter filters on the RAW Quest wrist pose, applied before homing and
        # displacement. Distinct from the output smoothing above.
        self.pos_filter = _OneEuroFilter()
        self.quat_filter = _OneEuroQuatFilter()

    def tip_pose_b(self, robot, root_pose_w):
        return compute_gripper_tip_pose_b(
            robot, root_pose_w, self.body_id, self.finger_body_ids,
            self.finger_tip_bodies, self.finger_distal_local,
        )

    def solve_and_apply(self, robot, device, target_pos_b, target_quat_b, dt: float):
        root_pose_w = robot.data.root_state_w[:, 0:7]
        wrist_pos_b, _ = _ee_pose_in_base(robot, self.body_id)
        tip_pos_b, tip_quat_b = self.tip_pose_b(robot, root_pose_w)
        jacobian_w = robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.arm_ids]
        jacobian_b = compute_tip_ik_jacobian(robot, jacobian_w, wrist_pos_b, tip_pos_b)

        # Fixed lambda and no output smoothing, matching run_quest_bimanual_teleop.py exactly --
        # adaptive damping and _smooth_damp were both tried here and made the IK feel wrong.
        # dt is unused, kept for signature compatibility.
        self.controller.set_command(torch.cat([target_pos_b, target_quat_b], dim=1))
        joint_pos = robot.data.joint_pos[:, self.arm_ids]
        joint_pos_des = self.controller.compute(tip_pos_b, tip_quat_b, jacobian_b, joint_pos)
        self.last_joint_pos_des = joint_pos_des

        # Position target only. This used to ALSO write_joint_state_to_sim(joint_pos_des, 0),
        # teleporting the joints every step: physics never ran on the arm, so gravity, contact
        # and every effort_limit_sim/stiffness value in BIMANUAL_ARM_CFG did nothing, and recorded
        # observation.state was just a copy of the action. The actuators drive the arm now, so
        # those gains are live -- retune them, not this, if tracking lags.
        robot.set_joint_position_target(joint_pos_des, joint_ids=self.arm_ids)
        return tip_pos_b, tip_quat_b


# ── recording (see dataset_schema_wato_arm_v2_push_box.yaml) ─────────────────

def _init_recorder(device: str):
    """Mirrors so101_leader_teleop.py's _init_recorder. Returns (recorder, schema_cfg)
    or (None, None) if --record wasn't passed."""
    if not args_cli.record:
        return None, None
    _ensure_il_on_path()
    try:
        from humanoid_il.record_utils import resolve_config_path
        from humanoid_il.schema import enabled_images, load_yaml
        from humanoid_il.sim_recorder import SimLeRobotRecorder
    except ImportError as exc:
        raise ImportError("Recording requires humanoid-il. Install with:\n  pip install -e src/il[sim]") from exc

    schema_path = resolve_config_path(args_cli.schema, anchor=_IL_PKG)
    cfg = load_yaml(schema_path)
    dataset_root = (
        Path(args_cli.dataset_root) if args_cli.dataset_root
        else Path((cfg.get("record") or {}).get("root", "datasets/record_wato_arm_v2_push_box"))
    )
    cameras = {name: {"height": spec["height"], "width": spec["width"]} for name, spec in enabled_images(cfg).items()}
    # device="cpu" always: the recorder's multi-GB circular frame buffer on "cuda" competes with
    # Isaac Sim's own GPU memory and silently kills the process. buffer_capacity_s=30 (not the
    # 120 default): at 2x 640x480x30fps, 120s is ~20GB RSS and hung the host twice; episodes run
    # 15-20s. Overrunning it drops frames with a warning, no crash.
    recorder = SimLeRobotRecorder(
        task_name=args_cli.task_description,
        repo_id=str(cfg.get("repo_id", "humanoid/wato_arm_v2_push_box")),
        dataset_root=dataset_root,
        fps=_RECORD_FPS,     # render cadence, not the schema -- see _RECORD_FPS
        rate_limit=False,    # the render gate is the sole decimator
        device="cpu",
        joint_names=list(cfg["joint_names"]),
        cameras=cameras,
        num_episodes=args_cli.num_episodes,
        buffer_capacity_s=30.0,
        # else the dataset records itself as an SO-101 (recorder default)
        robot_type=str(cfg.get("robot_type", "wato_arm_v2")),
    )
    recorder.init_dataset()
    print(f"[RECORD] Writing to {dataset_root} at {_RECORD_FPS} fps "
          f"(1 frame / {_POV_CAPTURE_EVERY_N_STEPS} physics steps)", flush=True)
    # Key bindings differ from humanoid_il's defaults -- see run_simulator's _on_keyboard_event.
    return recorder, cfg


def _capture_record_images(scene: InteractiveScene) -> dict:
    """{schema_key: HxWx3 uint8 rgb} for the two data-collection cameras --
    NOT the RSD455 headset pair, see module docstring."""
    images = {}
    for schema_key, scene_key in (("ego", "ego_cam"), ("wrist", "wrist_cam")):
        rgb = scene[scene_key].data.output["rgb"]
        frame = rgb[0].detach().cpu().numpy()
        if frame.shape[-1] > 3:
            frame = frame[..., :3]
        images[schema_key] = frame.astype("uint8")
    return images


def _write_wrist_cam_hud_frames(scene: InteractiveScene) -> None:
    """Both wrist cams' current frames -> the headset HUD panels, from the same sensor tensors
    recording uses (no extra viewport render). JPEG, not PNG: as a PNG this was ~155KB, ten times
    the two eye JPEGs combined and the largest thing the headset polled."""
    _save_frame_atomic(_camera_rgb_frame(scene["wrist_cam"]), _WRIST_CAM_FRAME_PATH_LEFT)
    _save_frame_atomic(_camera_rgb_frame(scene["wrist_cam_right"]), _WRIST_CAM_FRAME_PATH_RIGHT)


# ── main simulation loop ──────────────────────────────────────────────────────

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    device = sim.device
    sim_dt = sim.get_physics_dt()
    gain = args_cli.gain

    scene.update(sim_dt)
    apply_joint_limits(robot)
    _set_rigid_body_friction(scene["box"], _BOX_STATIC_FRICTION, _BOX_DYNAMIC_FRICTION)
    _set_rigid_body_friction(
        robot, _GRIPPER_STATIC_FRICTION, _GRIPPER_DYNAMIC_FRICTION,
        body_names=(*LEFT_FINGER_TIP_BODIES, *RIGHT_FINGER_TIP_BODIES),
    )

    _root_pose_w_diag = robot.data.root_state_w[0, :7].tolist()
    print(f"[Quest][diag] robot root world pose (pos xyz, quat wxyz)={_root_pose_w_diag}", flush=True)

    _base_link_prim_path = "/World/envs/env_0/Robot/base_link"  # num_envs is always 1 in this script
    _base_link_prim = omni.usd.get_context().get_stage().GetPrimAtPath(_base_link_prim_path)
    print(f"[Quest] base_link prim valid={_base_link_prim.IsValid()} "
          f"authored={_base_link_prim.HasAuthoredReferences() or _base_link_prim.GetTypeName() != ''} "
          f"body_names={list(robot.data.body_names)}", flush=True)
    left_eye_mount = _attach_rsd455_camera(
        _base_link_prim_path, "left_eye_camera_mount", _HEAD_VIEWPOINT_HOME_POS, _HEAD_VIEWPOINT_HOME_QUAT
    )
    right_eye_mount = _attach_rsd455_camera(
        _base_link_prim_path, "right_eye_camera_mount", _HEAD_VIEWPOINT_HOME_POS, _HEAD_VIEWPOINT_HOME_QUAT
    )
    print(f"[Quest] Stereo head-tracked RealSense D455 pair attached: {left_eye_mount}, {right_eye_mount}", flush=True)
    if _SHOW_WRIST_CAM_MARKER:
        _spawn_wrist_cam_marker("/World/envs/env_0/Robot/link6l/wrist_cam_marker", WRIST_CAM_POS)
        print(f"[Quest] wrist_cam calibration marker ON (2cm green sphere at {WRIST_CAM_POS} "
              "on link6l) -- set _SHOW_WRIST_CAM_MARKER=False to hide.", flush=True)
    # Read the native FOV BEFORE widening it for the headset display.
    _rsd455_native_fov = _read_camera_fov(f"{left_eye_mount}/{_RSD455_CAMERA_SUBPATH}")
    _widen_camera_fov(f"{left_eye_mount}/{_RSD455_CAMERA_SUBPATH}", _RSD455_WIDENED_FOCAL_LENGTH)
    _widen_camera_fov(f"{right_eye_mount}/{_RSD455_CAMERA_SUBPATH}", _RSD455_WIDENED_FOCAL_LENGTH)
    left_eye_camera = _open_pov_camera(f"{left_eye_mount}/{_RSD455_CAMERA_SUBPATH}", "Left Eye POV")
    right_eye_camera = _open_pov_camera(f"{right_eye_mount}/{_RSD455_CAMERA_SUBPATH}", "Right Eye POV")
    # Cached for _project_world_point_to_uv; valid because _widen_camera_fov already blocked
    # until these prims finished loading.
    _stage_for_cams = omni.usd.get_context().get_stage()
    _left_eye_cam_prim = _stage_for_cams.GetPrimAtPath(f"{left_eye_mount}/{_RSD455_CAMERA_SUBPATH}")
    _right_eye_cam_prim = _stage_for_cams.GetPrimAtPath(f"{right_eye_mount}/{_RSD455_CAMERA_SUBPATH}")
    # ego_cam gets the RSD455's aperture but the WIDENED focal length -- native FOV is too
    # zoomed in to be usable. Extra Kit preview windows for ego_cam/wrist_cam were removed here:
    # they render on every app tick regardless of the render gate, and cost real time.
    #
    # args_cli.record is the same flag main() uses to decide whether ego_cam exists -- without
    # it there is no prim to write to. Keep the two conditions identical.
    _ego_cam_prim_path = f"{_base_link_prim_path}/ego_cam"
    if args_cli.record and _rsd455_native_fov is not None:
        _ego_cam_fov = (_RSD455_WIDENED_FOCAL_LENGTH, _rsd455_native_fov[1], _rsd455_native_fov[2])
        _set_camera_fov(_ego_cam_prim_path, *_ego_cam_fov)
        print(f"[Quest] ego_cam FOV set to WIDENED RSD455 (not native) "
              f"(focalLength/hAperture/vAperture)={_ego_cam_fov}", flush=True)
    if left_eye_camera is not None and right_eye_camera is not None:
        _POV_STATIC_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[Quest] Stereo POV feed will be captured to {_POV_FRAME_PATH_LEFT} / {_POV_FRAME_PATH_RIGHT} "
              f"(served by webxr_server.py's static handler at /pov_left.jpg, /pov_right.jpg)", flush=True)

    left_arm_names = [resolve_joint_name(robot, n) for n in LEFT_ARM_JOINTS]
    right_arm_names = [resolve_joint_name(robot, n) for n in RIGHT_ARM_JOINTS]

    left_arm = _ArmDlsController(
        scene, robot, device, left_arm_names, LEFT_EE_BODY,
        LEFT_FINGER_TIP_BODIES, LEFT_FINGER_DISTAL_TIP_LOCAL,
    )
    right_arm = _ArmDlsController(
        scene, robot, device, right_arm_names, RIGHT_EE_BODY,
        RIGHT_FINGER_TIP_BODIES, RIGHT_FINGER_DISTAL_TIP_LOCAL,
        lambda_val=_DLS_LAMBDA_RIGHT, lambda_max=_DLS_LAMBDA_MAX_RIGHT,
    )

    quest_to_world = _QUEST_TO_WORLD.to(device)
    quest_to_world_quat = quat_from_matrix(quest_to_world.unsqueeze(0))  # RETIRED for orientation, see comment above _QUEST_TO_WORLD
    quest_to_cam_local = _QUEST_TO_CAM_LOCAL.to(device)
    quest_to_cam_local_quat = quat_from_matrix(quest_to_cam_local.unsqueeze(0))
    axis_sign_left = _AXIS_SIGN_LEFT.to(device)
    axis_sign_right = _AXIS_SIGN_RIGHT.to(device)
    # Camera-relative basis in world frame: the rsd455's local axes rotated by the mount tilt.
    # Hand position deltas are composed along these, not along fixed world axes.
    camera_tilt_quat = torch.tensor([_HEAD_VIEWPOINT_HOME_QUAT], dtype=torch.float32, device=device)
    cam_fwd_world = quat_apply(camera_tilt_quat, _CAM_LOCAL_FORWARD.to(device).unsqueeze(0)).squeeze(0)
    cam_up_world = quat_apply(camera_tilt_quat, _CAM_LOCAL_UP.to(device).unsqueeze(0)).squeeze(0)
    cam_right_world = quat_apply(camera_tilt_quat, _CAM_LOCAL_RIGHT.to(device).unsqueeze(0)).squeeze(0)
    left_arm.wrist_orient_offset = _WRIST_ORIENT_OFFSET_LEFT.to(device).unsqueeze(0)
    right_arm.wrist_orient_offset = _WRIST_ORIENT_OFFSET_RIGHT.to(device).unsqueeze(0)

    left_gripper_ids = _joint_ids(robot, LEFT_GRIPPER_JOINTS)
    right_gripper_ids = _joint_ids(robot, RIGHT_GRIPPER_JOINTS)

    left_g_open = torch.tensor([[LEFT_GRIPPER_OPEN["joint7l"], LEFT_GRIPPER_OPEN["joint8l"]]], device=device)
    left_g_closed = torch.tensor([[LEFT_GRIPPER_CLOSED["joint7l"], LEFT_GRIPPER_CLOSED["joint8l"]]], device=device)
    right_g_open = torch.tensor([[RIGHT_GRIPPER_OPEN["joint7"], RIGHT_GRIPPER_OPEN["joint8"]]], device=device)
    right_g_closed = torch.tensor([[RIGHT_GRIPPER_CLOSED["joint7"], RIGHT_GRIPPER_CLOSED["joint8"]]], device=device)
    # The open/closed DECISION stays binary at the pinch threshold, but the resulting joint
    # TARGET is smoothed -- snapping it produced a visible discontinuity in recorded demos.
    left_gripper_smoothed = left_g_open.clone()
    left_gripper_vel = torch.zeros_like(left_g_open)
    right_gripper_smoothed = right_g_open.clone()
    right_gripper_vel = torch.zeros_like(right_g_open)

    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    root_pose_w0 = robot.data.root_state_w[:, 0:7]
    init_tip_pos_b_l, init_tip_quat_b_l = left_arm.tip_pose_b(robot, root_pose_w0)
    init_tip_pos_b_r, init_tip_quat_b_r = right_arm.tip_pose_b(robot, root_pose_w0)
    print(f"[Quest][diag] rest EE pos (base frame, from FK of default joint pose) "
          f"L(tip)={init_tip_pos_b_l[0].tolist()} R(tip)={init_tip_pos_b_r[0].tolist()}", flush=True)
    init_tip_pos_b_l[:, 0] += _HOME_TIP_X_OFFSET
    init_tip_pos_b_r[:, 0] += _HOME_TIP_X_OFFSET
    init_tip_pos_b_l[:, 2] += _HOME_TIP_Z_OFFSET
    init_tip_pos_b_r[:, 2] += _HOME_TIP_Z_OFFSET
    print(f"[Quest][diag] rest EE pos AFTER _HOME_TIP_X_OFFSET={_HOME_TIP_X_OFFSET}/"
          f"_HOME_TIP_Z_OFFSET={_HOME_TIP_Z_OFFSET} (IK target, not yet solved) "
          f"L(tip)={init_tip_pos_b_l[0].tolist()} R(tip)={init_tip_pos_b_r[0].tolist()}", flush=True)

    rclpy.init()
    receiver = QuestRosReceiver()
    threading.Thread(target=rclpy.spin, args=(receiver,), daemon=True).start()

    # Real-hardware bridge, left arm only. Publishes on the SAME node/spin-thread as the Quest
    # receiver, not a second rclpy context. See --publish-real-left-arm for the safety rationale.
    real_left_arm_pub = None
    real_left_arm_elapsed_s = 0.0
    real_left_arm_since_publish_s = 0.0
    real_left_arm_started = False
    if args_cli.publish_real_left_arm:
        from common_msgs.msg import ArmPose as WatoArmPose, JointState as WatoJointState

        real_left_arm_pub = receiver.create_publisher(WatoArmPose, "/arm/joint_targets", 10)
        print(f"[Quest][REAL HARDWARE] Left arm will start publishing to /arm/joint_targets in "
              f"{_REAL_ARM_PUBLISH_START_DELAY_S:.0f}s. Position the REAL left arm near the sim's "
              f"rest pose NOW. Right arm is NOT published (no hardware CAN mapping exists).", flush=True)

    # Targets applied every frame (unconditionally), updated only while tracked.
    target_pos_b_left = init_tip_pos_b_l.clone()
    target_quat_b_left = init_tip_quat_b_l.clone()
    target_pos_b_right = init_tip_pos_b_r.clone()
    target_quat_b_right = init_tip_quat_b_r.clone()
    # Only reassigned inside "if msg is not None:" below, but read unconditionally every frame
    # (marker UV projection) -- initialized here so that holds even before the first message.
    root_quat_w = robot.data.root_state_w[:, 3:7]

    # Head-tracked stereo viewpoint state -- same home-on-first-tracked-sample
    # pattern as the arms above, homed independently (a person can start
    # moving their head before their wrists are tracked or vice versa).
    head_home_xyz: torch.Tensor | None = None
    head_home_quat: torch.Tensor | None = None
    head_home_viewpoint_pos_b = torch.tensor([_HEAD_VIEWPOINT_HOME_POS], dtype=torch.float32, device=device)
    head_home_viewpoint_quat_b = torch.tensor([_HEAD_VIEWPOINT_HOME_QUAT], dtype=torch.float32, device=device)
    head_viewpoint_pos_b = head_home_viewpoint_pos_b.clone()
    head_viewpoint_quat_b = head_home_viewpoint_quat_b.clone()
    eye_local_right = _EYE_LOCAL_RIGHT.to(device)
    eye_local_forward = _EYE_LOCAL_FORWARD.to(device)
    ego_cam_convention_fix_quat = torch.tensor([_EGO_CAM_CONVENTION_FIX_QUAT], dtype=torch.float32, device=device)
    ego_cam_extra_tilt_quat = torch.tensor([_EGO_CAM_EXTRA_TILT_QUAT], dtype=torch.float32, device=device)

    def _sync_camera_mounts() -> None:
        """Aim the stereo eye mounts (and ego_cam) at the current head viewpoint.

        NOT cheap: three USD transform authorings, each firing change notifications through Kit's
        render graph. Running it per physics step cost ~20ms/step while writing identical values,
        since the head viewpoint is static. Called once at startup; _HEAD_TRACKING_LIVE restores
        per-step tracking when head pose is actually driven."""
        right_offset_b = quat_apply(head_viewpoint_quat_b, eye_local_right.unsqueeze(0)) * (_EYE_IPD_M / 2)
        left_eye_pos_b = (head_viewpoint_pos_b - right_offset_b)[0].tolist()
        right_eye_pos_b = (head_viewpoint_pos_b + right_offset_b)[0].tolist()
        head_orient_b = head_viewpoint_quat_b[0].tolist()
        _set_mount_pose(left_eye_mount, tuple(left_eye_pos_b), tuple(head_orient_b))
        _set_mount_pose(right_eye_mount, tuple(right_eye_pos_b), tuple(head_orient_b))
        # ego_cam pinned to the left eye's pose so recorded frames match what the operator saw,
        # nudged clear of the RSD455 housing mesh. Skipped without --record: main() leaves the
        # sensor out of the scene entirely then.
        if _EGO_CAM_FREE_MOVE_DEBUG or not args_cli.record:
            return
        forward_offset_b = quat_apply(head_viewpoint_quat_b, eye_local_forward.unsqueeze(0)) * _EGO_CAM_FORWARD_OFFSET_M
        ego_cam_pos_b = (head_viewpoint_pos_b - right_offset_b + forward_offset_b)[0].tolist()
        ego_cam_pos_b[2] += _EGO_CAM_UP_OFFSET_M  # straight base-frame +Z, per live feedback
        ego_cam_orient_b = quat_mul(
            ego_cam_extra_tilt_quat, quat_mul(head_viewpoint_quat_b, ego_cam_convention_fix_quat)
        )[0].tolist()
        _set_mount_pose(_ego_cam_prim_path, tuple(ego_cam_pos_b), tuple(ego_cam_orient_b))

    _sync_camera_mounts()

    left_closed = False
    right_closed = False

    recorder, record_cfg = _init_recorder(device)
    if recorder is not None:
        # NOT recorder.start_keyboard() -- that attaches humanoid_il's shared pynput listener,
        # whose bindings other teleop scripts rely on. S/D are handled through the same
        # carb.input listener as R/T instead, calling save_episode()/cancel_recording() directly
        # so recording continues straight into the next demo. EpisodeFlags is still needed
        # (tick() no-ops without it) but is constructed directly.
        from humanoid_il.episode_keys import EpisodeFlags
        recorder._flags = EpisodeFlags(start=False)

    def _sphere_cfg(color, radius, opacity=1.0):
        return sim_utils.SphereCfg(
            radius=radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity),
        )

    left_target_vis = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/left_ik_target",
        markers={"sphere": _sphere_cfg((0.2, 0.5, 1.0), 0.025, opacity=0.3)},  # was 0.05, halved per live feedback
    ))
    right_target_vis = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/right_ik_target",
        markers={"sphere": _sphere_cfg((1.0, 0.3, 0.1), 0.025, opacity=0.3)},  # was 0.05, halved per live feedback
    ))
    if args_cli.record:
        # VisualizationMarkers instantiates its spheres at init regardless of whether
        # .visualize() is called, at (0,0,0) and NOT hidden -- so skipping .visualize() is not
        # enough. USD visibility IS always respected (unlike cameraVisibility, below), so force
        # it here as the actual guarantee that markers stay out of recorded frames.
        _stage_for_markers = omni.usd.get_context().get_stage()
        for _mp in ("/Visuals/left_ik_target", "/Visuals/right_ik_target"):
            _mprim = _stage_for_markers.GetPrimAtPath(_mp)
            if _mprim.IsValid():
                UsdGeom.Imageable(_mprim).MakeInvisible()
        print("[Quest] --record is active: IK target markers made invisible (guaranteed, not "
              "just unpositioned) to keep them out of recorded frames.", flush=True)

    # Try to hide the IK target markers from wrist_cam's recorded render. cameraVisibility
    # belongs on the RenderProduct prim, not the camera prim (CameraCfg sensors are Replicator
    # render products, a separate pipeline from Kit's Hydra viewports) -- applying it to the
    # camera prim silently does nothing. Even applied correctly this RTX version appears not to
    # honour it, which is why the args_cli.record block above force-hides the markers instead.
    _stage = omni.usd.get_context().get_stage()
    for _marker_path in ("/Visuals/left_ik_target", "/Visuals/right_ik_target"):
        _marker_prim = _stage.GetPrimAtPath(_marker_path)
        if _marker_prim.IsValid():
            UsdGeom.Imageable(_marker_prim).GetPurposeAttr().Set(UsdGeom.Tokens.guide)
    for _wrist_key in ("wrist_cam", "wrist_cam_right"):
        for _rp_path in scene[_wrist_key].render_product_paths:
            _rp_prim = _stage.GetPrimAtPath(_rp_path)
            if _rp_prim.IsValid():
                _cam_vis_collection = Usd.CollectionAPI.Apply(_rp_prim, "cameraVisibility")
                _cam_vis_collection.CreateIncludeRootAttr().Set(True)  # "/" isn't a valid rel target
                _excludes_rel = _cam_vis_collection.CreateExcludesRel()
                _excludes_rel.AddTarget("/Visuals/left_ik_target")
                _excludes_rel.AddTarget("/Visuals/right_ik_target")
                print(f"[Quest] Excluded both IK target markers from {_wrist_key}'s RenderProduct "
                      f"({_rp_path}) cameraVisibility", flush=True)
            else:
                print(f"[Quest] WARNING: {_wrist_key} RenderProduct prim {_rp_path} not valid -- "
                      f"marker exclusion did not apply", flush=True)

    def _recalibrate() -> None:
        nonlocal head_home_xyz, head_home_quat
        left_arm.quest_home_xyz = None
        right_arm.quest_home_xyz = None
        head_home_xyz = None
        head_home_quat = None
        # Otherwise the filter's stale x_prev (from before recalibration) makes the next real
        # sample look like a huge, fast motion -- momentarily defeating its own jitter smoothing
        # right when a fresh, clean home anchor is what's needed most.
        left_arm.pos_filter.reset()
        right_arm.pos_filter.reset()
        left_arm.quat_filter.reset()
        right_arm.quat_filter.reset()
        print("[Quest] Recalibrating — hold wrists in a comfortable pose.", flush=True)

    def _reset_scene() -> None:
        """Full scene reset (distinct from R's recalibration, which only re-anchors tracking):
        box/container back to their original spawn pose+velocity, arm joints back to the default
        rest pose, IK targets back to the rest tip pose (+ offsets), grippers back to open, and
        tracking re-homed (same as R) so the arm doesn't immediately snap back toward wherever
        your hand currently is."""
        nonlocal target_pos_b_left, target_quat_b_left, target_pos_b_right, target_quat_b_right
        nonlocal left_closed, right_closed
        nonlocal left_gripper_smoothed, left_gripper_vel, right_gripper_smoothed, right_gripper_vel
        scene["box"].write_root_state_to_sim(scene["box"].data.default_root_state.clone())
        scene["container"].write_root_state_to_sim(scene["container"].data.default_root_state.clone())
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        target_pos_b_left = init_tip_pos_b_l.clone()
        target_quat_b_left = init_tip_quat_b_l.clone()
        target_pos_b_right = init_tip_pos_b_r.clone()
        target_quat_b_right = init_tip_quat_b_r.clone()
        left_arm.smoothed_pos = None
        right_arm.smoothed_pos = None
        left_closed = False
        right_closed = False
        # Snap (not smoothly drift) to open on reset -- matches the arm/box/container all being
        # hard-reset too, not eased into their reset state.
        left_gripper_smoothed = left_g_open.clone()
        left_gripper_vel = torch.zeros_like(left_g_open)
        right_gripper_smoothed = right_g_open.clone()
        right_gripper_vel = torch.zeros_like(right_g_open)
        _recalibrate()
        print("[Quest] Scene reset -- box/container respawned, arm back to rest pose.", flush=True)

    def _on_keyboard_event(event, *args, **kwargs) -> None:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        if event.input.name == "R":
            _recalibrate()
        elif event.input.name == "T":
            _reset_scene()
        elif event.input.name == "S" and recorder is not None:
            recorder.save_episode()
            print("[RECORD] [S] Episode saved -- recording continues for the next one.", flush=True)
        elif event.input.name == "D" and recorder is not None:
            recorder.cancel_recording()
            print("[RECORD] [D] Episode discarded -- recording continues for the retry.", flush=True)

    _keyboard_iface = carb.input.acquire_input_interface()
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    _keyboard_sub = _keyboard_iface.subscribe_to_keyboard_events(_keyboard, _on_keyboard_event)

    print("[Quest] Ready. Waiting for /quest_teleop messages.", flush=True)
    print("[Quest] Both arms: Differential IK (DLS), fingertip-tip target.", flush=True)
    print("[Quest] Connect the Quest browser to start streaming hand data.", flush=True)
    print("[Quest] Press R (Isaac Sim window focused) to recalibrate to your current pose.", flush=True)
    if recorder is not None:
        print("[Quest] Press S to save the current episode, D to discard it -- recording "
              "continues either way for the next demo.", flush=True)
    print("[Quest] Press T (Isaac Sim window focused) to fully reset the scene (box/container/arm).", flush=True)

    diag_frame = 0
    pov_capture_frame = 0
    _vr_connected = False
    _fps_window_start_t = time.monotonic()
    _fps_window_frames = 0
    _fps_window_capture_ms = 0.0
    # Real-time-factor + pacer accounting. See main() for the cost model these feed.
    _pace_deadline = time.monotonic()
    _pace_max_debt_s = _PACER_MAX_DEBT_CYCLES * _POV_CAPTURE_EVERY_N_STEPS * sim_dt
    _rtf_window_sleep_ms = 0.0
    _rtf_window_start_t = time.monotonic()
    _rtf_window_sim_s = 0.0
    _rtf_window_step_ms = 0.0
    _rtf_window_steps = 0
    _rtf_window_step_min_ms = float("inf")
    _rtf_window_step_max_ms = 0.0
    # Ctrl-C / SIGTERM REQUEST a shutdown rather than exiting in the handler, so the single
    # cleanup path below the loop always runs. That matters because recorder.finalize() blocks
    # until the async save queue drains -- episodes save on a background thread, so exiting from
    # inside the handler silently loses one that looked saved. A second signal hard-exits, so a
    # hung finalize can never trap the operator.
    _shutdown_requested = False

    def _request_shutdown(signum, _frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            print("[Quest] Second signal -- exiting immediately (in-flight saves may be lost).",
                  flush=True)
            os._exit(1)
        _shutdown_requested = True
        _draining = " draining the recorder save queue," if recorder is not None else ""
        print(f"[Quest] Caught signal {signum} -- shutting down cleanly:{_draining} "
              "closing Isaac Sim. Do not force-kill; this can take a few seconds.", flush=True)

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    while simulation_app.is_running() and not _shutdown_requested:
        msg = receiver.poll()

        if msg is not None:
            if not _vr_connected:
                _vr_connected = True
                print("[Quest] VR connected (first /quest_teleop message received).", flush=True)
                # Auto-arm recording on headset connect rather than waiting for a keypress.
                # Buffering only -- S still has to be pressed to commit an episode to disk.
                if recorder is not None and recorder._flags is not None and not recorder._flags.start:
                    recorder._flags.start = True
                    print("[RECORD] VR connected -- recording started automatically.", flush=True)
            left_xyz_q = _wrist_xyz(msg.left_wrist).to(device)
            right_xyz_q = _wrist_xyz(msg.right_wrist).to(device)
            left_quat = _wrist_quat_wxyz(msg.left_wrist).to(device)
            right_quat = _wrist_quat_wxyz(msg.right_wrist).to(device)
            head_xyz_q = _wrist_xyz(msg.head_pose).to(device)
            head_quat = _wrist_quat_wxyz(msg.head_pose).to(device)
            left_joints = list(msg.left_hand_joints)
            right_joints = list(msg.right_hand_joints)

            left_tracked = _is_tracked(left_xyz_q, left_quat)
            right_tracked = _is_tracked(right_xyz_q, right_quat)
            head_tracked = _is_tracked(head_xyz_q, head_quat)

            # Filter tracking noise before homing/displacement use it. Only while tracked, so
            # the untracked sentinel (0,0,0 / identity) never enters either filter's state.
            if left_tracked:
                left_xyz_q = left_arm.pos_filter.filter(left_xyz_q, sim_dt)
                left_quat = left_arm.quat_filter.filter(left_quat, sim_dt)
            if right_tracked:
                right_xyz_q = right_arm.pos_filter.filter(right_xyz_q, sim_dt)
                right_quat = right_arm.quat_filter.filter(right_quat, sim_dt)

            root_quat_w = robot.data.root_state_w[:, 3:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            tip_pos_b_l, _ = left_arm.tip_pose_b(robot, root_pose_w)
            tip_pos_b_r, _ = right_arm.tip_pose_b(robot, root_pose_w)

            # ── Homing (fingertip-tip anchored) ─────────────────────────────────
            if left_arm.quest_home_xyz is None and left_tracked:
                left_arm.quest_home_xyz = left_xyz_q.clone()
                left_arm.quest_home_quat = left_quat.clone()
                # Anchor to the launch-time rest pose, not the current tip -- anchoring to the
                # current tip bakes in IK lag and makes recalibration a no-op.
                left_arm.home_tip_pos_b = init_tip_pos_b_l.clone()
                left_arm.home_tip_quat_b = init_tip_quat_b_l.clone()
                target_pos_b_left = left_arm.home_tip_pos_b.clone()
                target_quat_b_left = left_arm.home_tip_quat_b.clone()
                print(f"[Quest] Left wrist tracked — homed at {left_xyz_q.tolist()}", flush=True)

            if right_arm.quest_home_xyz is None and right_tracked:
                right_arm.quest_home_xyz = right_xyz_q.clone()
                right_arm.quest_home_quat = right_quat.clone()
                right_arm.home_tip_pos_b = init_tip_pos_b_r.clone()
                right_arm.home_tip_quat_b = init_tip_quat_b_r.clone()
                target_pos_b_right = right_arm.home_tip_pos_b.clone()
                target_quat_b_right = right_arm.home_tip_quat_b.clone()
                print(f"[Quest] Right wrist tracked — homed at {right_xyz_q.tolist()}", flush=True)

            # ── Target (fingertip-tip, base frame) ──────────────────────────────
            if left_arm.quest_home_xyz is not None and left_tracked:
                left_disp_raw = left_xyz_q - left_arm.quest_home_xyz
                left_gain = _ramped_gain(left_disp_raw.norm(), gain)
                left_right_amt = left_disp_raw[0] * axis_sign_left[0]
                left_up_amt = left_disp_raw[1] * axis_sign_left[1]
                left_fwd_amt = -left_disp_raw[2] * axis_sign_left[2]
                left_delta_w = (
                    left_right_amt * cam_right_world
                    + left_up_amt * cam_up_world
                    + left_fwd_amt * cam_fwd_world
                ) * left_gain
                left_delta_w = left_delta_w * min(1.0, _MAX_REACH_M / max(left_delta_w.norm().item(), 1e-6))
                target_pos_b_left_dbg = (
                    left_arm.home_tip_pos_b + quat_apply_inverse(root_quat_w, left_delta_w.unsqueeze(0))
                )
                if _SHOW_AXIS_DEBUG and left_disp_raw.norm().item() > 0.05:
                    print(f"[Quest][axisdbg] L quest_disp(x,y,z)={left_disp_raw.tolist()}  "
                          f"world_delta={left_delta_w.tolist()}  "
                          f"base_delta_from_home={(target_pos_b_left_dbg - left_arm.home_tip_pos_b)[0].tolist()}",
                          flush=True)
                target_pos_b_left = target_pos_b_left_dbg
                # Quest-local -> camera-local (_QUEST_TO_CAM_LOCAL) -> world (camera_tilt_quat),
                # matching the position mapping above. wrist_orient_offset is a residual
                # fine-tune knob, identity unless live testing shows one is needed.
                dq_left = quat_mul(left_quat.unsqueeze(0), quat_inv(left_arm.quest_home_quat.unsqueeze(0)))
                dq_left_camlocal = quat_mul(quat_mul(quest_to_cam_local_quat, dq_left), quat_inv(quest_to_cam_local_quat))
                dq_left_world = quat_mul(quat_mul(camera_tilt_quat, dq_left_camlocal), quat_inv(camera_tilt_quat))
                dq_left_world = quat_mul(quat_mul(left_arm.wrist_orient_offset, dq_left_world),
                                         quat_inv(left_arm.wrist_orient_offset))
                dq_left_base = quat_mul(quat_mul(quat_inv(root_quat_w), dq_left_world), root_quat_w)
                target_quat_b_left = quat_mul(dq_left_base, left_arm.home_tip_quat_b)

            if right_arm.quest_home_xyz is not None and right_tracked:
                right_disp_raw = right_xyz_q - right_arm.quest_home_xyz
                right_gain = _ramped_gain(right_disp_raw.norm(), gain)
                right_right_amt = right_disp_raw[0] * axis_sign_right[0]
                right_up_amt = right_disp_raw[1] * axis_sign_right[1]
                right_fwd_amt = -right_disp_raw[2] * axis_sign_right[2]
                right_delta_w = (
                    right_right_amt * cam_right_world
                    + right_up_amt * cam_up_world
                    + right_fwd_amt * cam_fwd_world
                ) * right_gain
                right_delta_w = right_delta_w * min(1.0, _MAX_REACH_M / max(right_delta_w.norm().item(), 1e-6))
                target_pos_b_right_dbg = (
                    right_arm.home_tip_pos_b + quat_apply_inverse(root_quat_w, right_delta_w.unsqueeze(0))
                )
                if _SHOW_AXIS_DEBUG and right_disp_raw.norm().item() > 0.05:
                    print(f"[Quest][axisdbg] R quest_disp(x,y,z)={right_disp_raw.tolist()}  "
                          f"world_delta={right_delta_w.tolist()}  "
                          f"base_delta_from_home={(target_pos_b_right_dbg - right_arm.home_tip_pos_b)[0].tolist()}",
                          flush=True)
                target_pos_b_right = target_pos_b_right_dbg
                # See the left-arm block above.
                dq_right = quat_mul(right_quat.unsqueeze(0), quat_inv(right_arm.quest_home_quat.unsqueeze(0)))
                dq_right_camlocal = quat_mul(quat_mul(quest_to_cam_local_quat, dq_right), quat_inv(quest_to_cam_local_quat))
                dq_right_world = quat_mul(quat_mul(camera_tilt_quat, dq_right_camlocal), quat_inv(camera_tilt_quat))
                dq_right_world = quat_mul(quat_mul(right_arm.wrist_orient_offset, dq_right_world),
                                          quat_inv(right_arm.wrist_orient_offset))
                dq_right_base = quat_mul(quat_mul(quat_inv(root_quat_w), dq_right_world), root_quat_w)
                target_quat_b_right = quat_mul(dq_right_base, right_arm.home_tip_quat_b)

            # Stereo viewpoint is fixed: head_viewpoint_pos_b/quat_b never change, and
            # head_home_xyz/quat/head_tracked feed only _is_tracked. Head-tracked and
            # rotate-only variants are in git history.

            diag_frame += 1
            if diag_frame % 100 == 0:
                if left_arm.quest_home_xyz is not None and left_tracked:
                    err_l = (target_pos_b_left - tip_pos_b_l).norm().item()
                    print(f"[Quest][diag] L(tip) gain={left_gain.item():.2f} target_err={err_l:.3f}m", flush=True)
                if right_arm.quest_home_xyz is not None and right_tracked:
                    err_r = (target_pos_b_right - tip_pos_b_r).norm().item()
                    print(f"[Quest][diag] R(tip) gain={right_gain.item():.2f} target_err={err_r:.3f}m", flush=True)
                # Data for tuning _DLS_MANIPULABILITY_EPSILON, should adaptive damping be
                # retried: lambda pinned at lambda_min means epsilon is too low, lambda near
                # lambda_max in comfortable poses means it is too high. Silent while
                # _adaptive_dls_lambda is uncalled, since last_manipulability stays None.
                if left_arm.last_manipulability is not None:
                    print(f"[Quest][ikdiag] L manipulability={left_arm.last_manipulability:.4f} "
                          f"lambda_used={left_arm.controller.cfg.ik_params['lambda_val']:.3f} "
                          f"(min={left_arm.lambda_min} max={left_arm.lambda_max})", flush=True)
                if right_arm.last_manipulability is not None:
                    print(f"[Quest][ikdiag] R manipulability={right_arm.last_manipulability:.4f} "
                          f"lambda_used={right_arm.controller.cfg.ik_params['lambda_val']:.3f} "
                          f"(min={right_arm.lambda_min} max={right_arm.lambda_max})", flush=True)

            # Markers are only positioned when NOT recording. Two attempts at excluding them
            # from wrist_cam's render via cameraVisibility both failed silently (this RTX version
            # ignores the collection), so keeping them out of the scene entirely is the only
            # reliable guarantee against contaminating training data. The browser-side marker's
            # world positions are computed unconditionally below, independent of this gate.
            if not args_cli.record:
                _rp = robot.data.root_state_w[:, :3]
                if left_arm.quest_home_xyz is not None:
                    left_target_vis.visualize(translations=_rp + quat_apply(root_quat_w, target_pos_b_left))
                if right_arm.quest_home_xyz is not None:
                    right_target_vis.visualize(translations=_rp + quat_apply(root_quat_w, target_pos_b_right))

            if left_tracked:
                ld = _pinch_dist(left_joints)
                if ld < _PINCH_CLOSE_M:
                    left_closed = True
                elif ld > _PINCH_OPEN_M:
                    left_closed = False

            if right_tracked:
                rd = _pinch_dist(right_joints)
                if rd < _PINCH_CLOSE_M:
                    right_closed = True
                elif rd > _PINCH_OPEN_M:
                    right_closed = False

        # Decided here so sim.step(render=), the recorder and the POV capture share one answer.
        # +1 because pov_capture_frame is incremented later in the loop.
        _render_this_step = (pov_capture_frame + 1) % _POV_CAPTURE_EVERY_N_STEPS == 0

        # Sampled here, before the solve and before sim.step -- this is the state the action
        # responds to, giving the (s_t, a_t -> s_t+1) pairing training expects. Reading it in
        # the recording block instead samples post-step, which used to make observation.state a
        # near-copy of action (r>0.999 on all 6 arm joints) and teaches a BC policy to echo
        # proprioception rather than move.
        if recorder is not None and _render_this_step:
            _record_state = torch.cat([
                robot.data.joint_pos[:, left_arm.arm_ids],
                robot.data.joint_pos[:, left_gripper_ids[:1]],
            ], dim=1)[0].clone()

        # ── Differential IK (DLS) solve, both arms, every frame ─────────────────
        tip_pos_b_l_now, tip_quat_b_l_now = left_arm.solve_and_apply(robot, device, target_pos_b_left, target_quat_b_left, sim_dt)
        tip_pos_b_r_now, tip_quat_b_r_now = right_arm.solve_and_apply(robot, device, target_pos_b_right, target_quat_b_right, sim_dt)

        # ── Real-hardware bridge (left arm only) ─────────────────────────────────
        # The publish period only starts accumulating AFTER the startup delay, so the hold
        # cannot build a backlog and burst-publish the moment it ends.
        if real_left_arm_pub is not None:
            real_left_arm_elapsed_s += sim_dt
            if real_left_arm_elapsed_s >= _REAL_ARM_PUBLISH_START_DELAY_S:
                real_left_arm_since_publish_s += sim_dt
            if (real_left_arm_elapsed_s >= _REAL_ARM_PUBLISH_START_DELAY_S
                    and real_left_arm_since_publish_s >= _REAL_ARM_PUBLISH_PERIOD_S):
                real_left_arm_since_publish_s -= _REAL_ARM_PUBLISH_PERIOD_S
                if not real_left_arm_started:
                    real_left_arm_started = True
                    print("[Quest][REAL HARDWARE] Publishing left arm to /arm/joint_targets now.", flush=True)
                _publish_real_left_arm_pose(real_left_arm_pub, receiver, left_arm.last_joint_pos_des)

        if pov_capture_frame % 100 == 0:
            err_l = (target_pos_b_left - tip_pos_b_l_now).norm().item()
            err_r = (target_pos_b_right - tip_pos_b_r_now).norm().item()
            print(f"[Quest][diag] home-offset convergence: L target_err={err_l:.3f}m tip={tip_pos_b_l_now[0].tolist()} "
                  f"R target_err={err_r:.3f}m tip={tip_pos_b_r_now[0].tolist()}", flush=True)
            # Wrist-orientation calibration: commanded vs achieved delta. With
            # wrist_orient_offset tuned correctly the AXES should match closely; the angle may
            # lag during fast motion. For offline calibration, not normal use.
            if left_arm.home_tip_quat_b is not None:
                cmd_dq_l = quat_mul(target_quat_b_left, quat_inv(left_arm.home_tip_quat_b))
                ach_dq_l = quat_mul(tip_quat_b_l_now, quat_inv(left_arm.home_tip_quat_b))
                print(f"[Quest][wristdiag] L commanded_delta(wxyz)={cmd_dq_l[0].tolist()} "
                      f"achieved_delta(wxyz)={ach_dq_l[0].tolist()}", flush=True)
            if right_arm.home_tip_quat_b is not None:
                cmd_dq_r = quat_mul(target_quat_b_right, quat_inv(right_arm.home_tip_quat_b))
                ach_dq_r = quat_mul(tip_quat_b_r_now, quat_inv(right_arm.home_tip_quat_b))
                print(f"[Quest][wristdiag] R commanded_delta(wxyz)={cmd_dq_r[0].tolist()} "
                      f"achieved_delta(wxyz)={ach_dq_r[0].tolist()}", flush=True)

        # ── Gripper targets ────────────────────────────────────────────────────
        # smooth_time=0.05s keeps grasp timing snappy; max_speed is large relative to the
        # ~0.05-unit gripper range, so smooth_time shapes the motion, not the speed cap.
        left_gripper_target = left_g_closed if left_closed else left_g_open
        right_gripper_target = right_g_closed if right_closed else right_g_open
        left_gripper_smoothed, left_gripper_vel = _smooth_damp(
            left_gripper_smoothed, left_gripper_vel, left_gripper_target, 0.05, 1.0, sim_dt,
        )
        right_gripper_smoothed, right_gripper_vel = _smooth_damp(
            right_gripper_smoothed, right_gripper_vel, right_gripper_target, 0.05, 1.0, sim_dt,
        )
        robot.set_joint_position_target(left_gripper_smoothed, joint_ids=left_gripper_ids)
        robot.set_joint_position_target(right_gripper_smoothed, joint_ids=right_gripper_ids)
        robot.set_joint_velocity_target(torch.zeros(1, len(left_gripper_ids), device=device), joint_ids=left_gripper_ids)
        robot.set_joint_velocity_target(torch.zeros(1, len(right_gripper_ids), device=device), joint_ids=right_gripper_ids)

        scene.write_data_to_sim()
        # Gate rendering explicitly: render_interval only gates rendering inside the Isaac Lab
        # ENV classes, which this standalone script has none of, so bare sim.step() (render=True
        # by default) renders every physics step. _render_this_step is decided at the top of the
        # loop and shared with the recorder + capture.
        _step_t0 = time.monotonic()
        sim.step(render=_render_this_step)
        _step_ms = (time.monotonic() - _step_t0) * 1000.0
        _rtf_window_step_ms += _step_ms
        _rtf_window_step_min_ms = min(_rtf_window_step_min_ms, _step_ms)
        _rtf_window_step_max_ms = max(_rtf_window_step_max_ms, _step_ms)
        _rtf_window_sim_s += sim_dt
        _rtf_window_steps += 1
        scene.update(sim_dt)

        _rtf_elapsed = time.monotonic() - _rtf_window_start_t
        if _rtf_elapsed >= 2.0:
            _rtf = _rtf_window_sim_s / max(_rtf_elapsed, 1e-6)
            _pace_state = "ON" if _PACE_TO_REALTIME else "OFF"
            _pace_pct = _rtf_window_sleep_ms / max(_rtf_elapsed * 1000.0, 1e-9) * 100.0
            print(f"[Quest][rtfdiag] real-time factor={_rtf:.2f} (1.0=real-time, <1.0=running "
                  f"behind, pacer {_pace_state}, slept {_pace_pct:.0f}% of wall time) "
                  f"| avg sim.step() wall time: {_rtf_window_step_ms / max(_rtf_window_steps, 1):.2f}ms "
                  f"(min={_rtf_window_step_min_ms:.2f}ms max={_rtf_window_step_max_ms:.2f}ms) -- "
                  f"bimodal fast/slow (min << max) is EXPECTED now that rendering is gated -- "
                  f"min is a physics-only step, max includes the render. min~=max would mean the "
                  f"gate is not taking effect and Kit is still rendering every step.",
                  flush=True)
            _rtf_window_start_t = time.monotonic()
            _rtf_window_step_min_ms = float("inf")
            _rtf_window_step_max_ms = 0.0
            _rtf_window_sim_s = 0.0
            _rtf_window_step_ms = 0.0
            _rtf_window_steps = 0
            _rtf_window_sleep_ms = 0.0

        # ── Recording (L-suffixed/link6l arm only -- the one ego_cam/
        # wrist_cam are mounted for) ─────────────────────────────────────────
        if recorder is not None:
            if recorder.is_complete:
                print("[RECORD] Session complete.", flush=True)
                break
            # Render steps only: rate_limit=False, so this is the sole frame-rate gate, and it
            # is the same gate that decided whether the cameras produced new pixels.
            if _render_this_step:
                # left_gripper_smoothed, not the binary open/closed decision: the decision
                # snaps 0 -> -0.05 in one frame while _smooth_damp eases the joint over ~50ms,
                # and it is the eased value that was actually commanded. Logging the snap
                # trained the policy to slam a gripper the demos never slammed.
                action = torch.cat([left_arm.last_joint_pos_des, left_gripper_smoothed[:, :1]], dim=1)[0]
                # images as a callable, not a dict: two 640x480 GPU->CPU copies that must only
                # run on pushed frames. Every step tanked RTF -- looked like the arm barely moving.
                recorder.tick(action, _record_state, lambda: _capture_record_images(scene))

        # Mounts are static while _HEAD_TRACKING_LIVE is False -- positioned once before the
        # loop by _sync_camera_mounts (see there for why this is not a cheap no-op).
        if _HEAD_TRACKING_LIVE:
            _sync_camera_mounts()

        pov_capture_frame += 1
        if left_eye_camera is not None and right_eye_camera is not None and _render_this_step:
            _capture_t0 = time.monotonic()
            # dt here only feeds the sensor's own update-period bookkeeping -- these are not
            # scene entities, so scene.update() never ticks them.
            #
            # Guarded because this is a best-effort preview: unguarded frame writes killed the
            # whole sim twice, and both times it presented as "the grasp broke", because the arm
            # simply stopped being driven.
            try:
                left_eye_camera.update(dt=sim_dt * _POV_CAPTURE_EVERY_N_STEPS)
                right_eye_camera.update(dt=sim_dt * _POV_CAPTURE_EVERY_N_STEPS)
                _write_pov_jpeg(left_eye_camera, _POV_FRAME_PATH_LEFT)
                _write_pov_jpeg(right_eye_camera, _POV_FRAME_PATH_RIGHT)
                _write_wrist_cam_hud_frames(scene)
            except Exception as _pov_exc:  # noqa: BLE001 -- see comment above
                print(f"[Quest] WARNING: POV capture failed this cycle: {_pov_exc}", flush=True)

            # Project each arm's current IK target through both eye cameras so the browser draws
            # the marker on the pixel it actually occupies. See _project_world_point_to_uv.
            _rp_now = robot.data.root_state_w[:, :3]
            _target_world_left = (_rp_now + quat_apply(root_quat_w, target_pos_b_left))[0].tolist()
            _target_world_right = (_rp_now + quat_apply(root_quat_w, target_pos_b_right))[0].tolist()
            _lu_l, _lv_l, _lvis_l = _project_world_point_to_uv(_left_eye_cam_prim, _target_world_left)
            _ru_l, _rv_l, _rvis_l = _project_world_point_to_uv(_right_eye_cam_prim, _target_world_left)
            _lu_r, _lv_r, _lvis_r = _project_world_point_to_uv(_left_eye_cam_prim, _target_world_right)
            _ru_r, _rv_r, _rvis_r = _project_world_point_to_uv(_right_eye_cam_prim, _target_world_right)
            # PID-unique temp name + guarded, for the same reasons as _save_frame_atomic.
            _marker_uv_tmp = _MARKER_UV_PATH.with_name(f"{_MARKER_UV_PATH.name}.{os.getpid()}.tmp")
            try:
                _marker_uv_tmp.write_text(json.dumps({
                    "left_eye": {
                        "left_arm": {"u": _lu_l, "v": _lv_l, "visible": _lvis_l and left_arm.quest_home_xyz is not None},
                        "right_arm": {"u": _lu_r, "v": _lv_r, "visible": _lvis_r and right_arm.quest_home_xyz is not None},
                    },
                    "right_eye": {
                        "left_arm": {"u": _ru_l, "v": _rv_l, "visible": _rvis_l and left_arm.quest_home_xyz is not None},
                        "right_arm": {"u": _ru_r, "v": _rv_r, "visible": _rvis_r and right_arm.quest_home_xyz is not None},
                    },
                }))
                _marker_uv_tmp.replace(_MARKER_UV_PATH)
            except OSError as _uv_exc:
                print(f"[Quest] WARNING: could not write marker_uv.json: {_uv_exc}", flush=True)
            _capture_dt_ms = (time.monotonic() - _capture_t0) * 1000.0
            # Measured (not theoretical) capture rate, split into the capture itself vs the
            # rest of the cycle, so it is clear which one is the bottleneck. Printed every ~2s.
            _fps_window_frames += 1
            _fps_window_capture_ms += _capture_dt_ms
            _fps_elapsed = time.monotonic() - _fps_window_start_t
            if _fps_elapsed >= 2.0:
                _avg_capture_ms = _fps_window_capture_ms / max(_fps_window_frames, 1)
                _avg_cycle_ms = (_fps_elapsed / max(_fps_window_frames, 1)) * 1000.0
                # Derived, never hardcoded -- these are read while sweeping
                # _POV_CAPTURE_EVERY_N_STEPS, so a literal would misreport the run being tuned.
                _sim_time_hz = 1.0 / max(_POV_CAPTURE_EVERY_N_STEPS * sim_dt, 1e-9)
                print(f"[Quest][fps] measured POV capture rate: {_fps_window_frames / _fps_elapsed:.1f} fps "
                      f"(sim-time target ~{_sim_time_hz:.0f}Hz at dt={sim_dt * 1000:.0f}ms, render every "
                      f"{_POV_CAPTURE_EVERY_N_STEPS} steps) | avg POV capture (update+jpeg) time: "
                      f"{_avg_capture_ms:.1f}ms | avg full cycle ({_POV_CAPTURE_EVERY_N_STEPS} steps) time: "
                      f"{_avg_cycle_ms:.1f}ms", flush=True)
                _fps_window_start_t = time.monotonic()
                _fps_window_frames = 0
                _fps_window_capture_ms = 0.0

        # ── wall-clock pacer: hold RTF <= 1.0 ────────────────────────────────
        # Last in the loop so it absorbs the whole iteration. The deadline accumulates sim_dt and
        # is deliberately NOT reset on overrun -- that is what lets cheap physics-only steps
        # repay the render step. time.sleep releases the GIL, so the rclpy spin thread and the
        # recorder's writer thread keep running through it.
        if _PACE_TO_REALTIME:
            _pace_deadline += sim_dt
            _pace_slack = _pace_deadline - time.monotonic()
            if _pace_slack > 0.0:
                time.sleep(_pace_slack)
                _rtf_window_sleep_ms += _pace_slack * 1000.0
            elif _pace_slack < -_pace_max_debt_s:
                _pace_deadline = time.monotonic() - _pace_max_debt_s  # write off, never sprint

    if recorder is not None:
        # finalize() drains episodes already committed with S; it never saves the live buffer.
        # Exiting is not an implicit save, so the message says exactly what was written.
        recorder.finalize()
        _pending_frames = getattr(recorder, "_current_frame", 0)  # noqa: SLF001
        print(f"[RECORD] {recorder.num_recorded_episodes} episode(s) written to "
              f"{recorder.dataset_root}", flush=True)
        if _pending_frames:
            print(f"[RECORD] {_pending_frames} un-committed frame(s) discarded -- only S "
                  "saves an episode, exiting never does.", flush=True)


def main() -> None:
    # Performance model (measured, live): render ~35.5ms once per cycle (the bottleneck),
    # physics step ~4.8ms + loop work ~8.2ms every step, POV capture ~5ms per cycle.
    # => cycle(n) ~= 35.5 + n*13 + 5. Raising _POV_CAPTURE_EVERY_N_STEPS (n) trades fps for RTF
    # favourably (physics-only steps are cheap, render is fixed): n=2 -> 15fps/RTF 0.60,
    # n=5 -> 9.5fps/RTF 0.95. RTF is felt arm latency; fps is only video smoothness.
    # _PACE_TO_REALTIME caps RTF at 1.0.
    #
    # dt was 0.01; halved for RTF (per-step cost doesn't scale with dt). Larger dt = coarser
    # contacts and the grasp is marginal (holds on GPU PhysX, slips on CPU), so if the box slips,
    # restore dt first. render_interval below does NOT gate rendering -- see _KIT_RENDERING_INTERVAL.
    sim_cfg = sim_utils.SimulationCfg(dt=_PHYSICS_DT, render_interval=_KIT_RENDERING_INTERVAL,
                                      device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.8])

    # Trim RTX cost -- reflections/AO are real per-frame GPU time and buy nothing here.
    # Best-effort: an unknown setting path across Kit versions must not break teleop.
    _settings = carb.settings.get_settings()
    for _rtx_setting, _value in (
        ("/rtx/reflections/enabled", False),
        ("/rtx/ambientOcclusion/enabled", False),
        ("/rtx/indirectDiffuse/enabled", False),
    ):
        try:
            _settings.set(_rtx_setting, _value)
        except Exception as exc:  # noqa: BLE001 -- best-effort, see docstring above
            print(f"[Quest] Could not set {_rtx_setting}: {exc}", flush=True)

    scene_cfg = ArmV2SceneCfg(num_envs=1, env_spacing=2.0)
    if not args_cli.record:
        # ego_cam only feeds recording. Dropping it removes a whole RenderProduct from every
        # render tick and costs the operator nothing. None is InteractiveScene's skip value.
        scene_cfg.ego_cam = None
        print("[Quest] ego_cam disabled (no --record): one fewer camera rendered per frame.", flush=True)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[Quest] Simulation ready (lightbox enclosure walls added).")
    exit_code = 0
    try:
        run_simulator(sim, scene)
    except Exception:
        # os._exit below terminates without unwinding, so Python never prints this itself and
        # a crash would look exactly like a clean shutdown. Print it, and exit non-zero.
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        # recorder.finalize() has already returned, so everything committed is durably on disk
        # and nothing here can lose data. The only remaining job is to not hang.
        #
        # Each step prints BEFORE it runs, so a stall names the call that blocked. Keep the
        # prints: a stalled Kit/ROS call holds the GIL, so no in-process watchdog can report it.
        print("[Quest] Teardown 1/2: rclpy.shutdown()", flush=True)
        try:
            # rclpy installs its own SIGINT handler at rclpy.init(), so on Ctrl-C the context is
            # usually already down ("[rclcpp]: signal_handler" in the log) and calling shutdown()
            # again is an error rather than a no-op.
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as exc:  # noqa: BLE001 -- teardown must never raise past this point
            print(f"[Quest]   skipped: {exc}", flush=True)

        # sim.stop() and simulation_app.close() are both skipped -- each blocks forever tearing
        # down sensors (the hand-initialised eye Cameras) against still-live RenderProducts.
        # os._exit skips interpreter teardown too, since Kit's atexit hooks are part of the hang.
        # Everything durable is already written. Re-add a call only with proof it returns.
        print(f"[Quest] Teardown 2/2: exiting (status {exit_code}).", flush=True)
        os._exit(exit_code)


if __name__ == "__main__":
    main()
