"""Quest Touch controller teleoperation in Isaac Sim.

WebXR poses/buttons -> always-on wrist targets -> sequential article IK -> simulated
joint actuators. Stereo/wrist camera JPEGs and status return to the headset.
Supports pick-and-place, cloth folding, and optional demonstration recording.
Launch with run_quest_controller_teleop_always_on.sh inside the simulation_isaac container.
"""

from __future__ import annotations

import argparse
import faulthandler
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

# Preserve all Python thread stacks if a native extension aborts. This does not
# start a simulator/server and helps distinguish Python integration from SDK faults.
faulthandler.enable(all_threads=True)

from isaaclab.app import AppLauncher

# ── path setup (must be before AppLauncher so PYTHONPATH is correct) ─────────
_THIS_DIR = Path(__file__).resolve().parent
_AUTONOMY = _THIS_DIR.parents[2]  # src/teleop/quest_controller_teleop/isaac -> src
_SIM_DIR = _AUTONOMY / "simulation"
# pioneer_humanoid package (canonical arm config). Editable-installed in the image; this fallback
# keeps a bare bind-mounted checkout working.
sys.path.insert(0, str(_AUTONOMY / "pioneer_humanoid"))
_ROBOT_LEARNING_PKG = _AUTONOMY / "robot_learning"
from quest_scenes import add_scene_arguments, make_cloth_scene_cfg, reset_workspace, SceneHooks

sys.path.insert(0, str(_THIS_DIR.parent))  # the simulator-independent controller modules
import joint_limits  # noqa: E402
from scene_driver import add_controller_args  # noqa: E402


def _ensure_robot_learning_on_path() -> None:
    if str(_ROBOT_LEARNING_PKG) not in sys.path:
        sys.path.insert(0, str(_ROBOT_LEARNING_PKG))


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Quest Touch controller teleoperation in Isaac Sim")
add_scene_arguments(parser)
parser.add_argument("--single-task-thread", action="store_true",
                    help="Work around the native carb.assets/tasking mutex crash by launching Kit with one task worker")
add_controller_args(parser)
parser.add_argument("--record", action="store_true",
                    help="Record demonstrations (requires: pip install -e src/robot_learning[record]). Records the "
                         "L-suffixed (link6l) arm only -- that's the arm ego_cam/wrist_cam are mounted for.")
parser.add_argument("--schema", type=str,
                    default=None,
                    help="Path to dataset_schema.yaml (only used with --record)")
parser.add_argument("--sink", type=str, default="lerobot",
                    help="Output sinks when --record: lerobot, hdf5, or lerobot,hdf5")
parser.add_argument("--dataset_root", type=str, default=None,
                    help="Output directory (default: <repo>/<schema record.root>)")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--task_description", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.single_task_thread:
    # Same Mutex.h:158 workaround documented by PushBlockEnvCfg_PLAY. This is
    # startup-only: setting it after AppLauncher has created Kit is too late.
    task_setting = "--/plugins/carb.tasking.plugin/threadCount=1"
    args_cli.kit_args = f"{args_cli.kit_args or ''} {task_setting}".strip()
    print("[Quest] Native asset/tasking crash workaround: one Carbonite task worker.", flush=True)
if args_cli.schema is None:
    schema_name = "dataset_schema_wato_arm_v2_cloth_folding.yaml" if args_cli.scene == "cloth_folding" else "dataset_schema_wato_arm_v2_push_box.yaml"
    args_cli.schema = str(_ROBOT_LEARNING_PKG / "config" / schema_name)
if args_cli.task_description is None:
    args_cli.task_description = f"fold {args_cli.garment}" if args_cli.scene == "cloth_folding" else "pick up box and place it in container"
if args_cli.scene == "cloth_folding":
    if not str(args_cli.device).startswith("cuda"):
        parser.error("Cloth folding uses PhysX particle cloth and requires --device cuda")
    sys.path.insert(0, str(_SIM_DIR / "garment_fold_task"))
if not all(math.isfinite(v) and v > 0 for v in (args_cli.gain, args_cli.joint_speed,
                                               args_cli.wrist_speed, args_cli.rotation_gain, args_cli.angular_lead)):
    parser.error("Controller gains, speeds, and angular lead must be positive and finite")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
faulthandler.enable(all_threads=True)  # Reapply after Kit installs its native handlers.

# ── post-launch imports (require Omniverse runtime) ───────────────────────────
import torch  # noqa: E402

import carb  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    quat_apply,
    matrix_from_quat,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

# No aliasing: LEFT_* is the L-suffixed chain (physical left = left Quest wrist), RIGHT_* the
# unsuffixed one. This block used to swap them to undo a reversed upstream; don't bring that back.
from quest_cameras import (  # noqa: E402
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
    resolve_body_ids,
    resolve_joint_name,
)


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
# Headset orientation affects controller mapping, not camera positioning.
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
# Head tracking is not live -- head_viewpoint_pos_b/quat_b are set once at startup and never
# mutated (see run_simulator). With this False the camera mounts are positioned once before the
# sim loop instead of re-authored every physics step; flip to True the moment the head pose is
# actually driven from the Quest, and per-step tracking resumes with no other change.
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


def _camera_rgb_frame(camera):
    """HxWx3 uint8 RGB from a Camera sensor's output tensor, alpha dropped if present."""
    frame = camera.data.output["rgb"][0].detach().cpu().numpy()
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    return frame.astype("uint8")


# Render and publish fresh camera frames every N physics steps.
_POV_CAPTURE_EVERY_N_STEPS = 5
_PHYSICS_DT = 1/90 if args_cli.scene == "cloth_folding" else .02

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


# Draws a 2cm green sphere at quest_cameras.WRIST_CAM_POS so the camera's mount point is
# visible in the viewport while tuning it. Visual only -- no collision or physics. Turn it off
# before recording: it sits inside link6l and can appear in ego_cam/wrist_cam frames.
_SHOW_WRIST_CAM_MARKER = False


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
    # lens live in quest_cameras.py -- adjust there, not here. wrist_cam keeps its
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
    # Graspable box -- dynamic rigid body, using the imported block.usd mesh.
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
    # collide with the box).
    container: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Container",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CONTAINER_POS, rot=_CONTAINER_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=_CONTAINER_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


def _joint_ids(robot, names: list[str]) -> list[int]:
    name_to_id = {n: i for i, n in enumerate(robot.data.joint_names)}
    return [name_to_id[resolve_joint_name(robot, n)] for n in names]


def _make_entity_cfg(scene, joint_names: list[str], ee_body: str) -> SceneEntityCfg:
    cfg = SceneEntityCfg("robot", joint_names=joint_names, body_names=[ee_body], preserve_order=True)
    cfg.resolve(scene)
    return cfg


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


def _ee_pose_in_base(robot, body_id: int):
    root_pose_w = robot.data.root_state_w[:, 0:7]
    ee_pose_w = robot.data.body_state_w[:, body_id, 0:7]
    return subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
    )


class _ArmState:
    """Joint indices and fingertip geometry for controller commands and recording."""

    def __init__(self, scene, robot, device, arm_joint_names, ee_body, finger_tip_bodies, finger_distal_local,
                 ):
        entity = _make_entity_cfg(scene, arm_joint_names, ee_body)
        self.body_id = entity.body_ids[0]
        self.arm_ids = entity.joint_ids
        self.arm_joint_names = [robot.data.joint_names[i] for i in self.arm_ids]
        self.finger_body_ids = resolve_body_ids(robot, finger_tip_bodies)
        self.finger_tip_bodies = finger_tip_bodies
        self.finger_distal_local = finger_distal_local
        self.last_joint_pos_des = None

    def tip_pose_b(self, robot, root_pose_w):
        return compute_gripper_tip_pose_b(
            robot, root_pose_w, self.body_id, self.finger_body_ids,
            self.finger_tip_bodies, self.finger_distal_local,
        )


def _init_recorder(device: str):
    """Returns (recorder, schema_cfg), or (None, None) if --record wasn't passed."""
    if not args_cli.record:
        return None, None
    _ensure_robot_learning_on_path()
    try:
        from humanoid_robot_learning.record_utils import resolve_config_path, resolve_dataset_root
        from humanoid_robot_learning.schema import enabled_images, load_yaml
        from humanoid_robot_learning.sim_recorder import SimLeRobotRecorder
    except ImportError as exc:
        raise ImportError("Recording requires humanoid-robot-learning. Install with:\n  pip install -e src/robot_learning[sim]") from exc

    schema_path = resolve_config_path(args_cli.schema, anchor=_ROBOT_LEARNING_PKG)
    cfg = load_yaml(schema_path)
    dataset_root = resolve_dataset_root(cfg, args_cli.dataset_root, default="datasets/record_wato_arm_v2_push_box")
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
    # Key bindings differ from humanoid_robot_learning's defaults -- see run_simulator's _on_keyboard_event.
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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    device = sim.device
    sim_dt = sim.get_physics_dt()
    gain = args_cli.gain

    scene.update(sim_dt)
    apply_joint_limits(robot)
    if args_cli.scene == "pick_place":
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
    print(f"[Quest] Stereo fixed-view RealSense D455 pair attached: {left_eye_mount}, {right_eye_mount}", flush=True)
    if _SHOW_WRIST_CAM_MARKER:
        _spawn_wrist_cam_marker("/World/envs/env_0/Robot/link6l/wrist_cam_marker", WRIST_CAM_POS)
        print(f"[Quest] wrist_cam calibration marker ON (2cm green sphere at {WRIST_CAM_POS} "
              "on link6l) -- set _SHOW_WRIST_CAM_MARKER=False to hide.", flush=True)
    _widen_camera_fov(f"{left_eye_mount}/{_RSD455_CAMERA_SUBPATH}", _RSD455_WIDENED_FOCAL_LENGTH)
    _widen_camera_fov(f"{right_eye_mount}/{_RSD455_CAMERA_SUBPATH}", _RSD455_WIDENED_FOCAL_LENGTH)
    _left_eye_cam_prim = omni.usd.get_context().get_stage().GetPrimAtPath(
        f"{left_eye_mount}/{_RSD455_CAMERA_SUBPATH}")
    left_eye_camera = _open_pov_camera(f"{left_eye_mount}/{_RSD455_CAMERA_SUBPATH}", "Left Eye POV")
    right_eye_camera = _open_pov_camera(f"{right_eye_mount}/{_RSD455_CAMERA_SUBPATH}", "Right Eye POV")
    left_arm_names = [resolve_joint_name(robot, n) for n in LEFT_ARM_JOINTS]
    right_arm_names = [resolve_joint_name(robot, n) for n in RIGHT_ARM_JOINTS]

    left_arm = _ArmState(
        scene, robot, device, left_arm_names, LEFT_EE_BODY,
        LEFT_FINGER_TIP_BODIES, LEFT_FINGER_DISTAL_TIP_LOCAL,
    )
    right_arm = _ArmState(
        scene, robot, device, right_arm_names, RIGHT_EE_BODY,
        RIGHT_FINGER_TIP_BODIES, RIGHT_FINGER_DISTAL_TIP_LOCAL,
    )

    left_gripper_ids = _joint_ids(robot, LEFT_GRIPPER_JOINTS)
    right_gripper_ids = _joint_ids(robot, RIGHT_GRIPPER_JOINTS)

    left_g_open = torch.tensor([[LEFT_GRIPPER_OPEN["joint7l"], LEFT_GRIPPER_OPEN["joint8l"]]], device=device)
    left_g_closed = torch.tensor([[LEFT_GRIPPER_CLOSED["joint7l"], LEFT_GRIPPER_CLOSED["joint8l"]]], device=device)
    right_g_open = torch.tensor([[RIGHT_GRIPPER_OPEN["joint7"], RIGHT_GRIPPER_OPEN["joint8"]]], device=device)
    right_g_closed = torch.tensor([[RIGHT_GRIPPER_CLOSED["joint7"], RIGHT_GRIPPER_CLOSED["joint8"]]], device=device)
    # Smooth proportional gripper targets to avoid discontinuities in recorded demos.
    left_gripper_smoothed = left_g_open.clone()
    left_gripper_vel = torch.zeros_like(left_g_open)
    right_gripper_smoothed = right_g_open.clone()
    right_gripper_vel = torch.zeros_like(right_g_open)

    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    controller_path = _THIS_DIR.parent
    sys.path.insert(0, str(controller_path))
    from scene_driver import ControllerSceneDriver, add_controller_args
    controller_frame = None
    if args_cli.controller_frame == "camera":
        # WebXR and USD cameras both use +X right, +Y up, -Z forward.
        # Read the actual composed camera transform (including its payload
        # mount convention and tilt), then express it in the robot base.
        camera_rotation = UsdGeom.Xformable(_left_eye_cam_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()).ExtractRotation().GetQuat()
        camera_quat_w = torch.tensor([[float(camera_rotation.GetReal()),
                                      *[float(v) for v in camera_rotation.GetImaginary()]]],
                                    dtype=torch.float32, device=device)
        camera_quat_b = quat_mul(quat_inv(robot.data.root_state_w[:, 3:7]), camera_quat_w)
        controller_frame = matrix_from_quat(camera_quat_b)[0].detach().cpu().numpy()
    controller_driver = ControllerSceneDriver(
        {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS},
        gain, args_cli.joint_speed, args_cli.controller_calibration, args_cli.controller_port,
        wrist_velocity=args_cli.wrist_speed, rotation_gain=args_cli.rotation_gain,
        angular_lead=args_cli.angular_lead, position_lead=args_cli.position_lead, frame_calibration=controller_frame,
        position_damping=args_cli.position_damping, orientation_damping=args_cli.orientation_damping,
        limits_deg=joint_limits.from_args(args_cli), elbow_velocity=args_cli.elbow_speed, slip=args_cli.slip,
    )
    print(f"[Quest] Controller frame: {args_cli.controller_frame}; rotation gain={args_cli.rotation_gain}; "
          f"proximal/wrist speed={args_cli.joint_speed}/{args_cli.wrist_speed} rad/s; "
          f"angular lead={args_cli.angular_lead} rad", flush=True)
    print("[Quest] Always-on control with article-based sequential IK", flush=True)
    controller_driver.start()
    # Position the fixed stereo mounts once at startup.
    head_home_viewpoint_pos_b = torch.tensor([_HEAD_VIEWPOINT_HOME_POS], dtype=torch.float32, device=device)
    head_home_viewpoint_quat_b = torch.tensor([_HEAD_VIEWPOINT_HOME_QUAT], dtype=torch.float32, device=device)
    head_viewpoint_pos_b = head_home_viewpoint_pos_b.clone()
    head_viewpoint_quat_b = head_home_viewpoint_quat_b.clone()
    eye_local_right = _EYE_LOCAL_RIGHT.to(device)
    def _sync_camera_mounts() -> None:
        """Set the fixed stereo mounts with the configured inter-eye separation."""
        right_offset_b = quat_apply(head_viewpoint_quat_b, eye_local_right.unsqueeze(0)) * (_EYE_IPD_M / 2)
        left_eye_pos_b = (head_viewpoint_pos_b - right_offset_b)[0].tolist()
        right_eye_pos_b = (head_viewpoint_pos_b + right_offset_b)[0].tolist()
        head_orient_b = head_viewpoint_quat_b[0].tolist()
        _set_mount_pose(left_eye_mount, tuple(left_eye_pos_b), tuple(head_orient_b))
        _set_mount_pose(right_eye_mount, tuple(right_eye_pos_b), tuple(head_orient_b))

    _sync_camera_mounts()


    recorder, record_cfg = _init_recorder(device)
    if recorder is not None:
        # NOT recorder.start_keyboard() -- that attaches humanoid_robot_learning's shared pynput listener,
        # whose bindings other teleop scripts rely on. S/D are handled through the
        # carb.input listener instead, calling save_episode()/cancel_recording() directly
        # so recording continues straight into the next demo. EpisodeFlags is still needed
        # (tick() no-ops without it) but is constructed directly.
        from humanoid_robot_learning.episode_keys import EpisodeFlags
        recorder._flags = EpisodeFlags(start=False)

    def _reset_scene() -> None:
        """Restore workspace, arm state, actuator targets, grippers, and clutch references."""
        nonlocal left_gripper_smoothed, left_gripper_vel, right_gripper_smoothed, right_gripper_vel
        reset_workspace(scene, args_cli.scene)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        # Snap (not smoothly drift) to open on reset -- matches the arm/box/container all being
        # hard-reset too, not eased into their reset state.
        left_gripper_smoothed = left_g_open.clone()
        left_gripper_vel = torch.zeros_like(left_g_open)
        right_gripper_smoothed = right_g_open.clone()
        right_gripper_vel = torch.zeros_like(right_g_open)
        # Reset the actuator commands as well as measured joint state. Otherwise a
        # held controller command from the previous episode pulls the arm back.
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.default_joint_vel))
        robot.reset()
        controller_driver.reset({
            "left": robot.data.default_joint_pos[0, left_arm.arm_ids].detach().cpu().numpy(),
            "right": robot.data.default_joint_pos[0, right_arm.arm_ids].detach().cpu().numpy(),
        })
        print(f"[Quest] Scene reset: {args_cli.scene}; arms back to rest pose.", flush=True)

    def _on_keyboard_event(event, *args, **kwargs) -> None:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        if event.input.name == "S" and recorder is not None:
            recorder.save_episode()
            print("[RECORD] [S] Episode saved -- recording continues for the next one.", flush=True)
        elif event.input.name == "D" and recorder is not None:
            recorder.cancel_recording()
            print("[RECORD] [D] Episode discarded -- recording continues for the retry.", flush=True)

    _keyboard_iface = carb.input.acquire_input_interface()
    _app_window = omni.appwindow.get_default_app_window()
    if _app_window is not None:
        _keyboard = _app_window.get_keyboard()
        _keyboard_sub = _keyboard_iface.subscribe_to_keyboard_events(_keyboard, _on_keyboard_event)

    print(f"[Quest] Controller mode: http://localhost:{args_cli.controller_port}/; left Y resets scene.", flush=True)
    print("[Quest] Both arms: always-on article IK; trigger to grasp.", flush=True)
    if recorder is not None:
        print("[Quest] Press S to save the current episode, D to discard it -- recording "
              "continues either way for the next demo.", flush=True)

    pov_capture_frame = 0
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
        controller_joint_positions = {
            "left": robot.data.joint_pos[0, left_arm.arm_ids].detach().cpu().numpy().copy(),
            "right": robot.data.joint_pos[0, right_arm.arm_ids].detach().cpu().numpy().copy(),
        }
        if controller_driver.update(controller_joint_positions):
            _reset_scene()
            controller_joint_positions = {
                "left": robot.data.default_joint_pos[0, left_arm.arm_ids].detach().cpu().numpy().copy(),
                "right": robot.data.default_joint_pos[0, right_arm.arm_ids].detach().cpu().numpy().copy(),
            }
        if recorder is not None and any(t.active for t in controller_driver.targets.values()):
            if recorder._flags is not None and not recorder._flags.start:
                recorder._flags.start = True

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

        # ── Position-priority controller IK, both arms, every frame ─────────────────
        commands = controller_driver.solve(controller_joint_positions, sim_dt)
        for side, arm in (("left", left_arm), ("right", right_arm)):
            arm.last_joint_pos_des = torch.as_tensor(commands[side], dtype=torch.float32, device=device).unsqueeze(0)
            robot.set_joint_position_target(arm.last_joint_pos_des, joint_ids=arm.arm_ids)
            robot.set_joint_velocity_target(torch.zeros_like(arm.last_joint_pos_des), joint_ids=arm.arm_ids)
        # ── Gripper targets ────────────────────────────────────────────────────
        # smooth_time=0.05s keeps grasp timing snappy; max_speed is large relative to the
        # ~0.05-unit gripper range, so smooth_time shapes the motion, not the speed cap.
        left_gripper_target = left_g_open + controller_driver.grippers["left"] * (left_g_closed-left_g_open)
        right_gripper_target = right_g_open + controller_driver.grippers["right"] * (right_g_closed-right_g_open)
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

        # Stereo mounts are fixed -- positioned once before the
        # loop by _sync_camera_mounts (see there for why this is not a cheap no-op).

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
                controller_driver.publish_cameras(
                    pov_left=_camera_rgb_frame(left_eye_camera), pov_right=_camera_rgb_frame(right_eye_camera),
                    wrist_left=_camera_rgb_frame(scene["wrist_cam"]),
                    wrist_right=_camera_rgb_frame(scene["wrist_cam_right"]),
                )
            except Exception as _pov_exc:  # noqa: BLE001 -- see comment above
                print(f"[Quest] WARNING: POV capture failed this cycle: {_pov_exc}", flush=True)

            _capture_dt_ms = (time.monotonic() - _capture_t0) * 1000.0
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
        # repay the render step. time.sleep releases the GIL, so the
        # recorder's writer thread keeps running through it.
        if _PACE_TO_REALTIME:
            _pace_deadline += sim_dt
            _pace_slack = _pace_deadline - time.monotonic()
            if _pace_slack > 0.0:
                time.sleep(_pace_slack)
                _rtf_window_sleep_ms += _pace_slack * 1000.0
            elif _pace_slack < -_pace_max_debt_s:
                _pace_deadline = time.monotonic() - _pace_max_debt_s  # write off, never sprint

    controller_driver.close()
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
    if args_cli.scene == "cloth_folding":
        sim_cfg.use_fabric = False  # Keep particle-cloth geometry synchronized to rendered USD.
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

    if args_cli.scene == "cloth_folding":
        try:
            scene_cfg, scene_hooks = make_cloth_scene_cfg(
                BIMANUAL_ARM_CFG, record=args_cli.record, garment_name=args_cli.garment,
            )
        except ImportError as exc:
            raise RuntimeError("Cloth scene needs the garment_fold_task checkout and omegaconf in Isaac's Python") from exc
        sim.set_camera_view([1.8, -2., 1.3], [0., -.1, .55])
    else:
        scene_cfg = ArmV2SceneCfg(num_envs=1, env_spacing=2.0)
        scene_hooks = SceneHooks()
    if not args_cli.record:
        # ego_cam only feeds recording. Dropping it removes a whole RenderProduct from every
        # render tick and costs the operator nothing. None is InteractiveScene's skip value.
        scene_cfg.ego_cam = None
        print("[Quest] ego_cam disabled (no --record): one fewer camera rendered per frame.", flush=True)
    scene = InteractiveScene(scene_cfg)
    scene.quest_scene_hooks = scene_hooks
    sim.reset()
    if scene_hooks.initialize is not None:
        scene_hooks.initialize(scene, sim)
    print(f"[Quest] Simulation ready: {args_cli.scene}.")
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
        # prints: a stalled Kit call holds the GIL, so no in-process watchdog can report it.
        # sim.stop() and simulation_app.close() are both skipped -- each blocks forever tearing
        # down sensors (the manually initialized eye Cameras) against still-live RenderProducts.
        # os._exit skips interpreter teardown too, since Kit's atexit hooks are part of the hang.
        # Everything durable is already written. Re-add a call only with proof it returns.
        print(f"[Quest] Teardown 2/2: exiting (status {exit_code}).", flush=True)
        os._exit(exit_code)


if __name__ == "__main__":
    main()
