"""Garment-fold scene for teleop (`keyboard_teleop.py --scene garment_fold`).

NEW (WATonomous) -- not vendored. Single source of truth for the declarative
half of this task's scene (worksurface + cameras, same shape as
`GarmentPioneerEnvCfg`/`GarmentPioneerEnv._build_worksurface`), registered
with `humanoid_isaac_scenes` by `../../../isaac_scenes/humanoid_isaac_scenes/
garment_fold/scene.py`.

The one piece that can't be a declarative cfg field: the garment itself.
`GarmentObject` (assets/garment_object.py) is a `SingleClothPrim` built with a
live constructor call plus a separate `.initialize()`, not a spawn-able
`AssetBaseCfg` -- there's no cfg type `InteractiveScene` knows how to turn
into one automatically. `post_init()` below builds it the same way
`GarmentPioneerEnv._create_garment_object` does, just run once after
`InteractiveScene`/`sim.reset()` exist instead of inside a `DirectRLEnv`.
Which garment to load is controlled by the `GARMENT_NAME` env var (default
`Top_Long_Seen_1`) since `keyboard_teleop.py`'s argparse is shared by every
registered scene and shouldn't grow a garment-fold-specific flag.
"""
from __future__ import annotations

import os
from dataclasses import MISSING
from typing import Optional

import numpy as np
from omegaconf import OmegaConf

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from pioneer_humanoid.arm_params import CAMERAS as _PIONEER_CAMERAS
from pioneer_humanoid.arm_params import vertical_aperture as _pioneer_vertical_aperture

from humanoid_garment_fold.assets.scene import MARBLE_BEDROOM_USD_PATH, TABLE038_USD_PATH
from humanoid_garment_fold.tasks.garment_pioneer_cfg import GarmentPioneerEnvCfg

# Robot base pose that reaches the garment, from GarmentPioneerEnvCfg (see
# that file's comment for how these were derived) -- one source of truth.
_DEFAULTS = GarmentPioneerEnvCfg()
ROBOT_BASE_POS = _DEFAULTS.robot_base_pos
ROBOT_BASE_ROT = _DEFAULTS.robot_base_rot

_WRIST_RES = (480, 640)  # height, width

# Table centre Y. The robot base sits at y=-0.40 facing +Y; at table height its stand column
# spans y in [-0.415, -0.341] (base_link.STL, measured), so 0.18 puts the table's near edge
# (centre - 0.5, Table038 is 1.0m deep) at y=-0.32: 2cm clear of the column.
TABLE_TELEOP_Y = 0.18

# Metres to move the garment's spawn toward the robot (world -Y) for teleop -- see post_init.
# Measured settled (Top_Long_Seen_1): at -0.15 the shirt spanned y in [-0.24, 0.07]; -0.21 puts
# its near hem at ~-0.30, about 2cm inside the table's near edge (y=-0.32, see TABLE_TELEOP_Y).
GARMENT_TELEOP_Y_SHIFT = -0.21

# Cloth particle-vs-rigid friction for teleop (particle_garment_cfg.yaml default: 0.5) -- see post_init.
GARMENT_TELEOP_FRICTION = 0.8


def _wrist_cam_cfg(name: str) -> TiledCameraCfg:
    """TiledCameraCfg for wrist camera `name` under `{ENV_REGEX_NS}/Robot`
    (not GarmentPioneerEnvCfg's `/World/Robot` -- the registry namespaces the
    robot per-env). Deliberately not `pioneer_humanoid.bimanual_arm
    .make_camera_cfg`, which builds a plain `Camera`: that fails the
    `/isaaclab/cameras_enabled` check under this InteractiveScene+sim.reset()
    flow even with --enable_cameras set (verified). TiledCamera doesn't hit
    that check and is what GarmentPioneerEnvCfg's own cameras already use.
    """
    cam = _PIONEER_CAMERAS[name]
    h, w = _WRIST_RES
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + f"{cam.body}/{cam.prim}",
        offset=TiledCameraCfg.OffsetCfg(pos=cam.pos, rot=cam.rot, convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam.focal, horizontal_aperture=cam.h_aperture,
            vertical_aperture=_pioneer_vertical_aperture(name, h, w),
            clipping_range=cam.clip,
        ),
        width=w, height=h, update_period=1 / 30.0,
    )

# ``_build_worksurface``'s order, decided once here (same file-existence check
# that function makes at runtime): apartment USD if present, else ground +
# vendored fallback table.
_HAVE_APARTMENT = os.path.isfile(MARBLE_BEDROOM_USD_PATH)
_HAVE_TABLE038 = os.path.isfile(TABLE038_USD_PATH)

_SCENE_USD_DEFAULT = (
    AssetBaseCfg(prim_path="/World/Scene", spawn=UsdFileCfg(usd_path=MARBLE_BEDROOM_USD_PATH))
    if _HAVE_APARTMENT else None
)
_GROUND_DEFAULT = (
    None if _HAVE_APARTMENT else
    AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
)
_TABLE_DEFAULT = (
    AssetBaseCfg(
        prim_path="/World/Table",
        # No rotation: Table038.usd is already Z-up (upAxis=Z, 1.1 x 1.0 m, top at z=0.52). The
        # +90deg-about-X that GarmentPioneerEnv._build_worksurface applies ("Y-up" comment there)
        # tips it onto its side -- unnoticed because the side face happens to sit at ~0.5m too.
        # Raised 5cm (top at z~0.57) for teleop reach; the garment still spawns above it (z=0.63).
        # Shifted +Y so its near edge (y = TABLE_TELEOP_Y - 0.5) sits just in front of the robot
        # stand instead of the column passing through the top -- see TABLE_TELEOP_Y.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, TABLE_TELEOP_Y, 0.05)),
        spawn=UsdFileCfg(
            usd_path=TABLE038_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # Flat wood-ish color instead of Table038's own PNG-textured material: loading those
            # textures aborts Isaac Sim in the simulation_isaac image (carb "unlock() called by
            # non-owning thread", ~10-25s in). Verified headless: same table with its texture/
            # folder unavailable runs clean, with it crashes every run.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.28)),
        ),
    )
    if (not _HAVE_APARTMENT and _HAVE_TABLE038) else None
)


@configclass
class GarmentFoldSceneCfg(InteractiveSceneCfg):
    """Worksurface + cameras, declarative. The garment itself is built by
    ``post_init`` after the scene exists -- see module docstring."""

    robot: ArticulationCfg = MISSING

    scene_usd: Optional[AssetBaseCfg] = _SCENE_USD_DEFAULT
    ground: Optional[AssetBaseCfg] = _GROUND_DEFAULT
    table: Optional[AssetBaseCfg] = _TABLE_DEFAULT

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75)),
    )

    left_wrist: TiledCameraCfg = _wrist_cam_cfg("wrist_left")
    right_wrist: TiledCameraCfg = _wrist_cam_cfg("wrist_right")
    # world-fixed, not mounted on the robot -- absolute paths are fine since
    # teleop always runs num_envs=1.
    top_camera: TiledCameraCfg = _DEFAULTS.top_camera
    scene_camera: TiledCameraCfg = _DEFAULTS.scene_camera


def post_init(scene, sim) -> None:
    """Build the garment particle-cloth object. Run once, after
    ``InteractiveScene`` and ``sim.reset()`` both exist (needs physics
    playing for ``GarmentObject.initialize()``)."""
    from humanoid_garment_fold.assets.garment_object import GarmentObject
    from humanoid_garment_fold.tasks.challenge_garment_loader import ChallengeGarmentLoader
    from humanoid_garment_fold.tasks.garment_pioneer_env import fix_garment_asset_paths

    garment_name = os.environ.get("GARMENT_NAME", "Top_Long_Seen_1")
    d = _DEFAULTS

    loader = ChallengeGarmentLoader(d.garment_cfg_base_path)
    garment_config = loader.load_garment_config(garment_name, d.garment_version)
    fix_garment_asset_paths(loader, garment_config, garment_name, d.garment_cfg_base_path, d.garment_version)
    # Teleop only: pull the garment toward the arm (robot base sits at -Y, facing +Y). The
    # per-garment JSON spawns it at world y~0, the rest fingertip line -- a long reach for a
    # VR operator. Soft reset (T key) is shifted too so it respawns in the same spot.
    for key in ("initial_pos_range", "soft_reset_pos_range"):
        if garment_config.get(key) is not None:
            r = list(garment_config[key])
            r[1] += GARMENT_TELEOP_Y_SHIFT
            r[4] += GARMENT_TELEOP_Y_SHIFT
            garment_config[key] = r
    # Teleop resets (T key) respawn exactly like the first spawn. The per-garment soft-reset
    # ranges are for RL randomisation: up to +/-20-22 deg of tilt and a drop from z=0.73 instead
    # of 0.63, so the shirt landed edge-first and folded over itself -- it looked half-loaded.
    for key in ("pos", "rot"):
        if garment_config.get(f"initial_{key}_range") is not None:
            garment_config[f"soft_reset_{key}_range"] = list(garment_config[f"initial_{key}_range"])
    particle_config = OmegaConf.load(d.particle_cfg_path)
    # Teleop only: grippier cloth. Particle-vs-rigid contact (fingers, table) uses the PARTICLE
    # material's friction, not the gripper's rigid friction (1.5 in the Quest script), so this is
    # the knob for grasping fabric. particle_garment_cfg.yaml (RL) keeps 0.5.
    particle_config.objects.particle_material.friction = GARMENT_TELEOP_FRICTION
    rng = np.random.RandomState(42)

    garment = GarmentObject(
        prim_path=f"/World/Object/{garment_name}",
        particle_config=particle_config,
        garment_config=garment_config,
        rng=rng,
    )
    for _ in range(20):
        sim.step(render=True)
    garment.initialize()

    # GarmentObject.reset() restores particle POSITIONS but not velocities, so a shirt reset
    # mid-drag respawned still moving and flung/bunched up. Zero them after every reset.
    _soft_reset = garment.reset

    def _reset_at_rest() -> None:
        _soft_reset()
        # GPU only: GarmentObject creates the cloth view only when running on cuda (teleop does).
        view = getattr(garment, "_cloth_prim_view", None)  # GarmentObject exposes no velocity API
        if view is not None:
            view.set_velocities(view.get_velocities() * 0.0)

    garment.reset = _reset_at_rest
    scene.garment = garment
