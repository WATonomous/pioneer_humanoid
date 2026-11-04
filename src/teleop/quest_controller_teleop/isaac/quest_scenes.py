"""Scene selection and reset hooks shared by the Quest Isaac entry points.

Isaac imports are lazy so flag/reset tests do not launch or import the simulator.
"""
from dataclasses import dataclass
import os
from pathlib import Path


def add_scene_arguments(parser):
    parser.add_argument("--scene", choices=("pick_place", "cloth_folding"), default="pick_place",
                        help="Workspace: original pick-and-place (default), or Isaac particle-cloth folding")
    parser.add_argument("--cloth-folding", dest="scene", action="store_const", const="cloth_folding",
                        help="Shortcut for --scene cloth_folding")
    parser.add_argument("--garment", default=os.environ.get("GARMENT_NAME", "Top_Long_Seen_1"),
                        help="Cloth asset name, e.g. Top_Long_Seen_1 or Top_Short_Seen_0")


@dataclass
class SceneHooks:
    initialize: object = None
    reset: object = None


def asset_rotation_to_z_up(path):
    """Convert an asset's declared up axis to the Isaac world's Z-up frame.

    USD references preserve authored coordinates; the asset's upAxis metadata
    does not automatically rotate it when composed into another stage.
    """
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError(f"Could not read workspace asset: {path}")
    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis == UsdGeom.Tokens.z:
        return (1., 0., 0., 0.)
    if up_axis == UsdGeom.Tokens.y:
        from math import sqrt
        return (sqrt(.5), sqrt(.5), 0., 0.)
    raise ValueError(f"Unsupported workspace up axis {up_axis!r}: {path}")


def reset_workspace(scene, scene_name):
    """Reset workspace objects only; the caller separately resets the robot."""
    if scene_name == "pick_place":
        for name in ("box", "container"):
            entity = scene[name]
            entity.write_root_state_to_sim(entity.data.default_root_state.clone())
            entity.reset()
    elif scene_name == "cloth_folding":
        scene.quest_scene_hooks.reset()
    else:
        raise ValueError(f"Unknown Quest scene: {scene_name}")


def capture_cloth_reset(garment):
    """Snapshot particle geometry and pose, restoring zero particle velocity.

    Restoring only the cloth Xform leaves a folded mesh folded. Restoring only
    particles leaves momentum from the previous episode. Both are reset here.
    """
    def copy(value):
        return value.clone() if hasattr(value, "clone") else value.copy()

    view = garment._cloth_prim_view
    positions = copy(view.get_world_positions())
    velocities = view.get_velocities()
    zero_velocity = velocities*0
    position, orientation = garment.get_world_pose()
    position, orientation = copy(position), copy(orientation)

    def reset():
        garment.set_world_pose(position=copy(position), orientation=copy(orientation))
        view.set_world_positions(copy(positions))
        view.set_velocities(copy(zero_velocity))

    return reset


def make_cloth_scene_cfg(robot_cfg, *, record, garment_name):
    from humanoid_garment_fold.tasks.teleop_scene import (
        GarmentFoldSceneCfg, ROBOT_BASE_POS, ROBOT_BASE_ROT,
    )
    from quest_cameras import make_ego_cam_cfg, make_wrist_cam_cfg

    cfg = GarmentFoldSceneCfg(num_envs=1, env_spacing=2.)
    # The committed Table038 asset is already Z-up. The garment scene used
    # to apply a hardcoded +90deg X conversion, incorrectly laying it on its
    # side. Derive conversion from each selected asset instead of assuming Y-up.
    for name in ("table", "scene_usd"):
        workspace = getattr(cfg, name, None)
        if workspace is not None:
            rotation = asset_rotation_to_z_up(workspace.spawn.usd_path)
            workspace.init_state = workspace.init_state.replace(rot=rotation)
    cfg.robot = robot_cfg.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=robot_cfg.init_state.replace(pos=ROBOT_BASE_POS, rot=ROBOT_BASE_ROT),
    )
    # Reuse the working Quest camera pipeline. The task's TiledCameras have a
    # known freeze on stock Isaac Lab 2.3.2; plain Quest Cameras do not need the
    # task-specific downgraded image. No success-checker/Open3D import is needed.
    cfg.left_wrist = cfg.right_wrist = cfg.top_camera = cfg.scene_camera = None
    cfg.ego_cam = make_ego_cam_cfg() if record else None
    cfg.wrist_cam = make_wrist_cam_cfg()
    cfg.wrist_cam_right = make_wrist_cam_cfg(body="link6", name="wrist_cam_right", mirror=True)

    def initialize(scene, sim):
        from omegaconf import OmegaConf
        import numpy as np
        from humanoid_garment_fold.assets.garment_object import GarmentObject
        from humanoid_garment_fold.tasks.challenge_garment_loader import ChallengeGarmentLoader
        from humanoid_garment_fold.tasks.garment_pioneer_cfg import GarmentPioneerEnvCfg

        defaults = GarmentPioneerEnvCfg()
        loader = ChallengeGarmentLoader(defaults.garment_cfg_base_path)
        garment_config = loader.load_garment_config(garment_name, defaults.garment_version)
        garment_type = loader._get_garment_type(garment_name)
        directory = Path(defaults.garment_cfg_base_path)/defaults.garment_version/garment_type/garment_name
        # Challenge JSONs use /Assets/... paths; the committed samples live
        # beside the JSON. Avoid importing the full RL env/success checker.
        garment_config.asset_path = str(directory/Path(garment_config.asset_path).name)
        garment_config.visual_usd_paths = [str(directory/Path(p).name)
                                          for p in (garment_config.get("visual_usd_paths") or []) if p]
        if not Path(garment_config.asset_path).is_file():
            raise FileNotFoundError(f"Garment asset missing: {garment_config.asset_path}")
        garment = GarmentObject(
            prim_path=f"/World/Object/{garment_name}",
            particle_config=OmegaConf.load(defaults.particle_cfg_path),
            garment_config=garment_config, rng=np.random.RandomState(42),
        )
        for _ in range(20):
            sim.step(render=True)
        garment.initialize()
        scene.garment = garment
        scene.quest_scene_hooks.reset = capture_cloth_reset(garment)
        print(f"[Quest] Cloth initialized: {garment_name}; left Y restores cloth particles and arms.", flush=True)

    return cfg, SceneHooks(initialize=initialize)
