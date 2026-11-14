"""Garment-fold env for the pioneer_bimanual_arm_8dof.

NEW (WATonomous) -- not vendored. Subclasses the vendored `GarmentEnv`: inherits
the particle-cloth garment object, distance reward, success checker and reset
flow unchanged; overrides only the pieces that referenced two separate SO101
articulations (`left_arm` / `right_arm`).
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera

from humanoid_garment_fold.tasks.garment_env import GarmentEnv
from humanoid_garment_fold.tasks.challenge_garment_loader import ChallengeGarmentLoader
from humanoid_garment_fold.tasks.garment_pioneer_cfg import (
    TABLE038_POS,
    GarmentPioneerEnvCfg,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_JOINTS,
)
from humanoid_garment_fold.assets.scene import (
    MARBLE_BEDROOM_USD_PATH,
    TABLE038_USD_PATH,
)
from humanoid_garment_fold.utils.logger import get_logger
from pioneer_humanoid.arm_8dof_params import GRIPPER_STROKE

logger = get_logger(__name__)


def fix_garment_asset_paths(garment_loader, garment_config, garment_name, garment_cfg_base_path, garment_version):
    """Upstream resolves the garment json's `asset_path` ("/Assets/...") against
    os.getcwd() (the challenge repo root). We keep garments under
    `vendor_assets/garments/<version>/<type>/<name>/`, so rewrite the mesh path
    to sit next to the json we just loaded. Mutates `garment_config` in place.

    Standalone (not a GarmentPioneerEnv method) so the teleop scene registration
    (humanoid_isaac_scenes/garment_fold) can build a GarmentObject the same way
    without going through the full DirectRLEnv lifecycle.
    """
    gtype = garment_loader._get_garment_type(garment_name)
    gdir = os.path.join(garment_cfg_base_path, garment_version, gtype, garment_name)
    raw = garment_config.get("asset_path", "")
    if raw:
        garment_config.asset_path = os.path.join(gdir, os.path.basename(raw))
    fixed = [os.path.join(gdir, os.path.basename(v))
             for v in (garment_config.get("visual_usd_paths", []) or []) if v]
    garment_config.visual_usd_paths = fixed


class GarmentPioneerEnv(GarmentEnv):
    cfg: GarmentPioneerEnvCfg

    def __init__(self, cfg: GarmentPioneerEnvCfg, render_mode: str | None = None, **kwargs):
        # replicate GarmentEnv.__init__ WITHOUT its left_arm/right_arm tail
        self.cfg = cfg
        self.action_scale = self.cfg.action_scale
        self.object = None
        self._last_computed_reward = 0.0

        self.garment_loader = ChallengeGarmentLoader(cfg.garment_cfg_base_path)
        self.garment_config = self.garment_loader.load_garment_config(
            cfg.garment_name, cfg.garment_version
        )
        self._fix_garment_asset_paths(cfg)
        self.particle_config = OmegaConf.load(cfg.particle_cfg_path)

        if cfg.use_random_seed:
            self.garment_rng = np.random.RandomState()
        else:
            self.garment_rng = np.random.RandomState(cfg.random_seed)

        cfg.viewer = cfg.viewer.replace(eye=(2.0, -2.4, 1.6), lookat=(0.0, 0.2, 0.4))

        # DirectRLEnv.__init__ (grandparent) -- skips GarmentEnv.__init__ tail
        super(GarmentEnv, self).__init__(cfg, render_mode, **kwargs)

        dev = self.device
        self._l_arm = torch.tensor(
            self.robot.find_joints(LEFT_ARM_JOINTS, preserve_order=True)[0], device=dev)
        self._r_arm = torch.tensor(
            self.robot.find_joints(RIGHT_ARM_JOINTS, preserve_order=True)[0], device=dev)
        self._l_grip = torch.tensor(
            self.robot.find_joints(LEFT_GRIPPER_JOINTS, preserve_order=True)[0], device=dev)
        self._r_grip = torch.tensor(
            self.robot.find_joints(RIGHT_GRIPPER_JOINTS, preserve_order=True)[0], device=dev)

    # ------------------------------------------------------------------ assets
    def _fix_garment_asset_paths(self, cfg: GarmentPioneerEnvCfg) -> None:
        fix_garment_asset_paths(
            self.garment_loader, self.garment_config,
            cfg.garment_name, cfg.garment_cfg_base_path, cfg.garment_version,
        )

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.cfg.robot.init_state.pos = self.cfg.robot_base_pos
        self.cfg.robot.init_state.rot = self.cfg.robot_base_rot
        self.robot = Articulation(self.cfg.robot)

        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.left_camera = TiledCamera(self.cfg.left_wrist)
        self.right_camera = TiledCamera(self.cfg.right_wrist)
        self.scene_camera = TiledCamera(self.cfg.scene_camera)

        self._build_worksurface()
        self._create_garment_object()

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["left_camera"] = self.left_camera
        self.scene.sensors["right_camera"] = self.right_camera
        self.scene.sensors["scene_camera"] = self.scene_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _build_worksurface(self):
        """LeHome spawns `Scene_00_Apartment.usd` at /World/Scene (photoreal
        apartment + the table the garment rests on). Order:
          1. optional NuRec backdrop (visual only)
          2. that scene USD if the file exists (default)
          3. ground plane + vendored Table038.usd
          4. ground plane only (garment then drops to the floor)
        """
        if self.cfg.backdrop_usd_path and os.path.isfile(self.cfg.backdrop_usd_path):
            s = self.cfg.backdrop_scale
            bd = sim_utils.UsdFileCfg(usd_path=self.cfg.backdrop_usd_path, scale=(s, s, s))
            bd.func("/World/Backdrop", bd, translation=self.cfg.backdrop_pos)

        if os.path.isfile(MARBLE_BEDROOM_USD_PATH):
            cfg = sim_utils.UsdFileCfg(usd_path=MARBLE_BEDROOM_USD_PATH)
            cfg.func("/World/Scene", cfg, translation=(0.0, 0.0, 0.0),
                     orientation=(0.0, 0.0, 0.0, 0.0))
            return

        sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
        if os.path.isfile(TABLE038_USD_PATH):
            t = sim_utils.UsdFileCfg(
                usd_path=TABLE038_USD_PATH,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            t.func("/World/Table", t, translation=TABLE038_POS)
        else:
            logger.warning(
                "[GarmentPioneerEnv] no scene or table USD found -- garment will "
                "drop to the ground plane. Copy Scene_00_Apartment.usd into "
                "vendor_assets/scenes/marble/."
            )

    # ------------------------------------------------------------------ actions
    def _apply_action(self) -> None:
        a = self.actions
        self.robot.set_joint_position_target(a[:, :7], joint_ids=self._l_arm)
        self.robot.set_joint_position_target(a[:, 7:8] * GRIPPER_STROKE, joint_ids=self._l_grip)
        self.robot.set_joint_position_target(a[:, 8:15], joint_ids=self._r_arm)
        self.robot.set_joint_position_target(a[:, 15:16] * GRIPPER_STROKE, joint_ids=self._r_grip)

    # --------------------------------------------------------------- observation
    def _get_observations(self) -> dict:
        jp = self.robot.data.joint_pos
        state = torch.cat([jp[:, self._l_arm], jp[:, self._l_grip] / GRIPPER_STROKE,
                           jp[:, self._r_arm], jp[:, self._r_grip] / GRIPPER_STROKE], dim=-1).squeeze(0)
        top_rgb = self.top_camera.data.output["rgb"]
        top_depth = self.top_camera.data.output["depth"].squeeze()
        left_rgb = self.left_camera.data.output["rgb"]
        right_rgb = self.right_camera.data.output["rgb"]
        depth_mm = np.clip(
            top_depth.cpu().detach().numpy().copy() * 1000, 0, 65535
        ).astype(np.uint16)
        act = self.actions.squeeze(0) if hasattr(self, "actions") \
            else torch.zeros(16, device=self.device)
        return {
            "action": act.cpu().detach().numpy(),
            "observation.state": state.cpu().detach().numpy(),
            "observation.images.top_rgb": top_rgb.cpu().detach().numpy().squeeze(),
            "observation.images.left_rgb": left_rgb.cpu().detach().numpy().squeeze(),
            "observation.images.right_rgb": right_rgb.cpu().detach().numpy().squeeze(),
            "observation.top_depth": depth_mm,
        }

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # DirectRLEnv._reset_idx (grandparent) -- GarmentEnv._reset_idx touches left/right_arm
        super(GarmentEnv, self)._reset_idx(env_ids)
        self._last_computed_reward = 0.0
        default_jp = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(default_jp, joint_ids=None, env_ids=env_ids)
        if self.object is not None:
            self.object.reset()
        if getattr(self, "texture_cfg", {}) and self.texture_cfg.get("enable", False):
            self._randomize_table038_texture()
        if getattr(self, "light_cfg", {}) and self.light_cfg.get("enable", False):
            self._randomize_light()
