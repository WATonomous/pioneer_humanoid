"""Whole-body foothold walking on mild non-stair terrain, without gravity assistance."""
import copy
import os
from uuid import uuid4

from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from .flat_env_cfg import _play
from .foothold_env_cfg import PioneerWholeBodyFootholdEnvCfg, _REPO


@configclass
class PioneerWholeBodyRoughSpecialistEnvCfg(PioneerWholeBodyFootholdEnvCfg):
    """Keep the reference robot/rewards/interface; specialize only terrain and assistance.

    Each config gets its own predictor output directory. Warm starts must also
    set load_run_dir explicitly; fresh runs retain the existing predictor training.
    Generator ranges are curriculum parameters, not validated walking limits.
    """

    log_dir: str = ""
    load_run_dir: str | None = None

    def __post_init__(self):
        if not self.log_dir:
            self.log_dir = os.path.join(
                _REPO, "outputs", "rl", "pioneer_whole_body_rough_specialist", "predictors", uuid4().hex
            )
        super().__post_init__()
        generator = copy.deepcopy(self.scene.terrain.terrain_generator)
        existing = generator.sub_terrains
        generator.sub_terrains = {
            "flat": MeshPlaneTerrainCfg(proportion=0.20),
            "random_rough": existing["random_rough"].replace(
                proportion=0.30, noise_range=(0.01, 0.05), noise_step=0.01
            ),
            "boxes": existing["boxes"].replace(proportion=0.25, grid_height_range=(0.02, 0.08)),
            # slope_range is grade: 0.20 is about 11.3 degrees, not 20 degrees.
            "hf_pyramid_slope": existing["hf_pyramid_slope"].replace(proportion=0.125, slope_range=(0.0, 0.20)),
            "hf_pyramid_slope_inv": existing["hf_pyramid_slope_inv"].replace(proportion=0.125, slope_range=(0.0, 0.20)),
        }
        generator.curriculum = True
        self.scene.terrain.terrain_generator = generator
        self.scene.terrain.max_init_terrain_level = 0
        self.events.gravity_assist = None
        self.curriculum.gravity_offset_level = None


@configclass
class PioneerWholeBodyRoughSpecialistEnvCfg_PLAY(PioneerWholeBodyRoughSpecialistEnvCfg):
    """Default to the published predictor; trained policies must use their matched predictor."""

    # Stock play.py replaces log_dir with the selected checkpoint's directory.
    # Do not silently fall back to the published predictor for another policy.
    log_dir: str = os.path.join(_REPO, "models", "pioneer_whole_body_rough_foothold")

    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.scene.terrain.max_init_terrain_level = None
        generator = self.scene.terrain.terrain_generator
        generator.num_rows, generator.num_cols, generator.curriculum = 5, 5, False
        for group in (self.observations.policy, self.observations.critic):
            group.foothold_scan.params.update(train_fpm=False, force_model=True)
        self.observations.policy.foothold_scan.params["show_patches"] = os.environ.get("SHOW_FOOTHOLDS", "0") == "1"
