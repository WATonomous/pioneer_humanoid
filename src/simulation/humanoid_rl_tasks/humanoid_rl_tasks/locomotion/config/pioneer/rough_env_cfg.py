"""Pioneer leg-only humanoid on uneven ground: the flat task's gait-clock recipe, blind on mild terrain, or with a
height scan on Isaac Lab's full rough course (stairs up to 23 cm)."""
import isaaclab.terrains as terrain_gen
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import gait
from .flat_env_cfg import PioneerLegOnlyFlatEnvCfg, _play

# mild terrain a 32 kg robot with 3.7 rad/s knees can cross blind: bumps, box grids, gentle slopes, low steps
BLIND_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.25, noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.25),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(proportion=0.2, grid_width=0.45, grid_height_range=(0.02, 0.08), platform_width=2.0),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.1, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=0.1, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.03, 0.08), step_width=0.35, platform_width=3.0, border_width=1.0, holes=False
        ),
        "stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.03, 0.08), step_width=0.35, platform_width=3.0, border_width=1.0, holes=False
        ),
    },
)


@configclass
class PioneerLegOnlyRoughEnvCfg(PioneerLegOnlyFlatEnvCfg):
    """Blind: no height scan in the observation. A height scanner still measures the base height above the local ground for
    the reward. Terrain difficulty follows each robot's progress."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.5, 0.5]),
            mesh_prim_paths=["/World/ground"],
        )
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        self.rewards.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        # on rough ground stepping in place was safer than walking slowly (0.3 m/s asked -> 0.04 done): make tracking count
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.35
        # world-z swing height says nothing about clearance on raised ground
        self.rewards.feet_swing_height.func = gait.feet_swing_height_relative


@configclass
class PioneerLegOnlyRoughEnvCfg_PLAY(PioneerLegOnlyRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _play(self)
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class PioneerLegOnlyPerceptiveEnvCfg(PioneerLegOnlyRoughEnvCfg):
    """Isaac Lab's full rough course (stairs up to 23 cm, 20 cm boxes, 40 % slopes), with a 1.6 x 1.0 m height scan
    around the robot (187 rays) in the observation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.replace(curriculum=True)
        self.scene.terrain.max_init_terrain_level = 3
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            modifiers=[ModifierCfg(func=gait.finite)],
            clip=(-1.0, 1.0),
        )
        self.rewards.feet_swing_height.weight = -40.0


@configclass
class PioneerLegOnlyPerceptiveEnvCfg_PLAY(PioneerLegOnlyPerceptiveEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _play(self)
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
