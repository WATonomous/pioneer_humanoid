"""Pioneer stairs starter recipe, independent of the rough-only experiments.

Terrain, course completion/curriculum, forward commands, and resets are task-specific.
Robot geometry, selective self-collision, observations, actions, rewards, and
physics come from the repository's selective-knee-shaping recipe. This scaffold
does not import local experiment helpers or claim a trained stairs policy.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ...mdp import stair_course as course_mdp
from ...terrains.stair_course import COURSE_SIZE, COURSE_SPAWN_X, MeshStairCourseTerrainCfg, course_goal_distance
from .rough_env_cfg import PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg


PIONEER_STAIRS_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    size=COURSE_SIZE,
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # Every stairs tile includes ascent, a raised walkway, AND descent.
        # Keep the full course first: a single preview env uses column zero.
        "stair_course": MeshStairCourseTerrainCfg(
            proportion=0.8,
            step_height_range=(0.03, 0.15),
            step_width=0.35,
            num_steps=12,
            approach_length=2.0,
            walkway_length=2.0,
            spawn_x=COURSE_SPAWN_X,
        ),
        # Zero-height courses retain the SAME spawn and finish coordinates;
        # stock plane terrain would instead spawn at the center of the tile.
        "flat": MeshStairCourseTerrainCfg(proportion=0.2, step_height_range=(0.0, 0.0)),
    },
)


@configclass
class PioneerHumanoidStairsEnvCfg(PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg):
    """Walk up, cross a raised walkway, walk down, and reach the flat exit.

    Start every robot at row 0, before the ascent. A complete upright traversal
    advances difficulty; failures demote it. The original walking rewards stay
    unchanged. This is not yet a validated stairs policy.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = PIONEER_STAIRS_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 0
        course = self.scene.terrain.terrain_generator.sub_terrains["stair_course"]
        completion_params = {"goal_distance": course_goal_distance(course),
                             "lateral_limit": 2.0, "minimum_height": 0.65, "tilt_limit": 0.8}
        self.terminations.course_complete = DoneTerm(
            func=course_mdp.course_complete, params=completion_params.copy(), time_out=True,
        )
        self.terminations.course_out_of_bounds = DoneTerm(
            func=course_mdp.course_out_of_bounds, params={"lateral_limit": 2.0, "backward_limit": -0.7},
        )
        self.curriculum.terrain_levels = CurrTerm(
            func=course_mdp.complete_stair_course_levels, params=completion_params.copy(),
        )
        self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None
        # The slowest commanded speed needs ~50 s just to cover the full route;
        # allow extra time for foothold adjustment on both twelve-step flights.
        self.episode_length_s = 75.0

        command = self.commands.base_velocity
        command.heading_command = True
        command.rel_heading_envs = 1.0
        command.ranges.lin_vel_x = (0.25, 0.6)
        command.ranges.lin_vel_y = (0.0, 0.0)
        # This bound permits heading corrections, not deliberate turning tasks.
        command.ranges.ang_vel_z = (-0.5, 0.5)
        command.ranges.heading = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["x"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["y"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.15, 0.15)


@configclass
class PioneerHumanoidStairsEnvCfg_PLAY(PioneerHumanoidStairsEnvCfg):
    """Repeatable 8 cm full-course preview, without level advancement or noise."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        generator = self.scene.terrain.terrain_generator
        generator.num_rows = 5
        generator.num_cols = 10
        # Keep fixed terrain families per column. False randomly draws families
        # and can omit one; True here does NOT advance robots (term is None).
        generator.curriculum = True
        generator.sub_terrains["stair_course"].step_height_range = (0.08, 0.08)

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
