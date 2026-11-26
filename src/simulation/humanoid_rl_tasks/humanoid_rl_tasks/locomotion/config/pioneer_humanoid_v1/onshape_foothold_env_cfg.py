"""Onshape straight staircase with foothold-conditioned perception ("Watch Your Step").

The rough foothold task (``config/pioneer/foothold_env_cfg.py``) on the Onshape straight course:
* the policy sees, per foot, the predicted next foothold plus an 8 x 7 height patch (5 cm cells; one
  extra row behind the heel for descents); the critic sees the whole height map;
* the foot predictor is refit every 500 iterations (``FootPredictionSupervisor``), writing into ``log_dir``;
* the action space is the 12 leg joints; the arms swing on the gait clock (``ArmSwingAction``, 0 action dims).
"""

import math
import os

from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ...mdp import stair_course as course_mdp
from ...mdp import terminations as v1_terminations
from ..pioneer.foothold_env_cfg import PioneerWholeBodyFootholdEnvCfg

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7))

_YAW_90 = math.pi / 2.0
_SPAWN_X = 4.163
_SPAWN_Y = -1.549
_COURSE_USD = "assets/staircase_onshape/straight_stairs_terrain.usd"


@configclass
class PioneerHumanoidStairsFootholdEnvCfg(PioneerWholeBodyFootholdEnvCfg):
    """Straight Onshape stairs, foothold-conditioned perception, arms scripted."""

    log_dir: str = os.path.join(_REPO, "outputs", "rl", "pioneer_humanoid_stairs_foothold", "foot_models")

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.usd_path = _COURSE_USD
        self.scene.env_spacing = 0.0
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]

        # one extra patch row behind the heel (the critic sees the same layout)
        for group in (self.observations.policy, self.observations.critic):
            group.foothold_scan.params["patch_rows_behind"] = 1

        # spawn at the foot of any part (centres of the 0.4 m ground gaps)
        self.events.reset_base.func = course_mdp.reset_root_state_course_spawn
        self.events.reset_base.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "spawn_x": _SPAWN_X,
            "spawn_y": (_SPAWN_Y, 1.58, 4.08, 7.08, 10.68, 14.83),
            "x_jitter": 0.15,
            "y_jitter": 0.06,
            "yaw": _YAW_90,
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }

        command = self.commands.base_velocity
        command.heading_command = True
        command.rel_heading_envs = 1.0
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (0.3, 0.5)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.5, 0.5)
        command.ranges.heading = (_YAW_90, _YAW_90)

        self.episode_length_s = 75.0

        # fixed course: no curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.gravity_offset_level = None
        self.events.gravity_assist = None

        # course terminations (axis 1 = the course's +Y)
        self.terminations.course_complete = DoneTerm(
            func=course_mdp.course_complete,
            params={
                "goal_distance": 20.5,
                "lateral_limit": 6.0,
                "minimum_height": 0.65,
                "tilt_limit": 0.8,
                "axis": 1,
            },
            time_out=True,
        )
        self.terminations.course_out_of_bounds = DoneTerm(
            func=course_mdp.course_out_of_bounds,
            params={"lateral_limit": 8.0, "backward_limit": -8.0, "axis": 1},
        )
        # a ray miss resets: a mesh hole must not become a foothold height
        self.terminations.invalid_height_scan = DoneTerm(
            func=v1_terminations.height_scan_invalid,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

        self.rewards.course_forward_progress = RewTerm(
            func=course_mdp.course_forward_progress,
            weight=5.0,
            params={"axis": 1, "cap": 0.5},
        )
        self.rewards.course_retreat_cost = RewTerm(
            func=course_mdp.course_retreat_cost,
            weight=-2.0,
            params={"axis": 1},
        )

        # soft landings
        self.rewards.feet_impact = RewTerm(
            func=course_mdp.feet_impact_contact,
            weight=-5.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=["l_foot", "r_foot"], preserve_order=True),
            },
        )

        # ramp the push magnitude 0.35 -> 1.0 over ~250 iterations
        self.curriculum.push_speed = CurrTerm(
            func=course_mdp.push_speed_schedule,
            params={"term_name": "push_robot", "start_scale": 0.35, "final_scale": 1.0, "ramp_steps": 6000},
        )

        # torso stability
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.flat_orientation_l2.weight = -1.5


@configclass
class PioneerHumanoidStairsFootholdEnvCfg_PLAY(PioneerHumanoidStairsFootholdEnvCfg):
    """Repeatable visual check: exact marker spawn, predictor frozen, no corruption."""

    log_dir: str = os.path.join(_REPO, "models", "pioneer_humanoid_stairs_foothold")

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 3
        self.scene.env_spacing = 2.0
        self.episode_length_s = 1200.0
        # side camera riding on the robot
        self.viewer = ViewerCfg(
            eye=(1.4, -3.0, 0.1), lookat=(0.3, 0.0, -0.15), origin_type="asset_root", asset_name="robot"
        )
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum.push_speed = None
        self.terminations.invalid_height_scan = None
        # spawn on the marker
        self.events.reset_base.params["spawn_y"] = (_SPAWN_Y,)
        self.events.reset_base.params["x_jitter"] = 0.0
        self.events.reset_base.params["y_jitter"] = 0.0
        # use the trained predictor as-is, never refit
        for group in (self.observations.policy, self.observations.critic):
            group.foothold_scan.params.update(train_fpm=False, force_model=True)
        self.observations.policy.foothold_scan.params["show_patches"] = (
            os.environ.get("SHOW_FOOTHOLDS", "0") == "1"
        )
