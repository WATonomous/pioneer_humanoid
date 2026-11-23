"""Onshape straight staircase with foothold-conditioned perception ("Watch Your Step").

This is the rough-terrain foothold task (``config/pioneer/foothold_env_cfg.py``, the
ETH Zurich foothold-prediction port) moved onto the Onshape straight course:

* the **policy** sees, per foot, the *predicted* next foothold (x, y in the base
  frame) and a height patch (5 cm cells) around it -- perception is centred where
  the feet are going. The patch is 7 x 7 with one extra row behind the heel
  (8 x 7), because on a descent the step-off edge a heel can clip sits at the
  7 x 7 patch's rear boundary;
* the **critic** (training only) still sees the whole height map, which is what
  makes this an asymmetric actor-critic;
* the foot predictor is refit every 500 iterations (``FootPredictionSupervisor``)
  against whatever terrain this task runs on, writing into ``log_dir``;
* the action space is the **12 leg joints**; the arms are swung on the gait clock
  by ``ArmSwingAction`` (0 action dims), so this policy is warm-startable from any
  other leg-only foothold walker.

The course overrides below (USD terrain, marker spawn, +Y heading command, course
terminations, progress rewards, no terrain curriculum) are the same ones
``PioneerHumanoidOnshapeCourseArmSwingEnvCfg`` uses, so the two tasks differ only
in how the policy perceives the terrain and whether the arms are learned.
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

        # --- terrain: the user's straight staircase USD (a single mesh) ------
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.usd_path = _COURSE_USD
        self.scene.env_spacing = 0.0
        # The base scan is already 5 cm (the foothold patches are 5 cm cells);
        # only the collider path changes, the course is one merged mesh.
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]

        # --- one extra patch row behind the heel -----------------------------
        # A 7 x 7 patch at 5 cm reaches 15 cm behind the predicted foothold, but
        # the ankle sits 8.0 cm from the heel of a 24.3 cm sole, so the step-off
        # edge a descending heel can clip is right at the patch's rear boundary.
        # One extra row extends the rear reach to 20 cm; the extra heights are
        # flattened first, so the original 49 entries keep their order (see
        # foothold.foothold_scan). The critic sees the same patch layout so the
        # asymmetric pair stays consistent.
        for group in (self.observations.policy, self.observations.critic):
            group.foothold_scan.params["patch_rows_behind"] = 1

        # --- spawn on the Part Studio marker, facing +Y ----------------------
        # (replaced below by the multi-slot course spawn; kept as the slot list's
        # first entry so the original spawn is still one of the draws)

        # --- spawn along the course ------------------------------------------
        # Every episode used to start 1.5 m before the first riser, so the
        # 14.5-16 cm parts and their descents were a small slice of each 46 s
        # episode. Start instead at the foot of any part: the y slots are the
        # centres of the 0.4 m ground-level gaps measured off the merged
        # collider (assets/staircase_onshape/straight_stairs_centreline.npz),
        # where the whole 24 cm sole sits on flat ground. Jitters are small for
        # the same reason (0.45 m of gap, 0.24 m of sole).
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

        # --- command: walk the course, no standing reserves ------------------
        command = self.commands.base_velocity
        command.heading_command = True
        command.rel_heading_envs = 1.0
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (0.3, 0.5)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.5, 0.5)
        command.ranges.heading = (_YAW_90, _YAW_90)

        # 20.5 m of course at ~0.4 m/s, with room for the slower parts.
        self.episode_length_s = 75.0

        # --- a fixed course has nothing to curriculum ------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.gravity_offset_level = None
        self.events.gravity_assist = None

        # --- course terminations (axis 1 = the course's +Y) ------------------
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
        # A ray miss resets: the foothold patches sample the scan at specific
        # cells, so a mesh hole must not silently become a foothold height.
        self.terminations.invalid_height_scan = DoneTerm(
            func=v1_terminations.height_scan_invalid,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

        # --- pay for climbing, not just for tracking velocity ----------------
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

        # --- soft landings ---------------------------------------------------
        # Squared downward foot speed on terrain-contact frames only, i.e. the
        # touchdown spike and nothing in swing. Measured on model_3549: the
        # touchdown frames sum 91.7 (m/s)^2 over a 75 s episode, which at the
        # -0.5 the arm-swing task used is worth 0.012 reward units -- inert, and
        # exactly why that task never lost its "hard landing on the way down".
        # -5.0 is ~0.12 units, about 9 % of the 1.34 forward-progress income, so
        # a 1 m/s slam costs ~4x a normal 0.5 m/s touchdown without making the
        # descent itself expensive.
        self.rewards.feet_impact = RewTerm(
            func=course_mdp.feet_impact_contact,
            weight=-5.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=["l_foot", "r_foot"], preserve_order=True),
            },
        )

        # --- ramp the push magnitude over the run ----------------------------
        # push_robot kicks with up to 0.8 m/s per axis every 5-10 s, which is a
        # huge instantaneous velocity change; starting at ~1/3 of it lets the
        # gait settle before the full disturbance arrives. The endpoint is the
        # configured 0.8 m/s, so the disturbance the policy is finally optimised
        # against is unchanged and runs stay comparable. 6000 env steps is
        # ~250 iterations at 24 steps per iteration.
        self.curriculum.push_speed = CurrTerm(
            func=course_mdp.push_speed_schedule,
            params={"term_name": "push_robot", "start_scale": 0.35, "final_scale": 1.0, "ramp_steps": 6000},
        )

        # --- torso stability, slightly ---------------------------------------
        # Measured on model_2999 (28 s, pushes off): pitch -0.9 +- 4.6 deg, roll
        # +0.3 +- 3.2 deg, body-frame angular rate |wx| 0.36 / |wy| 0.44 rad/s.
        # The mean tilt is ~0, so the defect is oscillation rather than a lean:
        # price the rates (ang_vel_xy_l2) and the tilt amplitude
        # (flat_orientation_l2) a little harder. Both moves are deliberately
        # small -- at these magnitudes the rate term goes from ~7 % to ~14 % of
        # the velocity-tracking income, which is enough to matter without
        # reshaping the gait the walk was trained on.
        self.rewards.ang_vel_xy_l2.weight = -0.10       # was -0.05
        self.rewards.flat_orientation_l2.weight = -1.5  # was -1.0


@configclass
class PioneerHumanoidStairsFootholdEnvCfg_PLAY(PioneerHumanoidStairsFootholdEnvCfg):
    """Repeatable visual check: exact marker spawn, predictor frozen, no corruption."""

    log_dir: str = os.path.join(_REPO, "models", "pioneer_humanoid_stairs_foothold")

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 3
        self.scene.env_spacing = 2.0
        self.episode_length_s = 1200.0
        # Side camera riding on the robot, the same framing the other PLAY
        # configs use, so a recorded clip shows the gait instead of the whole
        # course from the default aerial view.
        self.viewer = ViewerCfg(
            eye=(1.4, -3.0, 0.1), lookat=(0.3, 0.0, -0.15), origin_type="asset_root", asset_name="robot"
        )
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum.push_speed = None
        # Disabled for the visual so a partial scan miss at a mesh edge does not
        # end the clip; the bounding term below covers real escapes.
        self.terminations.invalid_height_scan = None
        # Pin the multi-slot course spawn to the Part Studio marker for a
        # repeatable visual.
        self.events.reset_base.params["spawn_y"] = (_SPAWN_Y,)
        self.events.reset_base.params["x_jitter"] = 0.0
        self.events.reset_base.params["y_jitter"] = 0.0
        # Use the trained predictor as-is and never refit it in a visual run.
        for group in (self.observations.policy, self.observations.critic):
            group.foothold_scan.params.update(train_fpm=False, force_model=True)
        self.observations.policy.foothold_scan.params["show_patches"] = (
            os.environ.get("SHOW_FOOTHOLDS", "0") == "1"
        )
