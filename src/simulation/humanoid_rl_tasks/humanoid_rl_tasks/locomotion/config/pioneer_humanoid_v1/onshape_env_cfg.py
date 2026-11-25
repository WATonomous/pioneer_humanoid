# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Onshape straight staircase course: the user's STEP-built stairs as trained terrain.

Unlike the procedural stair_course generator, the terrain here is the merged
single-mesh USD (straight_stairs_terrain.usd: the 6 stair parts plus the Part Studio's
own ground, metres, triangle-soup collision baked). The RayCaster height scanner
reads that one mesh, so policy observations stay 235-dim and stairs checkpoints
load. The STEP export is not committed; place it in assets/staircase_onshape/
and rebuild the terrain from it with the course build tooling (not part of this
tree).

Spawn: directly on the Part Studio's ``marker`` sliver (x=4.163, y=-1.549), facing
+Y (yaw pi/2), so the local +X velocity command drives the robot up the course.

STRAIGHT STAIRS COURSE (2026-10-19, ``straight_stairs.step`` -> ``straight_stairs_terrain.usd``):
the same Part Studio laid out as 6 stepped pyramids with a built-in difficulty
ramp, so the geometry itself is the curriculum. Risers measured off the STEP
(0.28 m treads throughout):

    part 1   5.0 / 5.5 / 6.0 cm        peak 0.165 m
    part 2   7.0 / 7.5 / 8.0 cm        peak 0.290 m
    part 3   9.0 - 10.5 cm             peak 0.475 m
    part 4  11.5 - 13.5 cm             peak 0.735 m
    part 5  14.5 cm                    peak 1.085 m
    part 6  up to 16 cm                peak 1.540 m

The previous, harder course ramped 10.0 -> 21 cm with a 1.89 m summit; it is
now only in git history. The first riser here is 5 cm, against the ~8.6 cm foot
lift the 2026-10-19 policy already achieves, so the entrance no longer blocks
the gait.
"""

import json
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ...mdp import LidarNoiseCfg
from ...mdp import arm_swing_coupling
from ...mdp import feet_duty_symmetry
from ...mdp import joint_bias_mirror
from ...mdp import joint_deviation_l1
from ...mdp import push_by_setting_velocity
from ...mdp import shove as shove_mdp
from ...mdp import stair_course as course_mdp
from ...mdp import terminations as mdp
from .rough_env_cfg import (
    FOOT_BODIES,
    LEG_JOINTS,
    PioneerHumanoidRoughNoStairsSelectiveKneeShapeRewards,
    PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg,
)

from pioneer_humanoid.whole_body import (
    ARM_SWING_JOINTS,
    WHOLE_BODY_ARM_SWING_CFG,
    WHOLE_BODY_WALKER_CFG,
)

# Body names for the whole-body walker's leg links.
ONSHAPE_FOOT_BODIES = FOOT_BODIES


_ONSHAPE_COURSE_USD = "assets/staircase_onshape/straight_stairs_terrain.usd"
_ONSHAPE_COURSE_PROFILE = "assets/staircase_onshape/straight_stairs_profile.json"


def _load_course_band() -> list:
    """The course's own shape: the swing clearance each position actually needs.

    Generated from the merged USD by the course build tooling (not in this
    tree). A single global band has
    to cover the tallest riser, so it also pays a high-knee lift everywhere
    else; this table pays only for the riser that is actually coming.
    """
    try:
        with open(_ONSHAPE_COURSE_PROFILE) as handle:
            profile = json.load(handle)
        if "samples" in profile:
            # Modern layout: y is implicit (y0 + index * bin), each row is
            # [terrain height, required clearance band].
            y0 = float(profile["y0"])
            bin_size = float(profile["bin"])
            return [
                [y0 + index * bin_size, float(row[1])]
                for index, row in enumerate(profile["samples"])
            ]
        # Legacy layout: explicit [y, band] rows.
        return [[float(entry[0]), float(entry[1])] for entry in profile["band"]]
    except (OSError, KeyError, ValueError) as err:  # pragma: no cover - container path
        print(f"[ONSHAPE] course profile unavailable ({err}); falling back to the scalar band")
        return []


@configclass
class PioneerHumanoidOnshapeCourseRewards(PioneerHumanoidRoughNoStairsSelectiveKneeShapeRewards):
    """Onshape straight-course specific reward terms."""

    # Forward progress along the course axis (+Y)
    course_forward_progress = RewTerm(
        func=course_mdp.course_forward_progress,
        weight=5.0,
        params={"axis": 1, "cap": 0.5},
    )
    # Penalize retreating (moving backwards along the course)
    course_retreat_cost = RewTerm(
        func=course_mdp.course_retreat_cost,
        weight=-2.0,
        params={"axis": 1},
    )
    # Face the course direction (+Y)
    face_course_direction = RewTerm(
        func=course_mdp.face_course_direction,
        weight=-0.5,
        params={"target": 1.57079632679, "deadband": 0.25},
    )
    # Swing-foot height with position-dependent band (from course profile)
    foot_swing_clearance = RewTerm(
        func=course_mdp.foot_swing_clearance,
        weight=40.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("left_foot_terrain_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_terrain_contact"),
            "feet_cfg": SceneEntityCfg("robot", body_names=ONSHAPE_FOOT_BODIES, preserve_order=True),
            "command_name": "base_velocity",
            "forward_axis": 1,
            "max_clearance": 0.45,
            "band": 0.24,
            "band_profile": _load_course_band(),
            "excess_coef": 30.0,
            "back_allow": -0.02,
            "phase_ramp": 0.06,
        },
    )
    # Penalize rearward foot swing ("butt-kick")
    foot_backswing_cost = RewTerm(
        func=course_mdp.foot_backswing_cost,
        weight=-6.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("left_foot_terrain_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_terrain_contact"),
            "feet_cfg": SceneEntityCfg("robot", body_names=ONSHAPE_FOOT_BODIES, preserve_order=True),
            "forward_axis": 1,
            "deadband": 0.25,
        },
    )
# Course runs +Y as modeled (forward is +Y). Spawn directly on the marker
# sliver (metres, from the marker part's bounds: centre 4.1628, -1.5486),
# facing +Y; course rewards/terminations take axis=1 / target=pi/2 below.
_SPAWN_X = 4.163
_SPAWN_Y = -1.549
_YAW_90 = math.pi / 2

# Random shoving: velocity kicks every 8-12 s; +/-0.5 m/s sideways is the shipping
# setting (+/-0.9 was worse). See STAIR_REWARD_NOTES.md.
_PUSH_INTERVAL_S = (8.0, 12.0)
_PUSH_VELOCITY = {"x": (-0.5, 0.5), "y": (-0.3, 0.3)}


@configclass
class PioneerHumanoidOnshapeCourseEnvCfg(PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg):
    """Onshape course training template (same 19-reward recipe as rough-no-stairs).

    The course runs +Y as modeled, so the spawn faces +Y at yaw pi/2 and every
    course reward / termination takes axis=1 (or target=pi/2). All envs share
    origin ~0 (spacing 0); scatter comes from the pose ranges.
    """

    # Create the course reward set. The inherited feet_crossing_penalty already
    # uses FOOT_BODIES (l_foot/r_foot), so it needs no per-task override.
    rewards = PioneerHumanoidOnshapeCourseRewards()

    def __post_init__(self):
        super().__post_init__()

        # Walk the whole-body model (arms and head held rigid) up the course.
        self.scene.robot = WHOLE_BODY_WALKER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.usd_path = _ONSHAPE_COURSE_USD
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]
        self.scene.env_spacing = 0.0

        # Corrupt the height scan the way a real range sensor fails, not just
        # with range error: Gaussian noise + a fixed per-beam bias + beam
        # dropout (see mdp/noise.py). This replaces the shared +/-10 cm uniform
        # term inherited from locomotion_env_cfg.py. Rates ramp in over the
        # first ~500 iterations so a warm-started checkpoint adapts instead of
        # meeting full corruption on its first rollout. The straight course and
        # the spiral (plus both arm-swing variants) inherit from here; the
        # _PLAY configs strip the term by disabling observation corruption.
        self.observations.policy.height_scan.noise = LidarNoiseCfg()

        # Termination: goal is the flat exit past the final descent
        self.terminations.course_complete = DoneTerm(
            func=course_mdp.course_complete,
            params={"goal_distance": 12.0, "lateral_limit": 2.0, "minimum_height": 0.65, "tilt_limit": 0.8},
            time_out=True,
        )
        self.terminations.course_out_of_bounds = DoneTerm(
            func=course_mdp.course_out_of_bounds,
            params={"lateral_limit": 2.0, "backward_limit": -0.7},
        )
        # Stairs have sustained pitch during climbing; raise base_tilt limit from 0.8
        # (45°) to 1.2 rad (~69°) so normal stair ascent doesn't trigger termination.
        self.terminations.base_tilt = DoneTerm(
            func=mdp.base_tilt_over_limit, params={"limit": 1.2}
        )
        # Curriculum: advance difficulty on upright completion
        self.curriculum.terrain_levels = CurrTerm(
            func=course_mdp.complete_stair_course_levels,
            params={"goal_distance": 12.0, "lateral_limit": 2.0, "minimum_height": 0.65, "tilt_limit": 0.8},
        )

        # Episode long enough for full traversal (up + walkway + down)
        self.episode_length_s = 75.0

        # Forward velocity command with heading
        command = self.commands.base_velocity
        command.heading_command = True
        command.rel_heading_envs = 1.0
        command.ranges.lin_vel_x = (0.25, 0.6)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.5, 0.5)
        command.ranges.heading = (0.0, 0.0)

        # Small spawn jitter
        self.events.reset_base.params["pose_range"]["x"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["y"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.15, 0.15)

        # --- Onshape straight course spawn: marker at (4.163, -1.549), facing +Y ---
        self.events.reset_base.params["pose_range"]["x"] = (3.66, 4.66)
        self.events.reset_base.params["pose_range"]["y"] = (-2.05, -1.05)
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (_YAW_90, _YAW_90)
        # Fixed course, fixed start: spawn at rest on the marker
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.5)
        self.commands.base_velocity.ranges.heading = (_YAW_90, _YAW_90)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Knee target clip widened to the hardware limit so the tall risers are within
        # sagittal reach (task-local; see STAIR_REWARD_NOTES.md).
        self.actions.joint_pos.clip = {"Knee_.*": (-1.10, -0.05)}
        # Model-fix era: the symmetry-compensation terms below are weight 0 because
        # the URDF axis defects they compensated for are fixed (STAIR_REWARD_NOTES.md).
        # Soft landings: -0.5 (the earlier weight was inert against the progress income).
        self.rewards.feet_impact = RewTerm(
            func=course_mdp.feet_impact, weight=-0.5,
            params={
                "left_sensor_cfg": SceneEntityCfg("left_foot_terrain_contact"),
                "right_sensor_cfg": SceneEntityCfg("right_foot_terrain_contact"),
                "feet_cfg": SceneEntityCfg("robot", body_names=ONSHAPE_FOOT_BODIES, preserve_order=True),
            },
        )

        # Land flat: whole-landing-window orientation penalty. -15 is deliberate; a
        # smaller weight is inert against the progress income. See STAIR_REWARD_NOTES.md.
        self.rewards.foot_flat = RewTerm(
            func=course_mdp.foot_flat_l2,
            weight=-15.0,
            params={
                "left_sensor_cfg": SceneEntityCfg("left_foot_terrain_contact"),
                "right_sensor_cfg": SceneEntityCfg("right_foot_terrain_contact"),
                "feet_cfg": SceneEntityCfg("robot", body_names=ONSHAPE_FOOT_BODIES, preserve_order=True),
                "deadband": 0.23,
                "landing_window": 0.15,
            },
        )

        # Left/right symmetry terms on slow averages (not the anti-phase swing); held
        # at weight 0 until needed. See STAIR_REWARD_NOTES.md.
        self.rewards.joint_bias_mirror = RewTerm(
            func=joint_bias_mirror,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "tau_s": 1.0, "deadband": 0.05},
        )
        self.rewards.feet_duty_symmetry = RewTerm(
            func=feet_duty_symmetry,
            weight=0.0,
            params={
                "left_sensor_cfg": SceneEntityCfg("left_foot_terrain_contact"),
                "right_sensor_cfg": SceneEntityCfg("right_foot_terrain_contact"),
                "tau_s": 2.0,
            },
        )

        # Price skating: -0.1 was inert, -0.5 makes planting cheaper (STAIR_REWARD_NOTES.md).
        self.rewards.feet_slide.weight = -0.5

        # Fixed course: no curriculum, no generator-only edge termination.
        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None
        # Goal just past the final descent so the summit drop stays in distribution
        # and finished episodes reset promptly (STAIR_REWARD_NOTES.md).
        self.terminations.course_complete.params["goal_distance"] = 20.5
        self.terminations.course_complete.params["lateral_limit"] = 6.0
        self.terminations.course_complete.params["axis"] = 1
        self.terminations.course_out_of_bounds.params = {
            "lateral_limit": 8.0,
            "backward_limit": -8.0,
            "axis": 1,
        }
        # --- Course rewards -------------------------------------------------
        # Capped forward progress; retreat is a real (non-negative) cost.
        self.rewards.face_course_direction.params["target"] = _YAW_90
        # Price hip-abduction splay beyond a deadband set from the leg that already
        # swings correctly (STAIR_REWARD_NOTES.md).
        self.rewards.hip_abduction_splay = RewTerm(
            func=course_mdp.hip_abduction_splay_l2,
            weight=-6.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["Hip_A_L", "Hip_A_R"], preserve_order=True
                ),
                "deadband": 0.30,
            },
        )
        # Knee tuck: 0.28 rad binds on the flat without producing a high-knee march;
        # risers get their tuck from the clearance reward (STAIR_REWARD_NOTES.md).
        self.rewards.knee_swing_flexion.params["minimum_flexion"] = 0.28
        self.rewards.knee_swing_flexion.weight = -4.0
        # Feet stay down: cut the weight, don't raise the threshold (a higher threshold
        # makes the policy swing longer). See STAIR_REWARD_NOTES.md.
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.2
        # Trim two taxes that dominated the landscape: a stand-still yaw-rate
        # tracking income and the action-rate tax (the single biggest cost).
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.action_rate_l2.weight = -0.002
        # -800 is a warm-start setting; a fresh policy should use a smaller penalty
        # (STAIR_REWARD_NOTES.md).
        self.rewards.termination_penalty.weight = -800.0
        # Stability: price torso tilt continuously, strong enough to give a
        # recovery gradient but still slight next to the walk (~-4/ep).
        self.rewards.flat_orientation_l2.weight = -3.0
        # Smooth legs: price joint accelerations on the leg joints so jerky
        # swings cost more than smooth step-throughs.
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
        # Freezing backstop: reset if not displaced 2 m along the course after
        # 15 s. Anchored at the spawn centre since scattered starts sit metres
        # off-origin. Relaxed for arm-swing warm-start: policy stays upright but
        # walks slowly; 1.5 m in 25 s gives more time to build momentum.
        self.terminations.course_no_progress = DoneTerm(
            func=course_mdp.course_no_progress,
            params={"axis": 1, "min_displacement": 1.5, "min_time": 25.0,
                    "spawn": (_SPAWN_X, _SPAWN_Y)},
        )
        # Random shoving: robustness domain randomisation (see the module-level
        # _PUSH_* constants for the calibration).
        self.events.push_robot = EventTerm(
            func=shove_mdp.push_robot_logged,
            mode="interval",
            interval_range_s=_PUSH_INTERVAL_S,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": _PUSH_VELOCITY,
            },
        )
        self.episode_length_s = 1200.0


@configclass
class PioneerHumanoidOnshapeCourseArmSwingEnvCfg(PioneerHumanoidOnshapeCourseEnvCfg):
    """Onshape straight course with learned arm swing (shoulders + elbows).

    Warm-started from the rough checkpoint with network surgery (12 -> 16 actions).
    Uses WHOLE_BODY_ARM_SWING_CFG which actuates l_shoulder_pitch, r_shoulder_pitch,
    l_elbow, r_elbow in addition to the 12 leg joints.
    """

    def __post_init__(self):
        super().__post_init__()

        # Walk the whole-body model with 4 arm joints free (arms + hands held in
        # fist pose). The arm default pose comes from WHOLE_BODY_ARM_SWING_CFG:
        # 0 rad on every arm joint, which is the URDF's arms-at-the-sides pose.
        self.scene.robot = WHOLE_BODY_ARM_SWING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Arm actions scale 0.5 like the framework default (the leg-only dict would
        # leave them at 1.0, 4-16x any leg channel). See STAIR_REWARD_NOTES.md.
        self.actions.joint_pos.scale = {
            **self.actions.joint_pos.scale,
            **dict.fromkeys(ARM_SWING_JOINTS, 0.5),
        }

        # Walking-swing envelope for the arms, not the joint's full range (STAIR_REWARD_NOTES.md).
        self.actions.joint_pos.clip = {
            "Knee_.*": (-1.10, -0.05),
            ".*_shoulder_pitch": (-0.9, 0.45),
            ".*_elbow": (-1.4, 0.0),
        }

        # Anchor the arms to the at-the-sides pose so the action means don't random-walk.
        self.rewards.joint_deviation_arm = RewTerm(
            func=joint_deviation_l1,
            weight=-0.03,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_SWING_JOINTS))},
        )

        # ... and pay for the swing: shoulder velocity follows same-side hip velocity (STAIR_REWARD_NOTES.md).
        self.rewards.arm_swing_coupling = RewTerm(
            func=arm_swing_coupling,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot"), "gain": 0.5, "deadband": 0.3},
        )


@configclass
class PioneerHumanoidOnshapeCourseArmSwingEnvCfg_PLAY(PioneerHumanoidOnshapeCourseArmSwingEnvCfg):
    """Repeatable visual verification: 3 robots, const forward command with arm swing."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 3
        self.scene.env_spacing = 2.0
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = None
        self.episode_length_s = 60.0
        # Spawn exactly on the marker (same start the training uses), root at
        # floor level, facing +Y up the course.
        self.events.reset_base.params["pose_range"]["x"] = (_SPAWN_X, _SPAWN_X)
        self.events.reset_base.params["pose_range"]["y"] = (_SPAWN_Y, _SPAWN_Y)
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (_YAW_90, _YAW_90)

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
        self.commands.base_velocity.ranges.heading = (_YAW_90, _YAW_90)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        # Shoving stays ON in the visual config: the point of the clip is to
        # watch the policy catch itself. Same calibration as training, so the
        # clip shows what the policy was actually trained against.
        from ...mdp import shove as shove_mdp
        from isaaclab.managers import EventTermCfg as EventTerm
        self.events.push_robot = EventTerm(
            func=shove_mdp.push_robot_logged,
            mode="interval",
            interval_range_s=_PUSH_INTERVAL_S,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": _PUSH_VELOCITY,
            },
        )

        # Generator-only edge termination rejects USD terrain; the course
        # bounds term below covers it.
        self.terminations.terrain_out_of_bounds = None
        # Training guardrail: any ray miss resets. Disabled for the visual so
        # walking is verifiable even with partial scan misses at mesh edges.
        self.terminations.invalid_height_scan = None
        self.terminations.course_out_of_bounds.params = {
            "lateral_limit": 12.0,
            "backward_limit": -8.0,
            "axis": 1,
        }
        # Past the summit and out to the flat exit ground (which ends at
        # y=27.61), so the visual shows the whole course walked rather than
        # stopping at the top of the last flight.
        self.terminations.course_complete.params["goal_distance"] = 27.0
        self.terminations.course_complete.params["axis"] = 1
