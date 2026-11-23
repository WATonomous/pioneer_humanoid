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
and rebuild the terrain with:

    isaaclab.sh -p tools/onshape_course/build_course.py \\
        --step assets/staircase_onshape/straight_stairs.step

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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ...mdp import push_by_setting_velocity
from ...mdp import shove as shove_mdp
from ...mdp import stair_course as course_mdp
from ...mdp.terrain_contact import (
    TerrainContactSensor,
    feet_air_time_positive_biped_terrain,
    feet_slide_terrain,
)
from .rough_env_cfg import (
    LEG_JOINTS,
    PioneerHumanoidRoughNoStairsSelectiveKneeShapeRewards,
    PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg,
)

from pioneer_humanoid.leg_only import LEG_ONLY_HUMANOID_CFG

# Body names for the new Pioneer leg-only model (different from old_leg_only)
ONSHAPE_FOOT_BODIES = ["l_foot", "r_foot"]


_ONSHAPE_COURSE_USD = "assets/staircase_onshape/straight_stairs_terrain.usd"
_ONSHAPE_COURSE_PROFILE = "assets/staircase_onshape/straight_stairs_profile.json"


def _load_course_band() -> list:
    """The course's own shape: the swing clearance each position actually needs.

    Written by tools/onshape_course/course_profile.py from the merged USD
    (see that script for how the table is derived). A single global band has
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
    # Flat foot landing (disabled - model-fixed)
    foot_flat = RewTerm(
        func=course_mdp.foot_flat_l2,
        weight=0.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("left_foot_terrain_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_terrain_contact"),
            "feet_cfg": SceneEntityCfg("robot", body_names=ONSHAPE_FOOT_BODIES, preserve_order=True),
            "deadband": 0.23,
            "landing_window": 0.15,
        },
    )
# Course runs +Y as modeled (forward is +Y). Spawn directly on the marker
# sliver (metres, from the marker part's bounds: centre 4.1628, -1.5486),
# facing +Y; course rewards/terminations take axis=1 / target=pi/2 below.
_SPAWN_X = 4.163
_SPAWN_Y = -1.549
_YAW_90 = math.pi / 2

# Random shoving (robustness domain randomisation). The root's velocity is
# perturbed by a random amount every 8-12 s -- the "someone bumps you" test.
# The lateral axis is the interesting one (a biped has to catch itself without
# stepping wide and re-introducing the splay this task fought to remove); the
# fore/aft one is kept smaller so a bump does not simply knock the robot back
# down the flight. +/-0.9 m/s sideways is ~2x the 0.4 m/s walking speed: a
# shove that is plainly visible in a recording, and one a person would call a
# stumble. Training and the visual config share these numbers, so the clip
# shows exactly what the policy was trained against.
# +/-0.5 m/s sideways is the LIGHT shove, and it is the one the shipping
# policy is trained against (run 2026-10-20_18-59-09: 83.5% completion with
# shoving on, 11.8% falls). The +/-0.9 variant was tried (2026-10-20_19-52-05)
# and the policy got measurably worse (62.6% completion, 27.5% falls, action
# noise std 3.7 -> 4.9), so the light shove is the one that stayed.
_PUSH_INTERVAL_S = (8.0, 12.0)
_PUSH_VELOCITY = {"x": (-0.5, 0.5), "y": (-0.3, 0.3)}


@configclass
class PioneerHumanoidOnshapeCourseEnvCfg(PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg):
    """Onshape course training template (same 19-reward recipe as rough-no-stairs).

    The course runs +Y as modeled, so the spawn faces +Y at yaw pi/2 and every
    course reward / termination takes axis=1 (or target=pi/2). All envs share
    origin ~0 (spacing 0); scatter comes from the pose ranges.
    """

    # Create rewards instance with overridden feet_crossing_penalty for pioneer_leg_only
    rewards = PioneerHumanoidOnshapeCourseRewards()
    rewards.feet_crossing_penalty = RewTerm(
        func=course_mdp.feet_crossing_l2,
        weight=-40.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ONSHAPE_FOOT_BODIES, preserve_order=True),
            "margin": 0.08,
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # Use the new Pioneer leg-only model (not the legacy old_leg_only)
        self.scene.robot = LEG_ONLY_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.usd_path = _ONSHAPE_COURSE_USD
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]
        self.scene.env_spacing = 0.0

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

        # Override contact sensors for pioneer_leg_only model (uses l_foot/r_foot)
        terrain_filter = f"{self.scene.terrain.prim_path}/terrain/mesh"
        for side, foot in (("left", "l_foot"), ("right", "r_foot")):
            setattr(
                self.scene,
                f"{side}_foot_terrain_contact",
                ContactSensorCfg(
                    class_type=TerrainContactSensor,
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot}",
                    filter_prim_paths_expr=[terrain_filter],
                    update_period=self.sim.dt,
                    history_length=3,
                    track_air_time=True,
                ),
            )
        # Update reward sensor params to use the new contact sensors
        sensor_params = {
            "left_sensor_cfg": SceneEntityCfg("left_foot_terrain_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_terrain_contact"),
        }
        self.rewards.feet_air_time.func = feet_air_time_positive_biped_terrain
        self.rewards.feet_air_time.params = {
            **sensor_params,
            "command_name": "base_velocity",
            "threshold": self.rewards.feet_air_time.params["threshold"],
        }
        self.rewards.feet_slide.func = feet_slide_terrain
        self.rewards.feet_slide.params = {
            **sensor_params,
            "asset_cfg": SceneEntityCfg("robot", body_names=ONSHAPE_FOOT_BODIES, preserve_order=True),
        }
        # --- Knee range: give the stair task the robot's real knee travel -----
        # The shared recipe clips the knee *target* to -0.95 rad while the joint
        # itself allows -1.134 rad (65 deg). Forward kinematics on this URDF
        # (hip height 0.788 m, thigh 0.262 m, shank 0.394 m) gives a maximum
        # sagittal foot lift of ~0.155 m with a 0.95 rad tuck, but ~0.217 m at
        # the hardware limit -- and the risers measured from the course CAD are
        # 0.100-0.115 m on part 1 rising to ~0.21 m on parts 5-6. Under -0.95
        # the tall risers are outside the sagittal reach, which is why the
        # pre-fix walker cleared them by abducting the hip sideways -- exactly
        # the splay/asymmetry this lineage exists to remove. This loosens a
        # training clip toward the hardware limit; it adds no knee reward and
        # does not tighten flexion. Task-local: the rough/flat recipes keep
        # their -0.95 clip.
        self.actions.joint_pos.clip = {"Knee_.*": (-1.10, -0.05)}
        # ------------------------------------------------------------------
        # MODEL-FIX ERA (2026-10-19): the terms below were written to compensate
        # for asymmetries that turned out to be defects in the robot description,
        # now fixed in whole_body_humanoid.urdf:
        #   * the right foot landed ~9.5 deg toe-down every stride because the
        #     right ankle-pitch axis was not the mirror of the left, which gave
        #     the two ankles mirrored (not equal) physical ranges;
        #   * the right leg circumducted because the right hip-abduction axis was
        #     not the mirror of the left, so one command moved both legs the same
        #     physical way.
        # With the axes mirror-consistent and the spawn re-centred, the source of
        # the bias is gone. These terms had only ever been validated by
        # warm-starting an existing walker; a fresh lineage does not discover
        # walking under them (three from-scratch probes sat at reward ~-12 while
        # the 2026-10-17 reward set reached +24 by iteration 210). They are set
        # to weight 0 rather than deleted: the shapes and calibrations stay for a
        # warm-started clean-up pass once a symmetric gait exists.
        # ------------------------------------------------------------------
        # Soft landings: downward foot speed at contact pays quadratically.
        # Reverted from -6.0 to the 2026-10-17 value: the hard-landing report it
        # was chasing is the right-foot toe-down defect fixed in the URDF.
        # Measured (model 1999 on this course): the term was inert -- it logged
        # ~-0.5 per episode against a ~+6500 forward-progress income, so the
        # policy could slam a foot down for free. That is the gap behind the
        # "hard landing on the way down" report. 12x stronger, so a real
        # touchdown spike is priced against the walk it interrupts.
        self.rewards.feet_impact = RewTerm(
            func=course_mdp.feet_impact, weight=-0.5,
            params={
                "left_sensor_cfg": SceneEntityCfg("left_foot_terrain_contact"),
                "right_sensor_cfg": SceneEntityCfg("right_foot_terrain_contact"),
                "feet_cfg": SceneEntityCfg("robot", body_names=ONSHAPE_FOOT_BODIES, preserve_order=True),
            },
        )

        # Land flat, don't land on a corner. Measured on this course (model 1999,
        # foot-body orientation at all 36 touchdowns of a 20 s clip): the RIGHT
        # foot lands 18.8 deg off horizontal on average (53.9 deg max) while the
        # LEFT lands flat (2.9 deg mean, 11.7 deg max) -- the same one-sided
        # defect as the hip splay above, and nothing in the reward set priced foot
        # orientation at all.
        # Measured on the foot *body*, not the ankle joint: the left foot's ankle
        # pitch (-0.365 rad mean) is larger than the right's (+0.264) yet only the
        # left lands flat, and the right's landings split into a flat mode (1-5
        # deg, ankle pitched +0.5..+0.68) and a tilted mode (20-41 deg, ankle near
        # neutral). The policy already demonstrates the flat landing -- left every
        # stride, right half the time -- so this term only has to select for it.
        # The 0.15 s landing window is load-bearing: across the whole stance the
        # foot legitimately pitches to 41.7 deg (left) / 38.4 deg (right) on
        # average, up to 67 deg at toe-off, so 17/18 left and 18/18 right strides
        # would be taxed by any whole-contact deadband. Deadband calibrated to the
        # leg that already lands correctly (left worst case 11.7 deg), so a
        # correct landing stays free.
        # Weight is large compared with its siblings by necessity, not by taste:
        # the term is alive for ~7 frames per step over a small squared excess
        # (~2 raw per episode) against hip_roll_splay's ~39, so a -12 weight would
        # repeat the inert-term mistake (feet_impact at -0.5 bought nothing).
        # Watch Episode_Reward/foot_flat in the run log: it should land at a few
        # percent of the forward-progress income, not a rounding error.
        # MODEL-FIX ERA: reverted. The right foot's landing tilt was measured
        # against the broken ankle-pitch axis (mirrored physical ranges); with the
        # axis fixed both feet have the same range and the term's premise is gone.
        self.rewards.foot_flat = RewTerm(
            func=course_mdp.foot_flat_l2,
            weight=0.0,
            params={
                "left_sensor_cfg": SceneEntityCfg("left_foot_terrain_contact"),
                "right_sensor_cfg": SceneEntityCfg("right_foot_terrain_contact"),
                "feet_cfg": SceneEntityCfg("robot", body_names=ONSHAPE_FOOT_BODIES, preserve_order=True),
                "deadband": 0.23,
                "landing_window": 0.15,
            },
        )

        # Fixed course: no curriculum, no generator-only edge termination.
        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None
        # Goal at the *exit* of the final pyramid, on the flat ground where the
        # standing root still clears minimum_height. This course is a sawtooth --
        # the Onshape generator is literally "Stair Wave (Up-Down Progressive):
        # stairs that go up then back down in triangles" -- so the goal must sit
        # past the descent to keep the summit drop inside the training
        # distribution. At an earlier 22.6 the episode ended on the plateau
        # *before* that drop, leaving every term (feet_impact included) blind to
        # the very landing the visual eval exposes: the descent was out of
        # distribution, not mis-shapen.
        # Curriculum course: part 6 ends at y=19.3, so 20.5 is just past the last
        # descent (the old hard course ran to 25.16, hence the old 25.5). Keeping
        # it tight means a completed episode resets promptly instead of spending
        # ~6 m of flat walking after the stairs.
        self.terminations.course_complete.params["goal_distance"] = 20.5
        self.terminations.course_complete.params["lateral_limit"] = 6.0
        self.terminations.course_complete.params["axis"] = 1
        self.terminations.course_out_of_bounds.params = {
            "lateral_limit": 8.0,
            "backward_limit": -8.0,
            "axis": 1,
        }
        # --- Course rewards -------------------------------------------------
        # Forward progress, weighted so the capped max payout clears every
        # stationary term; the cap bounds the pace (no rushing).
        # Retreat is a real cost now: the term returns a non-negative retreat
        # magnitude, so its inherited negative weight actually charges it (the
        # old clamp_max made it silently *pay* for retreat).
        self.rewards.face_course_direction.params["target"] = _YAW_90
        # Knee-led swings, not circumduction. Measured on this course (model
        # 1999, stride diagnostics): the RIGHT leg cleared risers by abducting
        # the hip ~0.9 rad with a near-straight knee (|knee| ~ 0.05) and dragged
        # the foot ~0.5 m out to the side, while the LEFT leg tucked properly
        # (knee ~0.9 rad, hip ~0). The base `joint_deviation_hip` L1 (-0.1) is
        # far too weak to matter, so price the splay directly: excess |Hip_A|
        # beyond the deadband, squared.
        # Re-measured after the knee-tuck fix (82 strides, the visual eval that
        # reported "foot still flailing"): the right hip came down to +0.39 rad
        # but this term had gone inert -- deadband 0.45 sat *above* the residual
        # it was meant to charge, so the right foot still swung 0.64 m sideways
        # (left: 0.03 m). The deadband is now calibrated to the leg the policy
        # already swings correctly: the left hip sits at mean +0.26 rad, so 0.30
        # leaves a correct gait free while the right leg's splay pays.
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
        # The other half of the same defect: the right leg's lateral swing was
        # as much hip *roll* as abduction -- -0.38 rad on every stride against
        # the left leg's +0.05 -- and nothing in the reward set priced Hip_R at
        # all. Same shape and same calibration rule: the left leg's measured
        # roll defines the free band, so a one-sided splay pays while a correct
        # swing stays free. Weight is heavier than the abduction term because
        # the excess it charges is smaller (0.23 rad vs 0.09).
        # MODEL-FIX ERA: reverted. The right leg's roll was a consequence of the
        # broken hip-abduction axis, not a policy preference.
        # hip-abduction axis showing through, not a policy preference.
        # chasing the same one-sided splay; it costs ~0.003/step even when correct
        # and its own notes record that it regressed an existing gait.
        # Restore a real knee tuck (the inherited 0.35 was relaxed to 0.22 while
        # fixing the rear "butt-kick"). A bent swing knee is what clears a riser
        # *under* the body; with the tuck relaxed the right leg kept the knee
        # straight and swung out sideways instead (measured above). The rear lift
        # it was relaxed for is now separately bounded by foot_swing_clearance's
        # forward phase gate and foot_backswing_cost, so the tuck can come back --
        # with a weight that actually binds (base -1.5 left it a rounding error
        # next to the forward-progress income).
        # Trimmed after the 2026-10-20 visual eval ("raises knee too high"): the
        # term forces a tuck on EVERY swing, flat approach included, so 0.40 rad
        # plus the clearance peak produced a high-knee march. 0.28 still binds on
        # the flat without marching, and the risers get their tuck from the
        # clearance reward (which pays up to band 0.20 m) rather than from this
        # deficit term.
        self.rewards.knee_swing_flexion.params["minimum_flexion"] = 0.28
        self.rewards.knee_swing_flexion.weight = -4.0
        # Feet stay down: the inherited rough-no-stairs air-time term is weight 4.0 with a
        # 0.2 s threshold, and the visual eval flagged the result as "jumping"
        # (the policy buys air time). Cut to 1.0, NOT re-thresholded: the term is
        # paid as max(air - threshold, 0), so raising the threshold makes the
        # policy swing *longer* to keep the income. Tried 2.5/0.28 on
        # 2026-10-20_03-52-36 and measured exactly that -- air-time income fell
        # only 20% (policy compensated), hip flexion rose 0.04/1.28 -> 0.07/1.45
        # and landing tilt went 4.6/2.0 -> 15.0/7.2 deg. Weight alone is the
        # honest knob. The 2026-10-17 lesson stands: do NOT zero it or gate it on
        # speed -- that produced the feet-glued stander. 1.0 keeps a real swing
        # phase while making ground contact the cheap option.
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.2
        # Trim two taxes that dominated the landscape: a stand-still yaw-rate
        # tracking income and the action-rate tax (the single biggest cost).
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.action_rate_l2.weight = -0.002
        # Falls must hurt. Reward weights are per-second rates (x dt = 0.02),
        # so the rough-no-stairs recipe's -50 is only -1.0 per fall and -800 is -16.
        # -800 is a *warm-start* setting: a fresh random policy sees a -16.0
        # spike (the whole iteration-0 return) and its noise std runs away into
        # the flail-and-fall collapse, which is why the from-scratch lineage
        # used -200. Every run that actually climbed this course was warm
        # started and used -800 (train35 on course_terrain.usd: iteration 0
        # reward -16.04, course_complete 0.718). Since 2026-10-19_23-14-19
        # (model_599) is a warm-started walker, -800 is the correct regime: a
        # policy that stalls at a riser now loses as much as one that falls,
        # instead of stalling cheaply.
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
        # off-origin.
        self.terminations.course_no_progress = DoneTerm(
            func=course_mdp.course_no_progress,
            params={"axis": 1, "min_displacement": 2.0, "min_time": 15.0,
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
class PioneerHumanoidOnshapeCourseEnvCfg_PLAY(PioneerHumanoidOnshapeCourseEnvCfg):
    """Repeatable visual verification: 3 robots, const forward command."""

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
        self.commands.base_velocity.ranges.heading = (_YAW_90, _YAW_90)

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
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
