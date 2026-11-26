# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Onshape straight staircase course: the STEP-built stairs as trained terrain.

The terrain is the merged single-mesh USD (``straight_stairs_terrain.usd``: six stepped pyramids with
a built-in difficulty ramp, peak 1.54 m, 5 cm to 16 cm risers, 0.28 m treads, plus the model's own
ground). The RayCaster height scanner reads that one mesh. The STEP export is not committed; the
terrain is rebuilt from it with course tooling that is not part of this tree.

The robot spawns on the model's ``marker`` sliver facing +Y (yaw pi/2), so the local +X velocity
command drives it up the course; course rewards and terminations take axis=1 / target=pi/2.
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
from ...mdp import push_by_setting_velocity
from ...mdp import stair_course as course_mdp
from ...mdp import terminations as mdp
from .rough_env_cfg import (
    FOOT_BODIES,
    LEG_JOINTS,
    PioneerHumanoidRoughNoStairsSelectiveKneeShapeRewards,
    PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg,
)

from pioneer_humanoid.whole_body import WHOLE_BODY_WALKER_CFG

# Body names for the whole-body walker's leg links.
ONSHAPE_FOOT_BODIES = FOOT_BODIES


_ONSHAPE_COURSE_USD = "assets/staircase_onshape/straight_stairs_terrain.usd"
_ONSHAPE_COURSE_PROFILE = "assets/staircase_onshape/straight_stairs_profile.json"


def _load_course_band() -> list:
    """The swing clearance each position needs, from the course profile JSON ([[y, band], ...])."""
    try:
        with open(_ONSHAPE_COURSE_PROFILE) as handle:
            profile = json.load(handle)
        if "samples" in profile:
            # rows are [terrain height, band]; y = y0 + index * bin
            y0 = float(profile["y0"])
            bin_size = float(profile["bin"])
            return [
                [y0 + index * bin_size, float(row[1])]
                for index, row in enumerate(profile["samples"])
            ]
        # legacy layout: explicit [y, band] rows
        return [[float(entry[0]), float(entry[1])] for entry in profile["band"]]
    except (OSError, KeyError, ValueError) as err:  # pragma: no cover - container path
        print(f"[ONSHAPE] course profile unavailable ({err}); falling back to the scalar band")
        return []


@configclass
class PioneerHumanoidOnshapeCourseRewards(PioneerHumanoidRoughNoStairsSelectiveKneeShapeRewards):
    """Onshape straight-course specific reward terms."""

    course_forward_progress = RewTerm(
        func=course_mdp.course_forward_progress,
        weight=5.0,
        params={"axis": 1, "cap": 0.5},
    )
    course_retreat_cost = RewTerm(
        func=course_mdp.course_retreat_cost,
        weight=-2.0,
        params={"axis": 1},
    )
    face_course_direction = RewTerm(
        func=course_mdp.face_course_direction,
        weight=-0.5,
        params={"target": 1.57079632679, "deadband": 0.25},
    )
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

# Marker centre (metres); the course runs +Y.
_SPAWN_X = 4.163
_SPAWN_Y = -1.549
_YAW_90 = math.pi / 2

# Random shoving (see STAIR_REWARD_NOTES.md).
_PUSH_INTERVAL_S = (8.0, 12.0)
_PUSH_VELOCITY = {"x": (-0.5, 0.5), "y": (-0.3, 0.3)}


@configclass
class PioneerHumanoidOnshapeCourseEnvCfg(PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg):
    """Onshape course training template (the rough-no-stairs reward recipe plus course terms).

    All envs share origin ~0 (spacing 0); scatter comes from the pose ranges.
    """

    rewards = PioneerHumanoidOnshapeCourseRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = WHOLE_BODY_WALKER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.usd_path = _ONSHAPE_COURSE_USD
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]
        self.scene.env_spacing = 0.0

        # range noise + per-beam bias + dropout, ramped in (mdp/noise.py); replaces the shared uniform noise
        self.observations.policy.height_scan.noise = LidarNoiseCfg()

        self.terminations.course_complete = DoneTerm(
            func=course_mdp.course_complete,
            params={"goal_distance": 12.0, "lateral_limit": 2.0, "minimum_height": 0.65, "tilt_limit": 0.8},
            time_out=True,
        )
        self.terminations.course_out_of_bounds = DoneTerm(
            func=course_mdp.course_out_of_bounds,
            params={"lateral_limit": 2.0, "backward_limit": -0.7},
        )
        # stair climbing holds a sustained pitch; 0.8 rad would terminate it
        self.terminations.base_tilt = DoneTerm(
            func=mdp.base_tilt_over_limit, params={"limit": 1.2}
        )
        self.curriculum.terrain_levels = CurrTerm(
            func=course_mdp.complete_stair_course_levels,
            params={"goal_distance": 12.0, "lateral_limit": 2.0, "minimum_height": 0.65, "tilt_limit": 0.8},
        )

        command = self.commands.base_velocity
        command.heading_command = True
        command.rel_heading_envs = 1.0
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.5, 0.5)
        command.ranges.heading = (0.0, 0.0)

        # spawn: +/-0.5 m around the marker, facing +Y, at rest
        self.events.reset_base.params["pose_range"]["x"] = (3.66, 4.66)
        self.events.reset_base.params["pose_range"]["y"] = (-2.05, -1.05)
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (_YAW_90, _YAW_90)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.5)
        self.commands.base_velocity.ranges.heading = (_YAW_90, _YAW_90)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # knee clip widened to the hardware limit so tall risers are reachable
        self.actions.joint_pos.clip = {"Knee_.*": (-1.10, -0.05)}
        # soft landings
        self.rewards.feet_impact = RewTerm(
            func=course_mdp.feet_impact, weight=-0.5,
            params={
                "left_sensor_cfg": SceneEntityCfg("left_foot_terrain_contact"),
                "right_sensor_cfg": SceneEntityCfg("right_foot_terrain_contact"),
                "feet_cfg": SceneEntityCfg("robot", body_names=ONSHAPE_FOOT_BODIES, preserve_order=True),
            },
        )

        # land flat (a smaller weight is inert against the progress income)
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

        # price skating
        self.rewards.feet_slide.weight = -0.5

        # fixed course: no curriculum, no generator-only edge termination
        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None
        # goal just past the final descent
        self.terminations.course_complete.params["goal_distance"] = 20.5
        self.terminations.course_complete.params["lateral_limit"] = 6.0
        self.terminations.course_complete.params["axis"] = 1
        self.terminations.course_out_of_bounds.params = {
            "lateral_limit": 8.0,
            "backward_limit": -8.0,
            "axis": 1,
        }
        self.rewards.face_course_direction.params["target"] = _YAW_90
        # hip-abduction splay beyond a deadband
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
        # knee tuck
        self.rewards.knee_swing_flexion.params["minimum_flexion"] = 0.28
        self.rewards.knee_swing_flexion.weight = -4.0
        # feet stay down
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.2
        # trim the stand-still yaw-rate income and the action-rate tax
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.action_rate_l2.weight = -0.002
        # warm-start setting; a fresh policy should use a smaller penalty
        self.rewards.termination_penalty.weight = -800.0
        # torso tilt
        self.rewards.flat_orientation_l2.weight = -3.0
        # smooth legs
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
        # freezing backstop: reset if not 1.5 m along the course after 25 s
        self.terminations.course_no_progress = DoneTerm(
            func=course_mdp.course_no_progress,
            params={"axis": 1, "min_displacement": 1.5, "min_time": 25.0,
                    "spawn": (_SPAWN_X, _SPAWN_Y)},
        )
        # random shoving
        self.events.push_robot = EventTerm(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=_PUSH_INTERVAL_S,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": _PUSH_VELOCITY,
            },
        )
        self.episode_length_s = 1200.0
