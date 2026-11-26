"""Spiral staircase task: the Onshape spiral as a path-relative course.

The straight course walks a sawtooth in a line; here the walking direction rotates (30 treads per turn,
1.7 turns, 6 m tall), so progress, facing, out-of-bounds and the swing-clearance gate are path-relative.
The path coordinate is height, the one progress measure that cannot be faked by circling. The rest of the
reward set, knee range, shoving and fall penalty come from ``onshape_env_cfg.py``.

Geometry (centre, walking radius, riser, angular step, clearance band) is read from
``assets/staircase_onshape/spiral_geometry.json``, extracted from the merged USD by
``tools/onshape_course/spiral_profile.py``.
"""

from __future__ import annotations

import json
import math

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ...mdp import spiral_course as spiral_mdp
from ...mdp.stair_course import foot_flat_l2
from .rough_env_cfg import FOOT_BODIES
from ...mdp.spiral_command import SpiralVelocityCommandCfg
from .onshape_env_cfg import PioneerHumanoidOnshapeCourseEnvCfg

_SPIRAL_USD = "assets/staircase_onshape/spiral_terrain.usd"
# spawn nudge away from the marker (metres, world frame)
_SPAWN_OFFSET_X = 0.25
_SPAWN_OFFSET_Y = 0.0
_SPIRAL_GEOMETRY = "assets/staircase_onshape/spiral_geometry.json"

# spawn on the marker facing +Y; the heading command then turns the robot onto the staircase's tangent
_YAW_90 = math.pi / 2


def _load_geometry() -> dict:
    """The staircase's own shape, extracted from its mesh."""
    with open(_SPIRAL_GEOMETRY) as handle:
        return json.load(handle)


@configclass
class PioneerHumanoidOnshapeSpiralEnvCfg(PioneerHumanoidOnshapeCourseEnvCfg):
    """Spiral staircase: climb 1.7 turns of a helicoid, following the path.

    Warm-started from the sawtooth-course policy. The reward set is the same
    shape as the straight course's, but every course term is measured along
    the path: the progress and retreat terms read the *height* (the path
    coordinate), the facing term reads the *tangent*, and the out-of-bounds
    term reads the *radius*.
    """

    def __post_init__(self):
        super().__post_init__()

        geometry = _load_geometry()
        centre = tuple(geometry["centre"])
        radius = float(geometry["walking_radius"])
        riser = float(geometry["riser"])
        step = float(geometry["angular_step"])
        z_top = float(geometry["z_top"])
        outer = float(geometry["tread_outer_radius"])
        marker = tuple(float(v) for v in geometry["marker"])
        # tread depth at the walking line sets the slope, which converts height rate into speed along the path
        tread = step * radius
        slope = riser / math.sqrt(riser * riser + tread * tread)
        # walk where the planks are widest (the spawn's radius)
        walk_radius = radius
        # clearance band = riser + margin (a bare-riser band left the swing knee nothing to reach for)
        band = riser + 0.06
        self.spiral_geometry = geometry
        self.spiral_slope = slope
        self.spiral_band = band

        # terrain
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.usd_path = _SPIRAL_USD
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]
        # scanner rays start at head level: 20 m up would hit the turn above
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.5))
        self.scene.env_spacing = 0.0

        # spawn on the marker (from the geometry file), facing +Y, with a small scatter
        spawn_x = marker[0] + _SPAWN_OFFSET_X
        spawn_y = marker[1] + _SPAWN_OFFSET_Y
        # tight scatter: the marker is only 0.18 x 0.14 m, a wider box puts the robot under a tread
        self.events.reset_base.params["pose_range"]["x"] = (spawn_x - 0.08, spawn_x + 0.08)
        self.events.reset_base.params["pose_range"]["y"] = (spawn_y - 0.08, spawn_y + 0.08)
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (_YAW_90, _YAW_90)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

        # command: the heading target follows the staircase
        self.commands.base_velocity = SpiralVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            # sharper heading servo: the spawn faces +Y but the tangent there is 35 deg
            heading_control_stiffness=1.0,
            debug_vis=False,
            centre=centre,
            # a hair inside the marker's radius: much more margin on the inside than the outer edge
            walking_radius=walk_radius,
            ranges=UniformVelocityCommandCfg.Ranges(
                # 0.22-0.30 m/s is the stable regime; faster strides were clearly worse (STAIR_REWARD_NOTES.md).
                lin_vel_x=(0.22, 0.30),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(0.0, 0.0),
            ),
        )

        # rewards: world-axis straight-course terms are zeroed and replaced by path-relative ones
        self.rewards.course_forward_progress.weight = 0.0
        self.rewards.course_retreat_cost.weight = 0.0
        self.rewards.face_course_direction.weight = 0.0
        # the scan sees the staircase rising, so this charges ~-13/episode; height gain covers its job
        self.rewards.base_height_l2.weight = 0.0
        # Scan-free stand-tall term: a crouched robot cannot clear the risers (STAIR_REWARD_NOTES.md).
        self.rewards.stance_height_l2 = RewTerm(
            func=spiral_mdp.stance_height_l2,
            weight=-5.0,
            params={
                "feet_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "target_height": 0.84,
            },
        )
        # Off: this term vetoes turning (a humanoid turns by differential hip roll).
        self.rewards.hip_yaw_splay.weight = 0.0
        # Eased to -1.0 (straight: -6.0): the step-up needs some abduction (STAIR_REWARD_NOTES.md).
        self.rewards.hip_abduction_splay.weight = -1.0
        # Fall penalty and riser bonus are a pair: -800 with a +40/riser bonus (STAIR_REWARD_NOTES.md).
        self.rewards.termination_penalty.weight = -800.0
        self.rewards.spiral_progress = RewTerm(
            func=spiral_mdp.spiral_progress,
            # same weight as the straight course
            weight=5.0,
            params={"tau": 0.7, "slope_factor": slope, "cap": 0.5},
        )
        # Dense progress along the tangent: pays the approach and the curve.
        self.rewards.spiral_path_progress = RewTerm(
            func=spiral_mdp.spiral_path_progress,
            # 2.0, not 5.0: at 5.0 it outweighed clearance shaping and the gait collapsed.
            weight=2.0,
            params={"centre": centre, "cap": 0.5},
        )
        # Discrete bonus per cleared riser: immediate and not farmable by walking (STAIR_REWARD_NOTES.md).
        self.rewards.spiral_riser_bonus = RewTerm(
            func=spiral_mdp.spiral_riser_bonus,
            # +40 per riser; must outweigh a failed attempt's cost, or PPO never tries (STAIR_REWARD_NOTES.md).
            weight=2000.0,
            params={"riser": riser, "standing_height": 0.845},
        )
        self.rewards.spiral_retreat_cost = RewTerm(
            func=spiral_mdp.spiral_retreat_cost,
            weight=-2.0,
            params={"tau": 0.7, "slope_factor": slope},
        )
        self.rewards.spiral_face_tangent = RewTerm(
            func=spiral_mdp.spiral_face_tangent,
            # Light: the heading command already steers.
            weight=-0.25,
            params={"centre": centre, "deadband": 0.25},
        )
        # Keep the robot on the walking line: the treads narrow towards the
        # axis and the outer edge is only 0.14 m past the line.
        self.rewards.spiral_radial_error = RewTerm(
            func=spiral_mdp.spiral_radial_error,
            # Sized against the progress income; -25 stopped the climb, -2.0 let it drift (STAIR_REWARD_NOTES.md).
            weight=-5.0,
            params={"centre": centre, "walking_radius": walk_radius, "deadband": 0.10},
        )
        # Swing clearance along the path with a scalar band (the real riser height),
        # since a per-position table would pay almost nothing on the flat approach.
        # Rationale for the weight and excess coefficient: STAIR_REWARD_NOTES.md.
        self.rewards.foot_swing_clearance.params.update({
            "forward_axis": 1,
            "max_clearance": 0.45,
            "band": band,
            "path_centre": centre,
            "excess_coef": 30.0,
            "back_allow": -0.02,
            "phase_ramp": 0.06,
        })
        self.rewards.foot_backswing_cost.params["path_centre"] = centre
        
        # landing-window term below replaces the inherited whole-stance foot_flat
        self.rewards.foot_flat.weight = 0.0

        # flat landing within the first 0.15 s of contact; deadband 0.23 rad leaves the good foot's worst case free
        self.rewards.foot_flat.weight = 0.0

        self.rewards.foot_flat_landing = RewTerm(
            func=foot_flat_l2,
            weight=-0.5,
            params={
                "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l_foot"]),
                "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["r_foot"]),
                "feet_cfg": SceneEntityCfg("robot", body_names=["l_foot", "r_foot"]),
                "local_up": (0.0, 0.0, 1.0),
                "deadband": 0.23,
                "landing_window": 0.15,
            },
        )

        # same skate pricing as the straight course
        self.rewards.feet_slide.weight = -0.5

        # terminations: goal = top of the staircase plus slack
        self.terminations.course_complete = DoneTerm(
            func=spiral_mdp.spiral_complete,
            params={
                "goal_height": z_top + 0.75,
                "centre": centre,
                "radius_min": 0.3,
                "radius_max": outer,
                "tilt_limit": 0.8,
            },
            time_out=True,
        )
        self.terminations.course_out_of_bounds = DoneTerm(
            func=spiral_mdp.spiral_out_of_bounds,
            params={
                "centre": centre,
                # A guidance error is not a fall: keep this band wide (STAIR_REWARD_NOTES.md).
                "radius_min": 0.05,
                "radius_max": outer + 0.3,
                "min_height": 0.3,
                "max_height": z_top + 1.5,
            },
        )
        self.terminations.course_no_progress = DoneTerm(
            func=spiral_mdp.spiral_no_progress,
            # Only has to catch a feet-glued stander; a stricter bar diverged (STAIR_REWARD_NOTES.md).
            params={"min_gain": 0.25, "min_time": 60.0, "spawn_height": 0.84},
        )
        # the climb takes ~50 s at the commanded speed; leave room to finish and retry
        self.episode_length_s = 240.0
