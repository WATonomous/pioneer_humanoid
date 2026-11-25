"""Spiral staircase task: the Onshape spiral as a path-relative course.

The straight Onshape course (see ``onshape_env_cfg.py``) is a sawtooth of
pyramids that the robot walks up in a straight line. This task is the same
robot on the user's *spiral* staircase, which changes three things:

  * the walking direction rotates continuously (15.65 treads per turn), so
    progress, facing, out-of-bounds and the swing-clearance gate all have to
    be expressed relative to the path instead of a world axis;
  * the path coordinate is the height (the staircase is a helicoid), which is
    the only progress measure that cannot be faked by circling;
  * the staircase is 7 m tall and 3.2 turns long, so the gait has to hold up
    for a lot longer than a straight flight.

Everything else -- the tuned reward set, the knee range, the light shoving,
the fall penalty -- is inherited from the Onshape course recipe, so the
policy that walks the sawtooth is the starting point for this one.

The geometry (centre, walking radius, riser, angular step, the clearance band
table) is read from ``assets/staircase_onshape/spiral_geometry.json``, which the
course build tooling (not in this tree) extracts from the merged USD.
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
# A nudge of the spawn away from the marker (metres, world frame). The marker
# is the user's own placement in Onshape, so this is zero by default.
_SPAWN_OFFSET_X = 0.25
_SPAWN_OFFSET_Y = 0.0
_SPIRAL_GEOMETRY = "assets/staircase_onshape/spiral_geometry.json"

# The spawn: the user's marker sliver, on the flat floor at the foot of the
# staircase. The robot faces +Y (the world's +Y) as the task requires; the
# heading command then turns it onto the staircase's tangent.
_YAW_90 = math.pi / 2


def _load_geometry() -> dict:
    """The staircase's own shape, extracted from its mesh."""
    with open(_SPIRAL_GEOMETRY) as handle:
        return json.load(handle)


@configclass
class PioneerHumanoidOnshapeSpiralEnvCfg(PioneerHumanoidOnshapeCourseEnvCfg):
    """Spiral staircase: climb 3.2 turns of a helicoid, following the path.

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
        # The arc the robot walks over per tread: the tread depth at the
        # walking line. It sets the staircase's slope, and hence the factor
        # that converts the height rate into a speed along the path.
        tread = step * radius
        slope = riser / math.sqrt(riser * riser + tread * tread)
        # The treads are thin radial planks, so their tangential width grows
        # with the radius; the walking line is the spawn's radius, where the
        # planks are widest. (The tread spacing at that radius is what the
        # stride has to match -- see the commanded speed below.)
        walk_radius = radius
        # The clearance band is the riser plus a margin. The straight course's
        # climb band measured 0.16-0.21 m, so 0.18 m here keeps the same
        # "lift the foot properly" incentive: with the band at the bare riser
        # (0.15 m) the swing knee had nothing to reach for.
        band = riser + 0.06
        self.spiral_geometry = geometry
        self.spiral_slope = slope
        self.spiral_band = band

        # --- the terrain -----------------------------------------------------
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.usd_path = _SPIRAL_USD
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]
        # The scanner's rays start at head level (~1.5 m above base), not 20 m up.
        # On a spiral, 20 m hits the ceiling (turns above); 1.5 m clears the head
        # and sits below the turn above.
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.5))
        self.scene.env_spacing = 0.0

        # --- the spawn ------------------------------------------------------
        # On the marker (from the geometry file), facing +Y, with a small scatter;
        # the heading command turns the robot onto the tangent during the approach.
        spawn_x = marker[0] + _SPAWN_OFFSET_X
        spawn_y = marker[1] + _SPAWN_OFFSET_Y
        # A tight scatter: the marker sliver is only 0.18 x 0.14 m, so the
        # straight course's 0.5 m box put the robot at the marker's *corners* --
        # and on this staircase that means under a tread.
        self.events.reset_base.params["pose_range"]["x"] = (spawn_x - 0.08, spawn_x + 0.08)
        self.events.reset_base.params["pose_range"]["y"] = (spawn_y - 0.08, spawn_y + 0.08)
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (_YAW_90, _YAW_90)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

        # --- the command: the heading target follows the staircase ---------
        self.commands.base_velocity = SpiralVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            # A sharper servo than the shared recipe's 0.5: the spawn faces
            # +Y while the tangent at the spawn is 35 deg, and a gentle
            # turn-in curves the robot into the middle of the staircase.
            heading_control_stiffness=1.0,
            debug_vis=False,
            centre=centre,
            # A hair inside the marker's radius: the treads end 0.14 m past
            # the walking line but 0.5 m before the middle, so the margin is
            # much better on the inside.
            walking_radius=walk_radius,
            ranges=UniformVelocityCommandCfg.Ranges(
                # 0.22-0.30 m/s is the stable regime; faster strides were clearly worse (STAIR_REWARD_NOTES.md).
                lin_vel_x=(0.22, 0.30),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(0.0, 0.0),
            ),
        )

        # --- the rewards ----------------------------------------------------
        # World-axis straight-course terms are zeroed and replaced by path-relative ones.
        self.rewards.course_forward_progress.weight = 0.0
        self.rewards.course_retreat_cost.weight = 0.0
        self.rewards.face_course_direction.weight = 0.0
        # The scan sees the staircase rising around the robot, so this term charges a
        # systematic -13/episode here; the height-gain progress term covers its job.
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
            # The straight course's weight: the whole reward set here is the
            # straight course's, so the climb is paid exactly as the
            # straight course paid its progress.
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
            # The target is the command's radius (a hair inside the marker):
            # the treads end 0.14 m past the walking line but 0.5 m before
            # the middle, so the margin is much better on the inside.
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
        
        # This course uses the landing-window term below instead of the
        # straight course's whole-stance foot_flat; zero the inherited copy.
        self.rewards.foot_flat.weight = 0.0

        # Flat foot landing: encourage the foot to land flat (not tilted) within
        # the first 0.15s of contact. The deadband is 0.23 rad (13.2 deg) -
        # calibrated to leave the good foot's worst case (11.7 deg) free.
        # The straight-course base now enables its own foot_flat at -0.5; this
        # task uses foot_flat_landing instead, so zero the inherited copy to
        # avoid pricing the same landing tilt twice.
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

        # Same skate pricing as the straight course (see its comment): the
        # spiral inherits the -0.1 weight otherwise.
        self.rewards.feet_slide.weight = -0.5

        # --- the terminations ----------------------------------------------
        # The goal is the top of the staircase: the root stands 0.84 m above
        # the top tread, so the goal height is that, with a little slack.
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
        # The staircase is 7 m tall and the climb takes ~50 s at the commanded
        # speed, so the episode needs room to finish (and to retry).
        self.episode_length_s = 240.0
