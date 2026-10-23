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
table) is read from ``assets/staircase_onshape/spiral_geometry.json``, which
``tools/onshape_course/spiral_profile.py`` extracts from the merged USD.
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

_SPIRAL_USD = "/workspace/humanoid/assets/staircase_onshape/spiral_terrain.usd"
# A nudge of the spawn away from the marker (metres, world frame). The marker
# is the user's own placement in Onshape, so this is zero by default.
_SPAWN_OFFSET_X = 0.25
_SPAWN_OFFSET_Y = 0.0
_SPIRAL_GEOMETRY = "/workspace/humanoid/assets/staircase_onshape/spiral_geometry.json"

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
        # On the marker (read from the geometry file, so it follows the
        # model), facing +Y, at rest. A small scatter keeps the envs from
        # being identical.
        #
        # The heading is the world's +Y, as the user specified. The
        # staircase's own tangent at the first tread is +Y too (the first
        # tread is at phi = 0, where the tangent is +Y), so a robot that
        # starts facing +Y is facing the right way *once it is on the
        # staircase*; the heading command turns it onto the tangent for the
        # approach.
        #
        # The spawn is the marker itself, which the user positions in Onshape
        # (the offsets are here for a nudge without a re-export).
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
                # 0.22-0.30 m/s, which is the configuration that made the
                # robot walk the spiral's base *stably* (falls 74% -> 6%,
                # out of bounds 0.1%, episode length 54 s, reward +13.9).
                # A faster stride was tried (0.45-0.55 m/s, so that the stride
                # matches the 0.61 m tread spacing) and it was clearly worse
                # (reward -81 at iteration 599, the path progress went
                # negative and the robot stopped moving).
                lin_vel_x=(0.22, 0.30),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(0.0, 0.0),
            ),
        )

        # --- the rewards ----------------------------------------------------
        # The straight course's terms are all measured along a world axis, so
        # they are switched off here and replaced by the path-relative ones.
        # The weights are kept identical to the straight course's, so the
        # reward landscape (and the balance against the -800 fall penalty)
        # carries over unchanged.
        self.rewards.course_forward_progress.weight = 0.0
        self.rewards.course_retreat_cost.weight = 0.0
        self.rewards.face_course_direction.weight = 0.0
        # The terrain-relative base height cannot be read on a spiral: the
        # height scan always sees the staircase *rising* around the robot (the
        # mean ray height was 1.6 m above the tread the robot stood on), so
        # the term charged a systematic -13 per episode. The height-gain
        # progress term does the same job here -- a crouched robot gains less
        # height per step, so it is paid less -- so the base-height term is
        # simply switched off for this task.
        self.rewards.base_height_l2.weight = 0.0
        # A scan-free stand-tall term in its place: the root's height above
        # the foot it stands on. The forward kinematics put the maximum
        # clearable riser at 0.182 m for a hip at 0.837 m but only 0.143 m
        # for a hip at 0.780 m, so a robot that crouches cannot clear this
        # staircase's 0.12 m risers at all -- and without the straight
        # course's base-height term (the scanner-based one is unusable here)
        # there was nothing holding the stance up.
        self.rewards.stance_height_l2 = RewTerm(
            func=spiral_mdp.stance_height_l2,
            weight=-5.0,
            params={
                "feet_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "target_height": 0.84,
            },
        )
        # The symmetric hip-roll deviation term is switched off for this task.
        #
        # It prices the two hip-roll joints differing from each other -- and a
        # humanoid *turns* by rolling them differentially (one in, one out).
        # On the straight course that term is harmless (nothing asks the robot
        # to turn), but here it is a mechanical veto on turning: with it
        # active the probe caught the robot turning the wrong way, the right
        # hip roll pinned at its limit (-0.524 rad), the angle around the
        # staircase stuck at -21 deg and the robot circling at r = 1.45 m
        # instead of the 1.16 m walking line -- it could not follow the
        # tangent at all.
        #
        # The *splay* this term was written for is a model defect that the
        # URDF fix removed, and the straight course keeps the term; the
        # absolute (both-legs-the-same-way) roll is still priced by
        # hip_abduction_splay and the ankle terms.
        self.rewards.hip_yaw_splay.weight = 0.0
        # The splay *is* priced (the lateral foot splay at -2.0 and the hip
        # abduction splay at -6.0). Leaving it unpriced was tried and it did
        # not reproduce the climb: with the splay priced, two separate runs
        # from the splayed donor put the riser bonus at 0.0146 and 0.0096; with
        # it unpriced the bonus stayed at zero. (A stage-2 curriculum -- learn
        # the turn with the splay, then price it out -- is the obvious next
        # step once a policy actually climbs.)
        #
        # The warm start for this task is the old *splayed* walker -- the only
        # policy in the repo that can already clear a 0.12 m riser, because it
        # walked the old hard course's 0.21 m risers by abducting the hip. It
        # is the only thing that gets the spiral's first step-up to happen at
        # all (the riser bonus goes to 0.0146 immediately, against 0.0000 for
        # the splay-free walker, which never attempts it).
        #
        # The splay is load-bearing for the step-up: pricing it out dropped the
        # bonus from 0.0146 to 0.0023, because the abducted stance is how that
        # walker clears a riser. So the splay is priced back in *after* the
        # policy has learned to turn and climb the spiral with it, and the
        # splay-free gait is then fine-tuned on top (the same two-stage
        # curriculum the straight course used).
        self.rewards.foot_lateral_splay.weight = -2.0
        # The hip abduction splay is eased to -1.0 (the straight course's
        # -6.0) for this task. It is the term that forbids the very hip
        # abduction the 0.12 m step-up needs, and the run log shows the
        # consequence exactly: with the splay priced, the policy clears
        # risers (the riser bonus reaches 0.0146), the abducted step-up is
        # unstable on the curve (83% of the episodes end in base_tilt), PPO
        # then removes the abduction because the falls cost -16 each, and the
        # bonus goes back to zero. Easing the price lets the policy keep
        # *some* abduction -- enough to clear the riser, not enough to be
        # unstable.
        self.rewards.hip_abduction_splay.weight = -1.0
        # The fall penalty is -200, not the straight course's -800.
        #
        # The chain of reasoning (each step measured):
        #
        # * The *lift* has to be worth learning before the *step-up* can be.
        #   A lift is paid by the swing-clearance term (+0.34 per swing) and
        #   is only risky because a lift on the staircase is followed by an
        #   attempt that might fall. At -800 a failure costs 16 per step, so
        #   lifting was never worth it: the probe caught the policy with
        #   straight legs (knee -0.01 rad), the right hip roll pinned at its
        #   limit and the clearance income at 0.0001 -- it had unlearned the
        #   gait the warm start brought.
        # * At -200 a lift is worth having (a fall costs 4 per step) and the
        #   riser bonus (+40 per riser) then makes the *step-up* worth
        #   attempting even at a 29% success rate.
        # * A cheap fall does invite flailing -- but only while something
        #   rewards raw motion. The tangent-speed term that used to do that
        #   is gone, so thrashing now earns nothing (the velocity tracking
        #   pays for the actual walking) and the good gait beats a flail
        #   outright: a 60 s walk earns +1.2 against a flail's -4.
        #
        # The two settings only work together: an expensive fall needs a
        # bonus that outweighs it, and a cheap fall needs the other terms to
        # keep the gait alive.
        self.rewards.termination_penalty.weight = -800.0
        self.rewards.spiral_progress = RewTerm(
            func=spiral_mdp.spiral_progress,
            # The straight course's weight: the whole reward set here is the
            # straight course's, so the climb is paid exactly as the
            # straight course paid its progress.
            weight=5.0,
            params={"tau": 0.7, "slope_factor": slope, "cap": 0.5},
        )
        # The DENSE progress term: the speed along the staircase's tangent --
        # the path-relative analogue of the straight course's
        # course_forward_progress, and the reason the policy keeps moving.
        #
        # It is the term that pays for the approach (the height-gain rate is
        # exactly zero for the whole flat approach, the robot walks ~0.9 m of
        # arc before the first tread) and for the curve. It is farmable in
        # principle -- a robot could circle the bottom of the staircase and
        # collect it without climbing -- and the guard against that is the
        # *no-progress* termination, which measures height gain and so ends an
        # episode that never climbs, and the riser bonus below, which is what
        # climbing is actually paid for.
        self.rewards.spiral_path_progress = RewTerm(
            func=spiral_mdp.spiral_path_progress,
            # 2.0, not 5.0: at 5.0 (0.026 per step) it was worth twice the
            # clearance shaping (0.014 per step) and the policy traded the gait
            # for raw speed -- the swing clearance income collapsed to zero and
            # the robot shuffled. The straight course's own ratio is velocity
            # tracking 0.02 against clearance 0.019, i.e. comparable.
            weight=2.0,
            params={"centre": centre, "cap": 0.5},
        )
        # A discrete bonus for every riser cleared.
        #
        # The height-gain rate is dense in *time* but not in *outcome*: a
        # step-up spreads its payoff over the filter's time constant, so a
        # sampled attempt looks like either -16 (the fall, a one-off) or
        # +0.9 (the filtered burst) and PPO cannot tell a success from a
        # failure. The bonus makes a cleared riser an unmistakable, immediate
        # +3 in the per-step units, next to a -16 failure, so the successes
        # in the batch stand out and the policy learns to try.
        #
        # It is also the right *kind* of density: it is per riser (the
        # staircase's own unit of progress) rather than per metre, so it
        # cannot be farmed by walking.
        self.rewards.spiral_riser_bonus = RewTerm(
            func=spiral_mdp.spiral_riser_bonus,
            # 2000, i.e. +40 per riser in the per-step units. The bonus has to
            # outweigh the *immediate* cost of a failed attempt, or PPO never
            # tries: a fall costs 16 per step, so a success has to be worth
            # more than that. At +3 (weight 150) a success earned less than a
            # failure cost and the policy kept its distance from the
            # staircase -- the riser bonus stayed at 0.0004 and the gait
            # collapsed to a shuffle, because a step-up that is never
            # attempted is never learned. At +40 a success is worth two and a
            # half failures, so attempting pays even at a 29% success rate.
            #
            # The fall penalty is -800 (the straight course's) for the same
            # reason the other way round: at -200 the policy stopped caring
            # about falling at all and flailed (96% of the episodes ended in
            # base_tilt within a second) and the gait never came back. The
            # two settings only work together: an expensive fall needs a
            # bonus that outweighs it, and a cheap fall destroys the gait.
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
            # Light: the heading command already steers the robot along the
            # tangent, so this is only a nudge. At the straight course's -0.5
            # with a 0.15 rad deadband it cost 0.06 per step -- more than
            # twice the progress income -- and the policy spent its reward
            # budget on the heading and ignored the climb.
            weight=-0.25,
            params={"centre": centre, "deadband": 0.25},
        )
        # Keep the robot on the walking line: the treads narrow towards the
        # axis and the outer edge is only 0.14 m past the line.
        self.rewards.spiral_radial_error = RewTerm(
            func=spiral_mdp.spiral_radial_error,
            # Sized against the progress income (0.026 per step at the
            # commanded speed): 0.2 m off the line costs 0.02 per step, and
            # 0.4 m off costs 0.18 -- seven times the income. At -25 the
            # term cost forty times the income and the policy never climbed.
            # -5.0: with the tangent-speed term gone, this is the only thing
            # holding the robot on the walking line, and at -2.0 the robot
            # circled the staircase at whatever radius it liked (60% of the
            # episodes ended out of bounds).
            weight=-5.0,
            # The target is the command's radius (a hair inside the marker):
            # the treads end 0.14 m past the walking line but 0.5 m before
            # the middle, so the margin is much better on the inside.
            params={"centre": centre, "walking_radius": walk_radius, "deadband": 0.10},
        )
        # The swing clearance is measured along the path, and the band is the
        # riser the staircase actually has: 0.14 m of foot-to-foot clearance
        # plus the term's own 0.03 m margin.
        #
        # It is a *scalar* band, not the per-position table the straight
        # course uses. A per-position table would be indexed by the height
        # under the stance foot, and on the flat approach -- the 0.84 m of
        # floor before the first tread -- that height is 0, so the table
        # would pay only 0.06 m of clearance there. That is exactly where the
        # policy needs to *prepare* to step up 0.14 m, and it arrived at the
        # first riser with no reason to have lifted its foot. Every riser on
        # this staircase is the same height, so one band is the right answer
        # everywhere; the excess term (30) keeps the lift at the band rather
        # than above it.
        # The clearance weight is 20 rather than the straight course's 4.0:
        # this is the term that pays for lifting the foot, and without it the
        # policy discovered that it could track its commanded velocity
        # perfectly by shuffling with straight legs (the run log showed
        # 0.074 per step of tracking income against 0.0006 of clearance
        # income, and the swing knees had collapsed from 0.97 rad to 0.13).
        # A tucked swing earns five times the clearance, which is what has to
        # beat the risk of a fall.

        # The swing clearance is the term that pays for *lifting the foot*, and
        # on this task the lift has to be worth much more than it is on the
        # straight course, because a lift on the staircase is followed by an
        # attempt that might fall: at the straight course's weight the probe
        # caught the policy with straight legs (knee -0.01 rad), the right hip
        # roll pinned at its limit and the clearance income at 0.0001 -- it had
        # unlearned the gait the warm start brought, because lifting was never
        # worth the risk. At 20 the lift earns 1.8 per swing, and the riser
        # bonus below (worth 40 per riser) is what makes the step-up itself
        # worth attempting.
        self.rewards.foot_swing_clearance.weight = 40.0
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
        
        # Flat foot landing: encourage the foot to land flat (not tilted) within
        # the first 0.15s of contact. The deadband is 0.23 rad (13.2 deg) -
        # calibrated to leave the good foot's worst case (11.7 deg) free.
        self.rewards.foot_flat_landing = RewTerm(
            func=foot_flat_l2,
            weight=-0.5,
            params={
                "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_L"]),
                "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_R"]),
                "feet_cfg": SceneEntityCfg("robot", body_names=["Foot_L", "Foot_R"]),
                "local_up": (0.0, 0.0, 1.0),
                "deadband": 0.23,
                "landing_window": 0.15,
            },
        )

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
                # A *guidance* error is not a fall. The treads are an annulus
                # barely a metre wide, and a shove (the light shove is +/-0.5
                # m/s of lateral velocity) slides the robot right off the
                # walking line -- terminating that with the -800 fall penalty
                # made 68% of the episodes end in "out of bounds" and the
                # robot never got to climb. The band here is only for a
                # genuine departure (onto the floor, or off the top); the
                # radial reward does the steering and the base-tilt
                # termination catches the actual falls.
                "radius_min": 0.05,
                "radius_max": outer + 0.3,
                "min_height": 0.3,
                "max_height": z_top + 1.5,
            },
        )
        self.terminations.course_no_progress = DoneTerm(
            func=spiral_mdp.spiral_no_progress,
            # 0.3 m of climb in 30 s is 0.01 m/s -- enough to catch a
            # feet-glued stander, gentle enough to let a learner climb
            # slowly. The straight course's 2 m in 15 s (0.13 m/s along the
            # course) is a fifth of the walk; the same ratio here would be
            # 0.2 m in 15 s, but the 0.84 m of flat approach before the
            # first riser eats the first few seconds, and a bar that is
            # unreachable truncates every episode before the policy ever
            # experiences a sustained climb.
            # A gentle nudge, not a cudgel: 0.25 m of climb in 60 s. At
            # 0.3 m in 15 s it fired on 60% of the episodes, and every one of
            # those -16 penalties dominated the return, so the value scale
            # blew up and the policy diverged (raw actions at +/-50, the
            # joints pinned at their limits). The riser bonus is what pays
            # for climbing; this only has to catch a feet-glued stander.
            params={"min_gain": 0.25, "min_time": 60.0, "spawn_height": 0.84},
        )
        # The staircase is 7 m tall and the climb takes ~50 s at the commanded
        # speed, so the episode needs room to finish (and to retry).
        self.episode_length_s = 240.0


@configclass
class PioneerHumanoidOnshapeSpiralEnvCfg_PLAY(PioneerHumanoidOnshapeSpiralEnvCfg):
    """Repeatable visual verification on the spiral staircase."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 3
        self.scene.env_spacing = 0.0
        self.episode_length_s = 120.0
        # Spawn exactly on the marker, facing +Y.
        _marker = tuple(float(v) for v in self.spiral_geometry["marker"])
        self.events.reset_base.params["pose_range"]["x"] = (_marker[0] + _SPAWN_OFFSET_X,) * 2
        self.events.reset_base.params["pose_range"]["y"] = (_marker[1] + _SPAWN_OFFSET_Y,) * 2
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (_YAW_90, _YAW_90)

        self.commands.base_velocity.ranges.lin_vel_x = (0.26, 0.26)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        # The generator-only edge termination rejects USD terrain; the spiral's
        # own radius/height bounds cover it.
        self.terminations.terrain_out_of_bounds = None
        self.terminations.invalid_height_scan = None
