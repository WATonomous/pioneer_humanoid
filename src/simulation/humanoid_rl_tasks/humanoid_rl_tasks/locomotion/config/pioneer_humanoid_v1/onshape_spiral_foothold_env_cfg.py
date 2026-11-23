"""Spiral staircase with foothold-conditioned perception ("Watch Your Step" on the spiral).

The straight-stairs foothold task (``onshape_foothold_env_cfg.py``) moved onto
the spiral course from ``onshape_spiral_env_cfg.py``:

* the course, spawn, path-relative command, rewards and terminations are the
  spiral's own -- nothing about the helicoid changes;
* the actor's perception is the straight task's: predicted next footholds plus
  8 x 7 height patches, while the critic keeps the whole height map;
* the arms are swung on the gait clock (12 leg actions), matching the straight
  foothold policy, whose checkpoint can be loaded here unchanged (actor 166,
  critic 859) because the observation layout is identical.

The foot predictor is seeded from the straight-stairs model (copied into
``log_dir``) and refits on the spiral during training.
"""

import math
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .onshape_spiral_env_cfg import _SPAWN_OFFSET_X, _SPAWN_OFFSET_Y, _YAW_90, PioneerHumanoidOnshapeSpiralEnvCfg
from ..pioneer import gait
from ..pioneer.flat_env_cfg import LEG_ORDER
from ..pioneer.foothold_env_cfg import _whole_body, foothold_observation_groups
from ... import mdp
from ...mdp import spiral_course as spiral_mdp
from ...mdp import stair_course as course_mdp

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7))

# Task knobs, read at import (one task per process, so this is unambiguous).
# Defaults reproduce the shipped model (model_249_slow): climb income is the
# dense height-rate term capped at the commanded path speed (weight 80 at a
# 0.10 m/s command), the per-riser bonus only makes a step-up attempt worth the
# fall risk (20), the flight penalty keeps a foot loaded (25), the commanded
# speed is tracked hard enough to matter (5), and the torso is damped against
# roll and tilt (-2 / -10).
#   SPIRAL_RISER_BONUS      per-riser bonus weight (default 600)
#   SPIRAL_BONUS_MIN_STEPS  minimum steps between paid risers (default 0 = off)
#   SPIRAL_CMD_SPEED        commanded forward speed, m/s (default 0.20)
_RISER_BONUS = float(os.environ.get("SPIRAL_RISER_BONUS", "20"))
_BONUS_MIN_STEPS = int(os.environ.get("SPIRAL_BONUS_MIN_STEPS", "0"))
# Dense climb-rate income (the capped height-rate term). Raising this and
# capping it at the commanded path speed makes the climb income independent of
# how fast the robot moves; the per-riser bonus then only has to make a
# step-up attempt worth the fall risk.
_PROGRESS_WEIGHT = float(os.environ.get("SPIRAL_PROGRESS_WEIGHT", "80"))
# Upper-body damping. The roll the torso carries (~11 deg at the step
# frequency) is what whips the head and arms; the straight course sits at 2-3
# deg. ang_vel_xy_l2 prices the rocking rate, flat_orientation_l2 the static
# tilt.
_ANG_VEL_XY = float(os.environ.get("SPIRAL_ANG_VEL_XY", "-2.0"))
# Weight on tracking the commanded forward speed. At 1.0 the term is worth at
# most 0.02 per step against ~0.16 of climb income, so the command is advisory
# and the policy ignores it. Raising it makes the commanded speed a real
# objective, which is the only way to make "walk at 0.1 m/s" mean anything.
_TRACK_LIN = float(os.environ.get("SPIRAL_TRACK_LIN", "5"))
# Gait prior: the stride cycle in seconds. One step per 0.55 m tread at the
# 0.36 m/s the policy settles at is ~0.65 steps/s, i.e. a ~3 s cycle. The
# scripted arms ride the same clock, so this also re-syncs them.
_CLOCK_PERIOD = float(os.environ.get("SPIRAL_CLOCK_PERIOD", "3.0"))
_MISMATCH_WEIGHT = float(os.environ.get("SPIRAL_MISMATCH_WEIGHT", "1.0"))
_TREAD_BONUS = float(os.environ.get("SPIRAL_TREAD_BONUS", "20"))
_FLAT_ORIENT = float(os.environ.get("SPIRAL_FLAT_ORIENT", "-10"))
_CMD_SPEED = float(os.environ.get("SPIRAL_CMD_SPEED", "0.10"))
# Landing quality. The inherited feet_impact reads the two terrain-contact
# sensors, which on this course filter the terrain mesh and not the staircase:
# the feet touch the helicoid, so the term was blind (it logged -0.0011 per
# episode, 0.1 % of income) and every landing was free. feet_impact_contact
# reads the same contact_forces sensor the gait terms use, as the straight
# stairs deliverable does.
_FEET_IMPACT = float(os.environ.get("SPIRAL_FEET_IMPACT", "-5"))
# Horizontal plant speed while the foot is loaded: 0.19 m/s at touchdown here
# against the straight course's 0.09, i.e. the foot is still travelling when
# it lands (that is the thud). The inherited -0.5 was worth 2 % of income; -4
# is worth 7 % and is the weight the shipped model (model_299_plant) uses --
# it is linear in the plant speed, so it is the term that actually prices a
# skid, where the squared vertical impact term cannot see one.
_FEET_SLIDE = float(os.environ.get("SPIRAL_FEET_SLIDE", "-4"))
# The swing leg's backward kick. Measured (kick_probe.py, 381 swings aligned at
# lift-off): the foot leaves the tread 0.24 m behind the root and travels a
# further 3 cm *backwards* over the next 0.26 s with the knee folded to -0.75
# rad, before reaching forward to land. Nothing priced the fore/aft direction of
# swing, so tucking the leg up behind the body is the cheapest way to clear the
# riser. This charges the squared rearward body-frame speed of an *airborne*
# foot, so a normal forward swing pays nothing and a planted foot (which moves
# backwards relative to the body as the robot passes over it) is not charged.
# Typical kick speed is ~0.14 m/s, i.e. 0.02 (m/s)^2 while it happens. Measured
# on the shipped model the term averages 0.055 (m/s)^2 per step (p90 0.117), and
# a full episode earns ~0.18 of reward per step, so -2 would have been ~60% of
# income and dominated the objective. -0.5 costs ~15%.
#
# This prices the foot's *world-frame* rearward velocity (body_lin_vel_w
# projected on the heading), gated on airborne. Measured with kick_probe: the
# mean swing looks fine but 43% of swings carry the foot backwards in world for
# the first 0.2 s after lift-off -- that is the visible kick, and averaging hides
# it. At -0.5 (~15% of income) it changed nothing; raw term is 0.055 (m/s)^2, so
# -1.5 charges ~46%.
_BACKKICK = float(os.environ.get("SPIRAL_BACKKICK", "-1.5"))
_BACKKICK_ALLOW = float(os.environ.get("SPIRAL_BACKKICK_ALLOW", "0.10"))
# The shank folding up behind the body after toe-off -- the kick itself, as
# distinct from where the foot ends up. Measured (kick_probe.py): the knee goes
# from -0.23 rad at lift-off to -0.73 rad within 0.24 s and stays folded, a
# ~29 deg backward rotation of the shank. Pricing the foot's position did not
# remove it, because the hip can carry the foot forward while the knee folds.
# This charges ((-q) - max_flexion)^2 for a leg whose foot is airborne, so
# pre-folding while the foot is still loaded (what a real step-up does) is free,
# and clearing the riser has to come from the hip.
# OFF: at -1.0 (34% of income) the knee fold was unchanged (-0.73 rad), because
# folding is geometrically required to clear the riser from a foot that leaves
# the tread 0.24 m behind the body -- the policy paid the penalty instead.
_KNEE_FOLD = float(os.environ.get("SPIRAL_KNEE_FOLD", "0"))
_KNEE_FOLD_ALLOW = float(os.environ.get("SPIRAL_KNEE_FOLD_ALLOW", "0.45"))
# The shank rotating backwards while airborne -- the kick as the eye sees it.
# Measured: shank pitch -22.9 -> -39.5 deg over the first 0.28 s of swing while
# the thigh only comes forward 12.5 deg. Pricing the joint angle or the foot
# position both failed; this prices the rotation itself, so the only way to
# lead the swing with the hip. Raw value is 1.54 (rad/s)^2 per step, and a full
# episode earns ~0.18 of reward per step, so -0.02 charges ~17% of income.
_SHANK_BACK = float(os.environ.get("SPIRAL_SHANK_BACK", "-0.02"))
# Leg smoothness. The policy's normalised action changes by 2.9 sigma per
# 20 ms control step here (straight course: 0.34) and the leg joints
# accelerate at 99 rad/s^2 (straight: 43) -- the visible shake.
#
# The action-rate weight is the straight-stairs deliverable's own -0.02, not
# the -0.002 this task inherited: at -0.002 the term logged 0.07 of reward per
# second against 2.1 of income (3 %, i.e. ignorable) and the policy kept
# buzzing at the control Nyquist rate -- the knee's target is clipped to
# (-1.1, -0.05) rad, so with outputs 10x the deliverable's it is a two-state
# relay and the leg visibly vibrates. At -0.02 the same buzz costs 0.7/s (35 %
# of income) and the deliverable's smooth regime costs 0.03/s, so the penalty
# now actually separates them.
# Slew cap on the leg joints' *targets*, inside the action term (trained in).
# Measured on the 2026-11-18_19-08-38 model: the mean target step is tiny
# (knee 0.03 rad) but the tails are enormous -- Ankle_P_L and Hip_F_L reach
# p99 0.74 rad and max 1.37 rad in a single 20 ms step (68 rad/s). The PD loop
# turns those into the 270 rad/s^2 leg-acceleration spikes that read as jerk, so
# pricing the mean (dof_acc_l2 x4) did almost nothing. 12 rad/s (0.24 rad/step)
# clips only those tails: it sits above every joint's mean and below the p99 of
# the ankle and hip pitch. Earlier knee-only attempts did nothing precisely
# because the knee's p99 (0.21) is well under any sane cap.
#
# DEFAULT OFF. Trained in at 12 rad/s it cost 20 % completion (0.951 -> 0.749)
# and left the leg-acceleration tail unchanged (p99 270 -> 246): the spikes are
# not command slew, they are the joints hitting their position stops (see
# SPIRAL_ACT_CLIP). It was briefly default-on, which silently put every probe
# and video of the earlier checkpoints under the cap.
_LEG_RATE = float(os.environ.get("SPIRAL_LEG_RATE", "0"))
_ACTION_RATE = float(os.environ.get("SPIRAL_ACTION_RATE", "-0.02"))
# Hinge on the raw action beyond +-_SAT_THRESHOLD (see gait.action_saturation_l2).
_SAT = float(os.environ.get("SPIRAL_SAT", "0"))
_SAT_THRESHOLD = float(os.environ.get("SPIRAL_SAT_THRESHOLD", "3.0"))
# Narrow fixes (all swing-phase only; see gait.py): forward foot speed in the
# first 0.2 s of swing, the knee's commanded-angle change in swing, and a
# knee-only saturation hinge.
# Re-parameterised action space (see ACT_RANGE below). 0 = the legacy scales.
_ACT_RANGE = float(os.environ.get("SPIRAL_ACT_RANGE", "0"))
# Commandable target window of each leg joint (rad) and its neutral pose: the joint
# range extended by effort/kp (joint_info.py) so full torque stays reachable at any
# angle. NOT the bare joint range -- clipping to that cannot climb (the policy
# commands torque by overshooting the target). The knee keeps its own clip.
_JOINT_WINDOW = {
    "Hip_F_.*": (-1.46, 2.68, 0.25), "Hip_A_.*": (-0.97, 1.85, 0.0), "Hip_R_.*": (-0.92, 1.97, 0.0),
    "Knee_.*": (-1.10, -0.05, -0.5), "Ankle_P_.*": (-2.02, 2.37, -0.25), "Ankle_R_.*": (-1.76, 1.76, 0.0),
}

if os.environ.get("SPIRAL_WINDOW_BARE"):   # the bare joint ranges
    _JOINT_WINDOW = {'Hip_F_.*': (-0.35, 1.57, 0.25), 'Hip_A_.*': (-0.17, 1.05, 0.0), 'Hip_R_.*': (-0.52, 1.57, 0.0), 'Knee_.*': (-1.1, -0.05, -0.5), 'Ankle_P_.*': (-0.52, 0.87, -0.25), 'Ankle_R_.*': (-0.26, 0.26, 0.0)}
_WIDE = float(os.environ.get("SPIRAL_WINDOW_WIDE", "0"))   # diagnostic: non-knee windows = +-WIDE rad (nothing clips)
if _WIDE:
    _JOINT_WINDOW = {k: ((-_WIDE, _WIDE, off) if not k.startswith("Knee") else (lo, hi, off))
                     for k, (lo, hi, off) in _JOINT_WINDOW.items()}
_STANCE_TILT = float(os.environ.get("SPIRAL_STANCE_TILT", "0"))
_STANCE_TILT_ALLOW = float(os.environ.get("SPIRAL_STANCE_TILT_ALLOW", "10"))
_SWING_FWD = float(os.environ.get("SPIRAL_SWING_FWD", "0"))
_KNEE_SWING_RATE = float(os.environ.get("SPIRAL_KNEE_SWING_RATE", "0"))
_KNEE_SAT = float(os.environ.get("SPIRAL_KNEE_SAT", "0"))
_KNEE_SAT_THRESHOLD = float(os.environ.get("SPIRAL_KNEE_SAT_THRESHOLD", "7.0"))
_DOF_ACC = float(os.environ.get("SPIRAL_DOF_ACC", "-1.5e-6"))
# Price the action *magnitude*, which nothing did. Measured on the shipped
# model (sat_probe.py, 400 steps, marker spawn): mean |a| 10.8-13.9 on the
# hips and 9.9-11.3 on the knees against the straight deliverable's 0.9-1.3;
# commanded targets outside the joint's range on 50-84% of steps; motors at
# their effort limit on 26-59% of steps; and the knee target sitting exactly on
# one of its clip values (-1.10 / -0.05 rad) on 85.2% of steps, i.e. the knee
# is a two-state actuator -- the visible clunk. The drift arrived with the
# action std growing 0.84 -> 2.01 rad during fine-tuning (entropy_coef = 0
# leaves the std with no restoring force).
#
# ``action_rate_l2`` only prices the *change*, so a policy that simply sits in
# saturation pays nothing and is never pulled back. It sums the squared
# actions over the channels, as IsaacLab's ``action_l2`` does.
#
# The weight is calibrated, not guessed (l2_calib.py): the reward manager
# charges ``weight * sum(a^2) * dt`` per step and logs it divided by
# ``max_episode_length_s``, which is why the first attempt at -2e-4 looked 39x
# smaller than the measured action. Measured on the shipped model the charge
# was 0.19-0.53 per step at sum(a^2) = 957-2673 -- about 5% of gross reward,
# i.e. it never outweighed the climb income that saturation buys. -2e-3 (10x)
# costs ~30% of gross reward at |a| ~ 10 and still only ~1% at |a| ~ 1, so it
# is free for the straight deliverable's regime and punitive here.
_ACTION_L2 = float(os.environ.get("SPIRAL_ACTION_L2", "-2e-3"))
# Price the knee's *unreachable* command (see the assignment below). The
# inherited weight charged 1% of reward. At -0.5 the policy paid it rather than
# leave the rail and completion fell 0.939 -> 0.903, so this is left at the
# inherited value until the target signal itself is measured.
_KNEE_OVERSHOOT = float(os.environ.get("SPIRAL_KNEE_OVERSHOOT", "-0.025"))
# The knee's command slew limit, in rad/s of joint target (None disables).
# Measured (knee_rate_stats.py): the knee target's step change has a median of
# exactly 0.000 rad and a p99 of 0.89 rad, max 1.05 rad -- the whole clip range
# in one 20 ms step, i.e. a relay, not a servo.
#
# TRIED AND OFF. The cap was trained in at the tightest survivable value
# (16 rad/s: at 512 envs x 6000 steps it costs nothing, completion 0.919-0.929
# vs 0.929) and it does not touch the shake: knee velocity reversals read 23.96
# /s unlimited, 23.93 /s at 16 rad/s, 24.16 /s on the retrained model, and taps
# per riser went 3.6 -> 5.1. A slew cap bounds how far the command moves per
# step, not how often it reverses, so a rate-limited relay still limit-cycles at
# the same frequency. The cause is loop gain, fixed by _KNEE_SCALE below; the
# mechanism is kept (mdp/joint_rate_limit.py) as an optional deploy-time rail.
_KNEE_RATE = os.environ.get("SPIRAL_KNEE_RATE", "0")
_KNEE_RATE = float(_KNEE_RATE) if _KNEE_RATE not in ("", "0", "none", "off") else None
# The knee's action scale, i.e. its loop gain. The inherited 0.30 maps the whole
# 1.05 rad clip range (-1.10..-0.05, offset -0.5) onto only a in [-2.0, +1.5],
# while this policy's knee output is |a| ~ 8-13: the command sits on a clip rail
# on ~85% of steps, so the knee is a two-state relay whose small-signal gain is
# ~5x too high and it limit-cycles at ~24 velocity reversals/s (the straight
# deliverable, which walks smoothly through the range, reverses at 12-16/s).
# 0.05 makes the same clip range span a in [-12, +9], matched to the output the
# network actually produces, so inside it a one-unit change moves the knee 0.05
# rad instead of slamming between the rails.
_KNEE_SCALE = float(os.environ.get("SPIRAL_KNEE_SCALE", "0.05"))
# Swing shaping: the straight deliverable prices the swing foot with a *target*
# (``feet_swing_height_relative``: squared error of the airborne sole's height
# above the lower foot from 0.10 m, weight -40), whereas this task inherited
# ``foot_swing_clearance``, a *band bonus* (+40) for any clearance inside a
# wide band. The bonus pays for the extreme knee angle -- with the knee target
# clipped to (-1.10, -0.05) rad the cheapest way to collect it is full flexion
# -- and the shipped model's knee sits on one of those two clip values on 85%
# of steps. Swapping to the target makes the knee worth a continuous angle
# instead of a rail. Set SPIRAL_SWING_TARGET=0 to keep the bonus.
_SWING_TARGET = float(os.environ.get("SPIRAL_SWING_TARGET", "-40"))
# The other two smoothness terms of the straight deliverable, which this task
# never had. ``dof_vel`` damps leg-joint speed outright; ``contact_no_vel``
# prices a loaded foot that is still travelling, which is the stomp/skate the
# touchdown probe measures (0.19 m/s here against 0.09 on the straight course).
_JOINT_VEL = float(os.environ.get("SPIRAL_JOINT_VEL", "-1e-3"))
_STANCE_VEL = float(os.environ.get("SPIRAL_STANCE_VEL", "-1.0"))
# The stomp itself: a swing foot that is low (within 6 cm of the lower foot)
# and still travelling horizontally when it meets the tread. The touchdown
# probes say the landings here are vertically quiet (0.01 m/s) but horizontally
# quick (0.16 m/s against the straight deliverable's 0.09), so this prices the
# approach rather than the impact, which no contact-gated term can see.
# OFF by default: at -2 over 300 iterations it did what it was asked (peak
# touchdown force 680 -> 670 N) but cost 10 points of completion (0.894 ->
# 0.797) and doubled out-of-bounds (0.039 -> 0.086) -- penalising the low
# forward carry of the swing foot makes the policy stub the risers instead of
# reaching over them. The horizontal arrival is a symptom of the short stride,
# not something a cost can squeeze out on its own.
_FLY_IN = float(os.environ.get("SPIRAL_FLY_IN", "0"))
# The commanded speed, and the threshold the "is the robot being asked to move"
# gates compare against. Those gates (swing clearance, air time, knee tuck) are
# a strict `>` on the command norm, and this task commands exactly 0.10 m/s --
# so with the shared 0.1 threshold they were ALL false on every step and the
# whole swing-shaping block (clearance 40, knee tuck -4, air time 1) was dead
# weight: the run logged 0.0000 for all three while the robot shuffled up the
# stairs on straight knees. The threshold now sits below the command, so the
# terms fire; the shared default is untouched, so the straight course is
# unchanged.
_MIN_CMD = float(os.environ.get("SPIRAL_MIN_CMD", "0.02"))
# Penalty on a flight phase (both feet off the ground), 0 = off. 120 is the
# value the shipped model was trained with: it took the gait from 27% foot
# contact (a hop, 4.3 footfalls/s) to 54% (a walk), and completion 0.939 ->
# 0.994 with falls 0.052 -> 0.005.
_FLIGHT = float(os.environ.get("SPIRAL_FLIGHT", "25"))

# Tread poses (x, y, root z, yaw) measured off the merged collider by
# ``tools/onshape_course/spiral_spawn_slots.py``: the marker (slot 0), then the
# centres of treads 4, 9, 16, 24 and 34 risers up, on the walking circle with
# the yaw on the path's tangent. ``z`` is the root height at rest on the tread,
# so the robot spawns standing instead of dropping onto it.
FEET_CONTACT = SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True)
FEET_POSE = SceneEntityCfg("robot", body_names=["l_foot", "r_foot"], preserve_order=True)
KNEE_JOINTS_CFG = SceneEntityCfg("robot", joint_names=["Knee_L", "Knee_R"], preserve_order=True)
SHANK_CFG = SceneEntityCfg("robot", body_names=["l_shank_foot", "r_shank_foot"], preserve_order=True)

SPIRAL_SPAWN_SLOTS: tuple[tuple[float, float, float, float], ...] = (
    (1.551, -0.777, 0.816, 1.5708),
    (0.213, 1.500, 1.417, 3.0005),
    (-1.504, -0.182, 2.017, 4.8331),
    (1.361, -0.666, 2.857, 7.3987),
    (-1.192, 0.935, 3.817, 10.3309),
    (1.500, -0.213, 5.017, 13.9960),
)



@configclass
class PioneerHumanoidSpiralFootholdEnvCfg(PioneerHumanoidOnshapeSpiralEnvCfg):
    """Climb the 3.2-turn helicoid seeing where the feet will land."""

    log_dir: str = os.path.join(_REPO, "outputs", "rl", "pioneer_humanoid_spiral_foothold", "foot_models")

    def __post_init__(self):
        super().__post_init__()
        os.makedirs(self.log_dir, exist_ok=True)
        # The critic's full map is 5 cm here, the same as the straight foothold
        # task, so the asymmetric checkpoint's critic shape carries over.
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0])
        # The foothold patches need the treads *under* the robot. The spiral
        # recipe lifts the scanner 1.5 m to dodge the turn above, but from
        # there ~75% of the rays hit that turn's underside (heights ~-1.5 m,
        # clipped to -1), which is precisely the information the patches need.
        # Ray origins at the base see the tread the robot stands on, i.e. the
        # same height encoding the straight-stairs checkpoint was trained on.
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0))
        # The straight foothold checkpoint was trained with the scripted arm
        # swing's gait clock in the actor observation (2 dims, between the
        # actions and the foothold block). This course's base recipe does not
        # carry that term, so add it in the same slot before the patches.
        self.observations.policy.gait_clock = ObsTerm(func=gait.gait_clock, params={"period": _CLOCK_PERIOD})
        foothold_observation_groups(self, patch_rows_behind=1)
        # Scripted arms on the gait clock (12 leg actions), exactly as the
        # straight foothold task: the checkpoint's action layer still lines up.
        _whole_body(self)

        # --- keep physics glitches out of the networks ----------------------
        # The recipe clips the scan terms (and maps non-finite rays to the
        # clip limit), but the base-motion, joint-velocity and last-action
        # terms are unbounded -- and last_action feeds back: one spurious
        # contact spike (a foot catching a riser edge on the helicoid) enters
        # the actor, whose output becomes the next step's input. Both spiral
        # runs died this way at ~450 iterations: action_rate_l2 spiked 350,000x
        # in one iteration, the value loss followed it to 1e12, then the
        # policy std went NaN -- while the actor weights were flat (max |w|
        # 0.44 from iteration 0 to 400), i.e. the blow-up came from the input,
        # not the network. The limits are far outside anything physical.
        for group in (self.observations.policy, self.observations.critic, self.observations.system_state):
            if getattr(group, "base_lin_vel", None) is not None:
                group.base_lin_vel.clip = (-10.0, 10.0)
            if getattr(group, "base_ang_vel", None) is not None:
                group.base_ang_vel.clip = (-10.0, 10.0)
            if getattr(group, "joint_vel", None) is not None:
                group.joint_vel.clip = (-30.0, 30.0)
            if getattr(group, "actions", None) is not None:
                group.actions.clip = (-5.0, 5.0)

        # A ray that leaves the staircase from the base-level scanner is not a
        # sensor fault: there is genuinely no ground there, and the observation
        # already carries that as -1 (a drop). Terminating on it charged a full
        # -800 fall for the outer grid columns alone and ended 25-67% of the
        # episodes in the transfer run. The physical bounds, base contact, tilt
        # and physics guards still catch every real fall.
        self.terminations.invalid_height_scan = None

        # --- spawn spread along the staircase -------------------------------
        # Every episode used to start on the flat 0.84 m approach, where
        # standing costs nothing: that is the attractor the straight-stairs
        # transfer sat in for 455 iterations without climbing once. Start
        # half the batch *on* treads further up instead, so the policy sees
        # mid-staircase states from the first update.
        markers = tuple(float(v) for v in self.spiral_geometry["marker"])
        self.spiral_marker_slot = (markers[0] + _SPAWN_OFFSET_X, markers[1] + _SPAWN_OFFSET_Y, 0.845, _YAW_90)
        self.events.reset_base = EventTerm(
            func=spiral_mdp.reset_root_state_spiral_slots,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "slots": SPIRAL_SPAWN_SLOTS or (self.spiral_marker_slot,),
                "riser": float(self.spiral_geometry["riser"]),
                "standing_height": 0.845,
                "xy_jitter": 0.05,
                "yaw_jitter": 0.05,
                "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
            },
        )

        # --- the climb reward, capped in rate --------------------------------
        # The riser bonus pays for a step-up, which has to outweigh a -16 fall
        # for the policy to keep attempting one. Left uncapped it is also worth
        # more *per second* the faster the robot moves, and that is what turned
        # the gait into a sprint (2.5-3.5 risers/s against a 0.20 m/s command).
        # The cap pays at most one riser per SPIRAL_BONUS_MIN_STEPS, so speed
        # above roughly the commanded cadence earns nothing extra while a
        # normal step-up still pays the full bonus. The two dense progress
        # terms are capped at the commanded path speed for the same reason.
        self.rewards.spiral_riser_bonus.weight = _RISER_BONUS
        self.rewards.spiral_riser_bonus.params["min_interval_steps"] = _BONUS_MIN_STEPS
        self.actions.arm_swing.period = _CLOCK_PERIOD
        # Knee loop gain (see _KNEE_SCALE). Rebuilding the dict keeps every other
        # joint's scale set upstream untouched.
        knee_scale = dict(self.actions.joint_pos.scale)
        knee_scale["Knee_.*"] = _KNEE_SCALE
        self.actions.joint_pos.scale = knee_scale
        if _ACT_RANGE:
            # Scale each joint so |a| = _ACT_RANGE reaches the *farther* edge of
            # its window, and clip the commanded angle to the window. Measured
            # on the old scales, the low edge was reached at |a| of only 2.3-3.5
            # while the policy sat at |a| of 8-10: outside the window the reward
            # is flat in the action, so the policy gets no gradient there (see
            # gait.action_saturation_l2). Here the whole window lives inside
            # |a| <= _ACT_RANGE, so every command that matters has a gradient.
            self.actions.joint_pos.scale = {
                k: max(hi - off, off - lo) / _ACT_RANGE for k, (lo, hi, off) in _JOINT_WINDOW.items()}
            self.actions.joint_pos.clip = {k: (lo, hi) for k, (lo, hi, off) in _JOINT_WINDOW.items()}
        if _LEG_RATE:
            self.actions.joint_pos.class_type = mdp.JointPositionActionRateLimited
            self.actions.joint_pos.max_rate = {
                "Hip_F_.*": _LEG_RATE, "Hip_A_.*": _LEG_RATE, "Hip_R_.*": _LEG_RATE,
                "Knee_.*": _LEG_RATE, "Ankle_P_.*": _LEG_RATE, "Ankle_R_.*": _LEG_RATE,
            }
        if _KNEE_RATE:
            # Rate-limit the knee's target inside the action term, so the limit
            # is part of the dynamics the policy trains against (a post-hoc
            # filter on the trained policy stops the climb -- see
            # mdp/joint_rate_limit.py). Mutating the inherited cfg rather than
            # rebuilding it keeps every scale/offset/clip set upstream.
            self.actions.joint_pos.class_type = mdp.JointPositionActionRateLimited
            self.actions.joint_pos.max_rate = {"Knee_.*": _KNEE_RATE}
        self.rewards.track_lin_vel_xy_exp.weight = _TRACK_LIN
        if _STANCE_TILT:
            self.rewards.stance_tilt_hinge = RewTerm(
                func=gait.stance_foot_tilt_hinge, weight=_STANCE_TILT,
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE,
                        "allowance_deg": _STANCE_TILT_ALLOW},
            )
        if _SWING_FWD:
            self.rewards.swing_start_forward = RewTerm(
                func=gait.swing_start_forward, weight=_SWING_FWD,
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE},
            )
        if _KNEE_SWING_RATE:
            self.rewards.knee_swing_rate = RewTerm(
                func=gait.knee_swing_target_rate_l2, weight=_KNEE_SWING_RATE,
                params={"sensor_cfg": FEET_CONTACT},
            )
        if _KNEE_SAT:
            self.rewards.knee_saturation = RewTerm(
                func=gait.action_saturation_l2, weight=_KNEE_SAT,
                params={"threshold": _KNEE_SAT_THRESHOLD, "joint_names": ["Knee_L", "Knee_R"]},
            )
        if _SAT:
            self.rewards.action_saturation = RewTerm(
                func=gait.action_saturation_l2,
                weight=_SAT,
                params={"threshold": _SAT_THRESHOLD},
            )
        if _BACKKICK:
            # Price the swing leg hanging behind the body (see _BACKKICK).
            self.rewards.feet_swing_behind = RewTerm(
                func=gait.feet_swing_behind_l2,
                weight=_BACKKICK,
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE,
                        "allowance": _BACKKICK_ALLOW},
            )
        if _SHANK_BACK:
            # Price the shank rotating backwards (see _SHANK_BACK).
            self.rewards.shank_backward_rotation = RewTerm(
                func=gait.shank_backward_rotation_l2,
                weight=_SHANK_BACK,
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": SHANK_CFG},
            )
        if _KNEE_FOLD:
            # Price the shank folding up behind the body after toe-off, which is
            # the kick the foot-position term cannot see (see _KNEE_FOLD).
            self.rewards.knee_flexion_airborne = RewTerm(
                func=gait.knee_flexion_airborne_l2,
                weight=_KNEE_FOLD,
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": KNEE_JOINTS_CFG,
                        "max_flexion": _KNEE_FOLD_ALLOW},
            )
        if _MISMATCH_WEIGHT:
            # Feet must be in the clock's stance/swing state (4 = both wrong).
            self.rewards.contact_mismatch = RewTerm(
                func=gait.contact_mismatch_l2,
                weight=-_MISMATCH_WEIGHT,
                params={"sensor_cfg": FEET_CONTACT, "period": _CLOCK_PERIOD},
            )
        if _TREAD_BONUS:
            # A footfall that lands on a tread above the other foot pays; a step
            # back onto the same tread pays nothing.
            self.rewards.tread_step = RewTerm(
                func=spiral_mdp.tread_step_bonus,
                weight=_TREAD_BONUS,
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE,
                        "riser": float(self.spiral_geometry["riser"])},
            )
        self.rewards.ang_vel_xy_l2.weight = _ANG_VEL_XY
        self.rewards.flat_orientation_l2.weight = _FLAT_ORIENT

        # --- landing quality -------------------------------------------------
        # The inherited term reads the terrain-contact sensors, which filter the
        # terrain mesh; the feet land on the staircase asset, so it never fired.
        self.rewards.feet_impact = RewTerm(
            func=course_mdp.feet_impact_contact,
            weight=_FEET_IMPACT,
            params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE},
        )
        # Plant the foot, don't skate it in: this is the horizontal half of the
        # same impact.
        self.rewards.feet_slide.weight = _FEET_SLIDE
        # Price the leg shake itself (action and joint-acceleration rate), and
        # the two smoothness terms the straight deliverable has and this task
        # never did: leg-joint speed, and a loaded foot that is still moving.
        self.rewards.action_rate_l2.weight = _ACTION_RATE
        self.rewards.dof_acc_l2.weight = _DOF_ACC
        if _ACTION_L2:
            self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=_ACTION_L2)
        # The knee relay, priced directly. The knee's usable action window is
        # only a in [-2, 1.5] (scale 0.3, offset -0.5, clip -1.10/-0.05 rad)
        # while the policy outputs |a| ~ 8-10, so the command sits ~1.9 rad
        # outside the reachable window on average and every small change in the
        # network's output is a ~2.4 rad jump in commanded knee angle: a
        # feedback loop with 4-5x too much gain, which limit-cycles at ~7 Hz
        # (knee velocity reverses sign 14x/s -- the visible jitter).
        #
        # ``knee_target_overshoot`` charges the *unclipped* target's distance
        # beyond those bounds (smooth-L1, linear tail, so it still has a
        # gradient deep in saturation), and is zero for any command inside the
        # window. Unlike ``action_l2`` it does not tax legitimate motion, so it
        # can be made large without flattening the gait. At the inherited
        # -0.025 it charged 0.031/s, about 1% of reward, and the relay survived
        # three runs. -0.5 is ~22% of gross reward at the measured overshoot.
        self.rewards.knee_target_overshoot.weight = _KNEE_OVERSHOOT
        legs = SceneEntityCfg("robot", joint_names=LEG_ORDER)
        self.rewards.dof_vel = RewTerm(
            func=mdp.joint_vel_l2, weight=_JOINT_VEL, params={"asset_cfg": legs},
        )
        self.rewards.contact_no_vel = RewTerm(
            func=gait.stance_foot_velocity,
            weight=_STANCE_VEL,
            params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE},
        )
        if _FLY_IN:
            self.rewards.foot_low_fly_in = RewTerm(
                func=course_mdp.foot_low_fly_in,
                weight=_FLY_IN,
                params={"sensor_cfg": FEET_CONTACT, "feet_cfg": FEET_POSE},
            )

        # --- turn the swing shaping back on ----------------------------------
        # Every one of these three is gated on `command norm > threshold`, and
        # the shared threshold (0.1) equals this task's commanded speed, so all
        # three read exactly zero on every step. That is why the policy walked
        # the staircase on locked-straight knees, stubbing each riser and
        # slamming onto the tread: the term that pays for lifting the foot
        # (weight 40), the one that pays for tucking the knee (-4) and the
        # air-time prior (1.0) never paid anything.
        self.rewards.feet_air_time.params["min_command_norm"] = _MIN_CMD
        self.rewards.knee_swing_flexion.params["move_threshold"] = _MIN_CMD
        if _SWING_TARGET:
            # Replace the clearance band bonus with the straight deliverable's
            # swing-height target (see _SWING_TARGET). Same feet, same sensor,
            # but the spiral's own band rather than the straight course's
            # 0.10 m, because this staircase's riser is 0.12 m.
            self.rewards.foot_swing_clearance = None
            self.rewards.feet_swing_height = RewTerm(
                func=gait.feet_swing_height_relative,
                weight=_SWING_TARGET,
                # The spiral's own clearance band (riser 0.12 + 0.06 margin),
                # not the straight course's 0.10 m: a target below the riser
                # would pay the policy to stub every step.
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE,
                        "target": self.spiral_band},
            )
        else:
            self.rewards.foot_swing_clearance.params["min_command_norm"] = _MIN_CMD
        self.commands.base_velocity.ranges.lin_vel_x = (_CMD_SPEED, _CMD_SPEED)
        if _PROGRESS_WEIGHT != 5.0 or _BONUS_MIN_STEPS > 0:
            # The dense climb terms stop paying above the commanded path
            # speed, so running earns nothing more per second than walking.
            self.rewards.spiral_progress.weight = _PROGRESS_WEIGHT
            self.rewards.spiral_progress.params["cap"] = _CMD_SPEED
            self.rewards.spiral_path_progress.params["cap"] = _CMD_SPEED

        # --- the gait: keep a foot on the ground ------------------------------
        # A staircase is a support-transfer task, so a flight phase is a fall
        # that lands higher up. The straight-course policy keeps a foot loaded
        # 66% of the time; the spiral policy was at 27% (4.3 footfalls/s, 44 deg
        # forward lean). This term is zero in every supported step, so it does
        # not prefer standing still (which earns no climb reward). With it the
        # gait keeps a foot loaded 54% of the time and completion went 0.939 ->
        # 0.994, falls 0.052 -> 0.005.
        if _FLIGHT != 0.0:
            self.rewards.feet_flight = RewTerm(
                func=gait.feet_flight,
                weight=-_FLIGHT,
                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"],
                                                     preserve_order=True)},
            )

@configclass
class PioneerHumanoidSpiralFootholdEnvCfg_PLAY(PioneerHumanoidSpiralFootholdEnvCfg):
    """Repeatable visual check: marker spawn, predictor frozen, no corruption."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # One robot, one staircase: at 3 envs with env_spacing 0 the extra two
        # robots stand inside the first one and the render reads as a crowd of
        # overlapping bodies all climbing at once.
        self.scene.num_envs = 1
        self.scene.env_spacing = 0.0
        self.episode_length_s = 120.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.terminations.invalid_height_scan = None
        self.terminations.terrain_out_of_bounds = None
        self.commands.base_velocity.ranges.lin_vel_x = (_CMD_SPEED, _CMD_SPEED)
        # Pin the spawn to the Part Studio marker (facing +Y, as specified) for
        # a repeatable visual: the training slots start halfway up the stairs.
        self.events.reset_base.params["slots"] = (self.spiral_marker_slot,)
        self.events.reset_base.params["xy_jitter"] = 0.0
        self.events.reset_base.params["yaw_jitter"] = 0.0

        # The straight foothold PLAY camera, riding on the robot. `asset_root`
        # tracks the root *position* only (the eye/lookat offsets stay world
        # axes), so on a spiral -- where the robot circles a column -- a fixed
        # offset alternates between outside the staircase and buried in the
        # turn above. A fixed wide shot of the whole 7 m helicoid shows the
        # climb without that; the trade is a smaller robot.
        from isaaclab.envs import ViewerCfg

        def _vec(name: str, default: str) -> tuple[float, float, float]:
            return tuple(float(v) for v in os.environ.get(name, default).split(","))

        self.viewer = ViewerCfg(
            eye=_vec("SPIRAL_CAM_EYE", "7.5,-8.0,5.6"),
            lookat=_vec("SPIRAL_CAM_LOOKAT", "1.0,0.0,2.4"),
            origin_type="world",
        )

        # Freeze the predictor and draw what the actor sees.
        for group in (self.observations.policy, self.observations.critic):
            group.foothold_scan.params.update(train_fpm=False, force_model=True)
        self.observations.policy.foothold_scan.params["show_patches"] = (
            os.environ.get("SHOW_FOOTHOLDS", "0") == "1"
        )
