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

# Task knobs, read at import (one task per process). Defaults reproduce the shipped
# model (model_249_slow); each SPIRAL_* env var overrides one. Rationale: STAIR_REWARD_NOTES.md.
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
# The inherited feet_impact was blind on this course (feet touch the helicoid, not the
# filtered terrain mesh); feet_impact_contact reads contact_forces instead.
_FEET_IMPACT = float(os.environ.get("SPIRAL_FEET_IMPACT", "-5"))
# Horizontal plant speed while loaded prices a skid; -4 matches model_299_plant.
_FEET_SLIDE = float(os.environ.get("SPIRAL_FEET_SLIDE", "-4"))
# Back-kick: squared world-frame rearward speed of an airborne foot (measured with
# kick_probe.py: 43 % of swings carry the foot backwards for 0.2 s after lift-off).
# -0.5 changed nothing; -1.5 is ~46 % of income.
_BACKKICK = float(os.environ.get("SPIRAL_BACKKICK", "-1.5"))
_BACKKICK_ALLOW = float(os.environ.get("SPIRAL_BACKKICK_ALLOW", "0.10"))
# Knee-fold-behind-body term. OFF: folding is geometrically needed to clear the riser.
_KNEE_FOLD = float(os.environ.get("SPIRAL_KNEE_FOLD", "0"))
_KNEE_FOLD_ALLOW = float(os.environ.get("SPIRAL_KNEE_FOLD_ALLOW", "0.45"))
# Shank rotation while airborne, the kick as seen by eye; prices the rotation itself.
_SHANK_BACK = float(os.environ.get("SPIRAL_SHANK_BACK", "-0.02"))
# Leg smoothness: action-rate weight is the straight deliverable's -0.02 (the inherited
# -0.002 was ~3 % of income, so the knee buzzed at the control Nyquist rate).
# Slew cap on leg targets: DEFAULT OFF (cost 20 % completion, did not reduce the
# acceleration tail; the spikes are joints hitting their position stops).
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
# Price the action magnitude (nothing did). The shipped model sat at |a| ~ 10 with the
# knee on a clip rail 85 % of steps; -2e-3 is ~30 % of gross reward there and ~1 % at
# the straight deliverable's |a| ~ 1 (calibrated with l2_calib.py).
_ACTION_L2 = float(os.environ.get("SPIRAL_ACTION_L2", "-2e-3"))
# Price the knee's *unreachable* command (see the assignment below). The
# inherited weight charged 1% of reward. At -0.5 the policy paid it rather than
# leave the rail and completion fell 0.939 -> 0.903, so this is left at the
# inherited value until the target signal itself is measured.
_KNEE_OVERSHOOT = float(os.environ.get("SPIRAL_KNEE_OVERSHOOT", "-0.025"))
# Knee command slew limit (rad/s, None disables). TRIED AND OFF: a slew cap bounds step
# size, not reversal rate, so the relay still limit-cycles; kept as an optional rail.
_KNEE_RATE = os.environ.get("SPIRAL_KNEE_RATE", "0")
_KNEE_RATE = float(_KNEE_RATE) if _KNEE_RATE not in ("", "0", "none", "off") else None
# Knee action scale (loop gain): 0.05 spans the clip range over a in [-12, +9], matched
# to the |a| ~ 8-13 the network produces, instead of a two-state relay at 0.30.
_KNEE_SCALE = float(os.environ.get("SPIRAL_KNEE_SCALE", "0.05"))
# Swing foot is priced with a target height (as the straight deliverable) instead of
# the inherited band bonus, which pays for full knee flexion. SPIRAL_SWING_TARGET=0 keeps the bonus.
_SWING_TARGET = float(os.environ.get("SPIRAL_SWING_TARGET", "-40"))
# The other two smoothness terms of the straight deliverable, which this task
# never had. ``dof_vel`` damps leg-joint speed outright; ``contact_no_vel``
# prices a loaded foot that is still travelling, which is the stomp/skate the
# touchdown probe measures (0.19 m/s here against 0.09 on the straight course).
_JOINT_VEL = float(os.environ.get("SPIRAL_JOINT_VEL", "-1e-3"))
_STANCE_VEL = float(os.environ.get("SPIRAL_STANCE_VEL", "-1.0"))
# Price low horizontal swing-foot arrival. OFF: it made the policy stub the risers.
_FLY_IN = float(os.environ.get("SPIRAL_FLY_IN", "0"))
# Command speed vs the 'being asked to move' gate: the shared 0.1 threshold equals this
# task's 0.10 m/s command, which silently disabled all swing shaping. Threshold sits below it.
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
        # Scanner origins at the base, not lifted 1.5 m: from above, ~75 % of rays hit the
        # turn's underside, which is what the patches need to see past.
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

        # --- keep physics glitches out of the networks (unbounded last_action feedback blew up
        # two spiral runs at ~450 iterations; limits are far outside anything physical)
        for group in (self.observations.policy, self.observations.critic, self.observations.system_state):
            if getattr(group, "base_lin_vel", None) is not None:
                group.base_lin_vel.clip = (-10.0, 10.0)
            if getattr(group, "base_ang_vel", None) is not None:
                group.base_ang_vel.clip = (-10.0, 10.0)
            if getattr(group, "joint_vel", None) is not None:
                group.joint_vel.clip = (-30.0, 30.0)
            if getattr(group, "actions", None) is not None:
                group.actions.clip = (-5.0, 5.0)

        # No ground under an outer scan column is a drop, not a sensor fault; terminating on it
        # charged -800 for 25-67 % of episodes.
        self.terminations.invalid_height_scan = None

        # --- spawn spread along the staircase: half the batch starts on treads, not the flat
        # approach where standing costs nothing.
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

        # --- the climb reward, capped in rate: an uncapped riser bonus turned the gait into a
        # sprint (2.5-3.5 risers/s vs a 0.20 m/s command).
        self.rewards.spiral_riser_bonus.weight = _RISER_BONUS
        self.rewards.spiral_riser_bonus.params["min_interval_steps"] = _BONUS_MIN_STEPS
        self.actions.arm_swing.period = _CLOCK_PERIOD
        # Knee loop gain (see _KNEE_SCALE). Rebuilding the dict keeps every other
        # joint's scale set upstream untouched.
        knee_scale = dict(self.actions.joint_pos.scale)
        knee_scale["Knee_.*"] = _KNEE_SCALE
        self.actions.joint_pos.scale = knee_scale
        if _ACT_RANGE:
            # Scale so |a| = _ACT_RANGE reaches each joint window's farther edge; outside the
            # window the reward is flat in the action, so there is no gradient.
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
            # Rate-limit the knee target inside the action term so it is part of trained dynamics.
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
        # The knee relay, priced directly: knee_target_overshoot charges the unclipped target
        # beyond the window with a linear tail; -0.5 is ~22 % of gross reward at the measured overshoot.
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

        # --- turn the swing shaping back on: its gates compare against the command norm, which
        # equalled the shared threshold, so all three terms read exactly zero.
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

        # --- the gait: keep a foot on the ground (support-transfer task). Zero in any supported
        # step; completion 0.939 -> 0.994, falls 0.052 -> 0.005.
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
