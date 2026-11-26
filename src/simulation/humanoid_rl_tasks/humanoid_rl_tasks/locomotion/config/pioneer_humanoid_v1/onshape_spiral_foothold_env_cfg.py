"""Spiral staircase with foothold-conditioned perception: the straight foothold task on the helicoid.

* course, spawn, path-relative command, rewards and terminations come from ``onshape_spiral_env_cfg.py``;
* the actor sees predicted next footholds plus 8 x 7 height patches and the critic the full height map
  (same layout as the straight task, so its checkpoint loads unchanged: actor 166, critic 859);
* the arms swing on the gait clock (12 leg actions); the foot predictor starts from the straight model
  and refits on the spiral during training.
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

# Task knobs, read at import (one task per process). Defaults reproduce the best spiral recipe;
# each SPIRAL_* env var overrides one. Rationale: STAIR_REWARD_NOTES.md.
_RISER_BONUS = float(os.environ.get("SPIRAL_RISER_BONUS", "20"))
_BONUS_MIN_STEPS = int(os.environ.get("SPIRAL_BONUS_MIN_STEPS", "0"))
_PROGRESS_WEIGHT = float(os.environ.get("SPIRAL_PROGRESS_WEIGHT", "80"))   # dense climb rate, capped at the command
_TRACK_LIN = float(os.environ.get("SPIRAL_TRACK_LIN", "5"))                # makes the commanded speed a real objective
_CMD_SPEED = float(os.environ.get("SPIRAL_CMD_SPEED", "0.10"))
_MIN_CMD = float(os.environ.get("SPIRAL_MIN_CMD", "0.02"))                 # 'asked to move' gate, below the command
_CLOCK_PERIOD = float(os.environ.get("SPIRAL_CLOCK_PERIOD", "3.0"))        # stride cycle (s); the scripted arms share it
_TREAD_BONUS = float(os.environ.get("SPIRAL_TREAD_BONUS", "20"))
_MISMATCH_WEIGHT = float(os.environ.get("SPIRAL_MISMATCH_WEIGHT", "0.4"))  # feet vs the clock's stance/swing state

# torso and landing
_ANG_VEL_XY = float(os.environ.get("SPIRAL_ANG_VEL_XY", "-2.0"))
_FLAT_ORIENT = float(os.environ.get("SPIRAL_FLAT_ORIENT", "-10"))
_FEET_IMPACT = float(os.environ.get("SPIRAL_FEET_IMPACT", "-100"))         # feet_impact_contact: the inherited one is blind here
_FEET_SLIDE = float(os.environ.get("SPIRAL_FEET_SLIDE", "-4"))
_STANCE_TILT = float(os.environ.get("SPIRAL_STANCE_TILT", "-0.05"))        # sole tilt beyond the allowance (heel lift)
_STANCE_TILT_ALLOW = float(os.environ.get("SPIRAL_STANCE_TILT_ALLOW", "10"))
_STANCE_VEL = float(os.environ.get("SPIRAL_STANCE_VEL", "-1.0"))           # a loaded foot that is still moving

# swing shaping
_BACKKICK = float(os.environ.get("SPIRAL_BACKKICK", "-1.5"))
_BACKKICK_ALLOW = float(os.environ.get("SPIRAL_BACKKICK_ALLOW", "0.10"))
_SHANK_BACK = float(os.environ.get("SPIRAL_SHANK_BACK", "-0.02"))
_SWING_FWD = float(os.environ.get("SPIRAL_SWING_FWD", "0.5"))
_SWING_TARGET = float(os.environ.get("SPIRAL_SWING_TARGET", "-40"))        # swing height target; 0 keeps the clearance bonus
_KNEE_SWING_RATE = float(os.environ.get("SPIRAL_KNEE_SWING_RATE", "-3"))
_FLIGHT = float(os.environ.get("SPIRAL_FLIGHT", "25"))                     # both feet airborne

# smoothness and action space
_ACTION_RATE = float(os.environ.get("SPIRAL_ACTION_RATE", "-0.02"))
_ACTION_L2 = float(os.environ.get("SPIRAL_ACTION_L2", "-2e-3"))
_DOF_ACC = float(os.environ.get("SPIRAL_DOF_ACC", "-6e-6"))
_JOINT_VEL = float(os.environ.get("SPIRAL_JOINT_VEL", "-1e-3"))
_SAT = float(os.environ.get("SPIRAL_SAT", "-0.01"))                        # hinge on |action| beyond _SAT_THRESHOLD
_SAT_THRESHOLD = float(os.environ.get("SPIRAL_SAT_THRESHOLD", "3.0"))
_KNEE_OVERSHOOT = float(os.environ.get("SPIRAL_KNEE_OVERSHOOT", "-0.025"))
_KNEE_SCALE = float(os.environ.get("SPIRAL_KNEE_SCALE", "0.05"))           # knee loop gain
_ACT_RANGE = float(os.environ.get("SPIRAL_ACT_RANGE", "3"))                # |a| that reaches each window's far edge; 0 = legacy scales

# Commandable target window of each leg joint (rad) and its neutral pose: the joint range extended by
# effort/kp, so full torque stays reachable at any angle. The knee keeps its own clip.
_JOINT_WINDOW = {
    "Hip_F_.*": (-1.46, 2.68, 0.25), "Hip_A_.*": (-0.97, 1.85, 0.0), "Hip_R_.*": (-0.92, 1.97, 0.0),
    "Knee_.*": (-1.10, -0.05, -0.5), "Ankle_P_.*": (-2.02, 2.37, -0.25), "Ankle_R_.*": (-1.76, 1.76, 0.0),
}

FEET_CONTACT = SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True)
FEET_POSE = SceneEntityCfg("robot", body_names=["l_foot", "r_foot"], preserve_order=True)
KNEE_JOINTS_CFG = SceneEntityCfg("robot", joint_names=["Knee_L", "Knee_R"], preserve_order=True)
SHANK_CFG = SceneEntityCfg("robot", body_names=["l_shank_foot", "r_shank_foot"], preserve_order=True)

# Tread poses (x, y, root z, yaw) measured off the merged collider by tools/onshape_course/spiral_spawn_slots.py:
# the marker, then treads 4, 11, 19, 27, 35 and 43 on the walking circle, yaw on the path tangent, z = root height at rest.
SPIRAL_SPAWN_SLOTS: tuple[tuple[float, float, float, float], ...] = (
    (1.551, -0.777, 0.816, 1.5708),
    (0.874, 1.238, 1.445, 2.5269),
    (-1.140, 0.998, 2.285, 3.9930),
    (-0.874, -1.238, 3.245, 5.6685),
    (1.322, -0.740, 4.205, 7.3440),
    (0.597, 1.392, 5.165, 9.0195),
    (-1.447, 0.448, 6.125, 10.6951),
)


@configclass
class PioneerHumanoidSpiralFootholdEnvCfg(PioneerHumanoidOnshapeSpiralEnvCfg):
    """Climb the 1.7-turn helicoid seeing where the feet will land."""

    log_dir: str = os.path.join(_REPO, "outputs", "rl", "pioneer_humanoid_spiral_foothold", "foot_models")

    def __post_init__(self):
        super().__post_init__()
        os.makedirs(self.log_dir, exist_ok=True)
        # The critic's full map is 5 cm here, the same as the straight foothold
        # task, so the asymmetric checkpoint's critic shape carries over.
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0])
        # scanner at the base: from above, ~75 % of rays hit the turn's underside
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0))
        # the straight checkpoint has the gait clock in the actor observation; add it before the patches
        self.observations.policy.gait_clock = ObsTerm(func=gait.gait_clock, params={"period": _CLOCK_PERIOD})
        foothold_observation_groups(self, patch_rows_behind=1)
        # scripted arms on the gait clock, as in the straight task
        _whole_body(self)

        # keep physics glitches out of the networks (unbounded last_action feedback blew up two runs)
        for group in (self.observations.policy, self.observations.critic, self.observations.system_state):
            if getattr(group, "base_lin_vel", None) is not None:
                group.base_lin_vel.clip = (-10.0, 10.0)
            if getattr(group, "base_ang_vel", None) is not None:
                group.base_ang_vel.clip = (-10.0, 10.0)
            if getattr(group, "joint_vel", None) is not None:
                group.joint_vel.clip = (-30.0, 30.0)
            if getattr(group, "actions", None) is not None:
                group.actions.clip = (-5.0, 5.0)

        # no ground under an outer scan column is a drop, not a sensor fault
        self.terminations.invalid_height_scan = None

        # spawn spread along the staircase: half the batch starts on treads
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

        # climb reward capped in rate (an uncapped riser bonus turned the gait into a sprint)
        self.rewards.spiral_riser_bonus.weight = _RISER_BONUS
        self.rewards.spiral_riser_bonus.params["min_interval_steps"] = _BONUS_MIN_STEPS
        self.actions.arm_swing.period = _CLOCK_PERIOD
        # knee loop gain
        knee_scale = dict(self.actions.joint_pos.scale)
        knee_scale["Knee_.*"] = _KNEE_SCALE
        self.actions.joint_pos.scale = knee_scale
        if _ACT_RANGE:
            # scale so |a| = _ACT_RANGE reaches each window's far edge (the reward is flat in the action outside it)
            self.actions.joint_pos.scale = {
                k: max(hi - off, off - lo) / _ACT_RANGE for k, (lo, hi, off) in _JOINT_WINDOW.items()}
            self.actions.joint_pos.clip = {k: (lo, hi) for k, (lo, hi, off) in _JOINT_WINDOW.items()}
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
        if _SAT:
            self.rewards.action_saturation = RewTerm(
                func=gait.action_saturation_l2,
                weight=_SAT,
                params={"threshold": _SAT_THRESHOLD},
            )
        if _BACKKICK:
            # swing leg hanging behind the body
            self.rewards.feet_swing_behind = RewTerm(
                func=gait.feet_swing_behind_l2,
                weight=_BACKKICK,
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE,
                        "allowance": _BACKKICK_ALLOW},
            )
        if _SHANK_BACK:
            # shank rotating backwards
            self.rewards.shank_backward_rotation = RewTerm(
                func=gait.shank_backward_rotation_l2,
                weight=_SHANK_BACK,
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": SHANK_CFG},
            )
        if _MISMATCH_WEIGHT:
            # feet must match the clock's stance/swing state
            self.rewards.contact_mismatch = RewTerm(
                func=gait.contact_mismatch_l2,
                weight=-_MISMATCH_WEIGHT,
                params={"sensor_cfg": FEET_CONTACT, "period": _CLOCK_PERIOD},
            )
        if _TREAD_BONUS:
            # a footfall onto a tread above the other foot pays
            self.rewards.tread_step = RewTerm(
                func=spiral_mdp.tread_step_bonus,
                weight=_TREAD_BONUS,
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE,
                        "riser": float(self.spiral_geometry["riser"])},
            )
        self.rewards.ang_vel_xy_l2.weight = _ANG_VEL_XY
        self.rewards.flat_orientation_l2.weight = _FLAT_ORIENT

        # landing quality: the inherited feet_impact reads terrain-filtered sensors and never fired here
        self.rewards.feet_impact = RewTerm(
            func=course_mdp.feet_impact_contact,
            weight=_FEET_IMPACT,
            params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE},
        )
        # plant, don't skate
        self.rewards.feet_slide.weight = _FEET_SLIDE
        # leg shake and a loaded foot that is still moving
        self.rewards.action_rate_l2.weight = _ACTION_RATE
        self.rewards.dof_acc_l2.weight = _DOF_ACC
        if _ACTION_L2:
            self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=_ACTION_L2)
        # knee relay: charge the unclipped target beyond the window
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

        # turn the swing shaping on: its gates compare against the command norm, which equalled the shared threshold
        self.rewards.feet_air_time.params["min_command_norm"] = _MIN_CMD
        self.rewards.knee_swing_flexion.params["move_threshold"] = _MIN_CMD
        if _SWING_TARGET:
            # swing-height target from the spiral's own band (riser + margin), replacing the clearance bonus
            self.rewards.foot_swing_clearance = None
            self.rewards.feet_swing_height = RewTerm(
                func=gait.feet_swing_height_relative,
                weight=_SWING_TARGET,
                params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET_POSE,
                        "target": self.spiral_band},
            )
        else:
            self.rewards.foot_swing_clearance.params["min_command_norm"] = _MIN_CMD
        self.commands.base_velocity.ranges.lin_vel_x = (_CMD_SPEED, _CMD_SPEED)
        if _PROGRESS_WEIGHT != 5.0 or _BONUS_MIN_STEPS > 0:
            # the dense climb terms stop paying above the commanded path speed
            self.rewards.spiral_progress.weight = _PROGRESS_WEIGHT
            self.rewards.spiral_progress.params["cap"] = _CMD_SPEED
            self.rewards.spiral_path_progress.params["cap"] = _CMD_SPEED

        # keep a foot on the ground (flight penalty)
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

        # one robot, one staircase: extra robots would overlap at env_spacing 0
        self.scene.num_envs = 1
        self.scene.env_spacing = 0.0
        self.episode_length_s = 120.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.terminations.invalid_height_scan = None
        self.terminations.terrain_out_of_bounds = None
        self.commands.base_velocity.ranges.lin_vel_x = (_CMD_SPEED, _CMD_SPEED)
        # spawn on the marker for a repeatable visual
        self.events.reset_base.params["slots"] = (self.spiral_marker_slot,)
        self.events.reset_base.params["xy_jitter"] = 0.0
        self.events.reset_base.params["yaw_jitter"] = 0.0

        # fixed wide shot: asset_root tracks position only, so a follow offset would end up inside the turn above
        from isaaclab.envs import ViewerCfg

        def _vec(name: str, default: str) -> tuple[float, float, float]:
            return tuple(float(v) for v in os.environ.get(name, default).split(","))

        self.viewer = ViewerCfg(
            eye=_vec("SPIRAL_CAM_EYE", "7.5,-8.0,5.6"),
            lookat=_vec("SPIRAL_CAM_LOOKAT", "1.0,0.0,2.4"),
            origin_type="world",
        )

        # freeze the predictor and draw what the actor sees
        for group in (self.observations.policy, self.observations.critic):
            group.foothold_scan.params.update(train_fpm=False, force_model=True)
        self.observations.policy.foothold_scan.params["show_patches"] = (
            os.environ.get("SHOW_FOOTHOLDS", "0") == "1"
        )
