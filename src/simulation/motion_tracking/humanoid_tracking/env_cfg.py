"""Wato flat tracking task on mjlab's BeyondMimic port
(mjlab.tasks.tracking), mirroring the Isaac Lab WatoFlatEnvCfg in
WATonomous/Humanoid_motion_tracking (tasks/tracking/config/wato/flat_env_cfg.py).

Differences from the Isaac task:
  * no self-collision cost: only the feet have colliders (Isaac also runs
    with self-collisions off)
  * no undesired-contacts reward: mjlab's base tracking task has none
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from humanoid_tracking.robot import FEET, WATO_ACTION_SCALE, WRISTS, get_wato_robot_cfg

# Same role as the G1 list: pelvis/torso, hips, knees, feet, shoulders, elbows, wrists.
TRACKED_BODIES = (
  "base_link",  # pelvis + torso (no waist joints)
  "left_hip_r_1",
  "left_calf_1",
  "left_foot_1",
  "right_hip_r_1",
  "right_calf_1",
  "right_foot_1",
  "Mirrorlink2__1__1",  # left shoulder roll link
  "Mirrorlink4__1__1",  # left elbow bend link
  "Mirrorlink6__1__1",  # left wrist link
  "link2__1__1",  # right shoulder roll link
  "link4__1__1",  # right elbow bend link
  "link6__1__1",  # right wrist link
)


def wato_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_wato_robot_cfg()}
  cfg.scene.sensors = ()
  cfg.rewards.pop("self_collisions")

  # Tuning on the boxing moves clip (logs/job/experiments.md).
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.1
  cfg.rewards["action_rate_l2"].weight = -0.2
  cfg.rewards["motion_body_ori"].params["std"] = 0.2
  cfg.rewards["motion_body_lin_vel"].params["std"] = 0.5
  cfg.rewards["motion_body_ang_vel"].params["std"] = 1.5

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = WATO_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = TRACKED_BODIES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot_1_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # terminate if feet/hands drift too far in z from the reference
  cfg.terminations["ee_body_pos"].params["body_names"] = FEET + WRISTS

  cfg.viewer.body_name = "base_link"

  if not has_state_estimation:
    cfg.observations["actor"] = ObservationGroupCfg(
      terms={
        k: v
        for k, v in cfg.observations["actor"].terms.items()
        if k not in ("motion_anchor_pos_b", "base_lin_vel")
      },
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg


# Stage 2 (punch power): final weights of the power and balance terms, ramped
# in over POWER_RAMP_STEPS env steps (24 per PPO iteration). Tracking stays the
# largest reward; balance terms outweigh the power terms.
POWER_WEIGHTS = {
  # priorities: land square, move the target (real momentum), keep balance
  "glove_on_target": 3.0,
  "ball_momentum": 3.0,  # x streak scale (power.py)
  # one step per good punch, value = the streak; x0.02 s per step, so a good
  # punch pays 0.5 x streak (6 in a row: 0.5 x 21 = 10.5)
  "streak_bonus": 25.0,
  # helpers (glove speed itself is not rewarded)
  "retract": 0.3,  # hand back to guard after the hit
  "hips_lead": 0.3,
  "weight_transfer": 0.2,
  # footwork: cover more ground per step than the clip
  "step_progress": 1.0,
}
BALANCE_WEIGHTS = {
  "com_over_feet": 0.5,
  "feet_off_ground": -1.0,
  "feet_slide": -2.0,
  "tilt_excess": -5.0,
  "steady_after_impact": -0.2,
  "steady_after_step": -0.2,
}
POWER_RAMP_STEPS = 24_000  # ~1000 iterations
TARGET_MASS_RANGE = (2.0, 40.0)  # kg, log-uniform per env (sampled once, at startup)


# footwork-only policy: of the power terms, only these are kept
FOOTWORK_POWER_TERMS = ("step_progress",)


def wato_power_env_cfg(play: bool = False, footwork_only: bool = False) -> ManagerBasedRlEnvCfg:
  """Tracking + a real target and power / balance terms around each punch
  (humanoid_tracking/power.py). The motion file must have a .yaml with impact
  labels next to it (scripts/build_move_library.py).

  footwork_only: no punch rewards (only FOOTWORK_POWER_TERMS and balance), for
  a separate footwork policy trained on the footwork clip; the punch policy is
  left untouched."""
  import dataclasses
  import math

  from mjlab.entity import EntityCfg
  from mjlab.envs.mdp import dr
  from mjlab.managers.curriculum_manager import CurriculumTermCfg
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  from humanoid_tracking import power
  from humanoid_tracking.target_ball import GLOVE_HIT, TARGET_MASS, free_ball_spec

  cfg = wato_flat_tracking_env_cfg(play=play)

  # gloves on the robot (touch only the target) and the target itself
  robot = cfg.scene.entities["robot"]
  base_spec_fn = robot.spec_fn

  def spec_with_gloves():
    from humanoid_boxing.fighters import add_hitboxes  # lazy: importing humanoid_boxing registers the arena

    spec = base_spec_fn()
    add_hitboxes(spec, "red")
    return spec

  robot.spec_fn = spec_with_gloves
  robot.collisions = (*robot.collisions, GLOVE_HIT)
  cfg.scene.entities["target"] = EntityCfg(spec_fn=free_ball_spec)
  # mass e^(2 alpha) x TARGET_MASS: alpha uniform -> mass log-uniform over TARGET_MASS_RANGE
  lo, hi = (math.log(m / TARGET_MASS) / 2 for m in TARGET_MASS_RANGE)
  cfg.events["target_mass"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={"asset_cfg": SceneEntityCfg("target", body_names=("target",)), "alpha_range": (lo, hi)},
  )

  # the command that places and holds the target
  old = cfg.commands["motion"]
  cfg.commands["motion"] = power.PowerMotionCommandCfg(**{f.name: getattr(old, f.name) for f in dataclasses.fields(old)})

  # tracking terms with the punch-window allowances (same stds, looser in the
  # free window: the pelvis, legs and punching arm pick their own mechanics)
  r = cfg.rewards
  r["motion_global_root_pos"].func = power.root_pos_free
  r["motion_global_root_ori"].func = power.root_ori_loose
  r["motion_body_pos"].func = power.body_pos_feet_loose
  r["motion_body_ori"].func = power.body_ori_feet_loose
  r["motion_body_lin_vel"].func = power.body_lin_vel_no_speed_cap
  r["motion_body_ang_vel"].func = power.body_ang_vel_no_turn_cap
  for name in ("motion_global_root_pos", "motion_global_root_ori", "motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"):
    r[name].params.pop("body_names", None)

  # power + balance terms, ramped in from 0
  power_terms = {k: w for k, w in POWER_WEIGHTS.items() if not footwork_only or k in FOOTWORK_POWER_TERMS}
  for name, weight in {**power_terms, **BALANCE_WEIGHTS}.items():
    r[name] = RewardTermCfg(func=getattr(power, name), weight=0.0, params={"command_name": "motion"})
    cfg.curriculum[f"ramp_{name}"] = CurriculumTermCfg(
      func=power.ramp_from_run_start,
      params={
        "reward_name": name,
        "stages": [{"step": int(POWER_RAMP_STEPS * f), "weight": weight * f} for f in (0.0, 0.25, 0.5, 0.75, 1.0)],
      },
    )

  # feet may be 35 cm off in height during a punch (hands keep 25 cm)
  ee = cfg.terminations["ee_body_pos"]
  ee.func = power.bad_body_z_feet_loose
  ee.params = {
    "command_name": "motion",
    "threshold": ee.params["threshold"],
    "feet_threshold_loose": 0.35,
    "body_names": tuple(ee.params["body_names"]),
  }
  return cfg
