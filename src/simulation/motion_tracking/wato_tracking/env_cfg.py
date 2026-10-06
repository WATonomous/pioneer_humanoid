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
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from wato_tracking.robot import FEET, WATO_ACTION_SCALE, WRISTS, get_wato_robot_cfg

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
