"""Wato whole-body motion tracking on mjlab (MuJoCo Warp).

Importing this package registers:

  Mjlab-Tracking-Flat-Wato                       BeyondMimic tracking
  Mjlab-Tracking-Flat-Wato-No-State-Estimation   same, without base position /
                                                 linear velocity in the actor obs
  Mjlab-Tracking-Flat-Wato-Power                 stage 2: tracking plus a real
                                                 target (2-40 kg) and power /
                                                 balance terms around punches
  Mjlab-Tracking-Flat-Wato-Footwork              the power task without punch
                                                 rewards: bigger steps with
                                                 balance, for a separate
                                                 footwork policy
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from humanoid_tracking.env_cfg import wato_flat_tracking_env_cfg, wato_power_env_cfg
from humanoid_tracking.rl_cfg import wato_power_ppo_runner_cfg, wato_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Wato",
  env_cfg=wato_flat_tracking_env_cfg(),
  play_env_cfg=wato_flat_tracking_env_cfg(play=True),
  rl_cfg=wato_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Wato-No-State-Estimation",
  env_cfg=wato_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=wato_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=wato_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Wato-Power",
  env_cfg=wato_power_env_cfg(),
  play_env_cfg=wato_power_env_cfg(play=True),
  rl_cfg=wato_power_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Wato-Footwork",
  env_cfg=wato_power_env_cfg(footwork_only=True),
  play_env_cfg=wato_power_env_cfg(play=True, footwork_only=True),
  rl_cfg=wato_power_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
