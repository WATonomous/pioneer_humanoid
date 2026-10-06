"""Wato whole-body motion tracking on mjlab (MuJoCo Warp).

Importing this package registers:

  Mjlab-Tracking-Flat-Wato                       BeyondMimic tracking
  Mjlab-Tracking-Flat-Wato-No-State-Estimation   same, without base position /
                                                 linear velocity in the actor obs
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from wato_tracking.env_cfg import wato_flat_tracking_env_cfg
from wato_tracking.rl_cfg import wato_tracking_ppo_runner_cfg

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
