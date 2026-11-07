"""Pioneer whole-body motion tracking on mjlab (MuJoCo Warp).

Importing this package registers:

  Mjlab-Tracking-Flat-Pioneer                       BeyondMimic tracking
  Mjlab-Tracking-Flat-Pioneer-No-State-Estimation   same, without base position /
                                                 linear velocity in the actor obs
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from humanoid_tracking.env_cfg import flat_tracking_env_cfg
from humanoid_tracking.rl_cfg import tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pioneer",
  env_cfg=flat_tracking_env_cfg(),
  play_env_cfg=flat_tracking_env_cfg(play=True),
  rl_cfg=tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pioneer-No-State-Estimation",
  env_cfg=flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
