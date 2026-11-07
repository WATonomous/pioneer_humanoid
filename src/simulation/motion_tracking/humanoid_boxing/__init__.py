"""Two-fighter boxing arena for the Pioneer humanoid on mjlab.

Importing this package registers:

  Mjlab-Boxing-Arena-Pioneer   red vs blue in a ring, no task yet (scene only)
"""

from mjlab.tasks.registry import register_mjlab_task

from humanoid_boxing.arena_env_cfg import boxing_arena_env_cfg
from humanoid_tracking.rl_cfg import tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Boxing-Arena-Pioneer",
  env_cfg=boxing_arena_env_cfg(),
  play_env_cfg=boxing_arena_env_cfg(),
  rl_cfg=tracking_ppo_runner_cfg(),
)
