"""Two-fighter boxing arena for Wato on mjlab.

Importing this package registers:

  Mjlab-Boxing-Arena-Wato   red vs blue in a ring, no task yet (scene only)
  Mjlab-Boxing-Fight-Wato   fight training: move-picking policy vs a curriculum opponent
"""

from mjlab.tasks.registry import register_mjlab_task

from humanoid_boxing.arena_env_cfg import boxing_arena_env_cfg
from humanoid_tracking.rl_cfg import wato_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Boxing-Arena-Wato",
  env_cfg=boxing_arena_env_cfg(),
  play_env_cfg=boxing_arena_env_cfg(),
  rl_cfg=wato_tracking_ppo_runner_cfg(),
)

# fight training: red's fight policy picks moves, frozen move skills carry them
# out; blue holds, moves on a script, or self-plays (humanoid_boxing/fight_env_cfg.py)
from humanoid_boxing.fight_env_cfg import boxing_fight_env_cfg  # noqa: E402
from humanoid_boxing.rl_cfg import FightRunner, wato_fight_ppo_runner_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-Boxing-Fight-Wato",
  env_cfg=boxing_fight_env_cfg(),
  play_env_cfg=boxing_fight_env_cfg(),
  rl_cfg=wato_fight_ppo_runner_cfg(),
  runner_cls=FightRunner,
)
