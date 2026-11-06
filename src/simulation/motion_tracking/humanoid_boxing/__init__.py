"""Two-fighter boxing arena for the Pioneer humanoid on mjlab.

Importing this package registers:

  Mjlab-Boxing-Arena-Pioneer   red vs blue in a ring, no task yet (scene only)
  Mjlab-Boxing-Fight-Pioneer   fight training: move-picking policy vs a curriculum opponent
  Mjlab-Boxing-Fight-Pioneer-Still     phase 1: the still target only (land and mix punches)
  Mjlab-Boxing-Fight-Pioneer-SelfPlay  phase 2, resumed from phase 1: scripted warm-up, then self-play
  Mjlab-Boxing-Fight-Pioneer-SelfPlayContinue  phase 2 resumed from a phase-2 checkpoint (no warm-up)
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

# fight training: red's fight policy picks moves, frozen move skills carry them
# out; blue holds, moves on a script, or self-plays (humanoid_boxing/fight_env_cfg.py)
from humanoid_boxing import fight_env_cfg as _f  # noqa: E402
from humanoid_boxing.fight_env_cfg import boxing_fight_env_cfg  # noqa: E402
from humanoid_boxing.rl_cfg import FightRunner, fight_ppo_runner_cfg  # noqa: E402

register_mjlab_task(
  task_id="Mjlab-Boxing-Fight-Pioneer",
  env_cfg=boxing_fight_env_cfg(),
  play_env_cfg=boxing_fight_env_cfg(),
  rl_cfg=fight_ppo_runner_cfg(),
  runner_cls=FightRunner,
)

_still = lambda: boxing_fight_env_cfg(iterations=_f.STILL_ITERATIONS, stages=_f.STILL_STAGES)  # noqa: E731
register_mjlab_task(
  task_id="Mjlab-Boxing-Fight-Pioneer-Still",
  env_cfg=_still(),
  play_env_cfg=_still(),
  rl_cfg=fight_ppo_runner_cfg(),
  runner_cls=FightRunner,
)

_selfplay = lambda stages=_f.SELFPLAY_STAGES: boxing_fight_env_cfg(  # noqa: E731
  iterations=_f.SELFPLAY_ITERATIONS,
  stages=stages,
  blocked_ramp=((0, 1.0),),  # already ramped in during the still phase
  taken_weight=-7.5,  # trading at the same spot pays, a worse trade loses (fight_env_cfg.py)
)


def _selfplay_play():
  """Play / video for the self-play tasks: both fighters are the loaded checkpoint
  (blue plays it from its side), every round. Training's opponent schedule
  would otherwise sit at its start in play (no iterations): blue still or scripted."""
  cfg = _selfplay([(1.00, (0.0, 0.0, 1.0))])
  cfg.actions["blue"].mode_probs = (0.0, 0.0, 1.0)
  cfg.actions["blue"].mirror = True
  return cfg


register_mjlab_task(
  task_id="Mjlab-Boxing-Fight-Pioneer-SelfPlay",
  env_cfg=_selfplay(),
  play_env_cfg=_selfplay_play(),
  rl_cfg=fight_ppo_runner_cfg(),
  runner_cls=FightRunner,
)

register_mjlab_task(
  task_id="Mjlab-Boxing-Fight-Pioneer-SelfPlayContinue",
  env_cfg=_selfplay(_f.SELFPLAY_CONTINUE_STAGES),
  play_env_cfg=_selfplay_play(),
  rl_cfg=fight_ppo_runner_cfg(),
  runner_cls=FightRunner,
)
