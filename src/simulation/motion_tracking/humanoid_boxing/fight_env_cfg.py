"""The fight task: red's fight policy picks moves; both fighters carry them out
with frozen move skills (humanoid_boxing/skills.py). Blue is the opponent: it
holds its guard, moves on a script, or plays a frozen snapshot of red's own
fight policy (self-play), by an opponent curriculum over the run:

  first 20%   hold (stands in guard: learn range and landing punches, knockdowns)
  next 15%    scripted (random footwork, defence and punches)
  last 65%    self-play (mostly), with some scripted and hold mixed in

The stages are relative to the run's start, so a resumed run starts again
with the still target.

The arena (ring, gloves and hitboxes, camera + ultrasonic perception, fight
rewards and round endings) is humanoid_boxing.arena_env_cfg's. Motor gains are the
tracking task's (gain_scale 1), which the skills were trained with.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from humanoid_boxing import impact
from humanoid_boxing import perception as vision
from humanoid_boxing import skills
from humanoid_boxing.arena_env_cfg import boxing_arena_env_cfg

PUNCH_CHECKPOINT = "logs/rsl_rl/wato_tracking/2026-10-17_00-01-11_power_v8/model_22000.pt"
FOOTWORK_CHECKPOINT = "logs/rsl_rl/wato_tracking/2026-10-17_00-34-56_footwork_v1/model_23500.pt"

# (end of stage as a fraction of the run, (p_hold, p_scripted, p_selfplay))
OPPONENT_STAGES = [
  (0.20, (1.0, 0.0, 0.0)),
  (0.35, (0.2, 0.8, 0.0)),
  (1.00, (0.05, 0.15, 0.80)),
]
# The stages restart with every resume. To resume a policy that already boxes,
# use a shorter schedule instead, e.g. self-play only:
#   OPPONENT_STAGES = [(1.00, (0.05, 0.15, 0.80))]

# Punches score per clean impact (humanoid_boxing/impact.py): a punch move landing
# on head or torso, worth more to the head, faster, and as a counter. Balance
# damage is paid per clean punch, on its peak effect, so leaning on the
# opponent or tapping it scores nothing. These replace the arena's per-step
# glove-force terms (clean_hits, hits_taken) and its bell points, which counted
# any glove contact (walking into the opponent's guard scored).
PUNCH_REWARDS = {
  "clean_punches": 15.0,  # (10 made self-play exchanges ~break-even against overspeed: fight_v7f drifted passive) x zone (head 1, liver/ribs 0.8, solar plexus 0.7, chest 0.4) x glove speed / 1 m/s (<= 1.5) x 1.5 counter x peak force / 50 N (<= 2)
  "clean_punches_taken": -5.0,
  "punch_imbalance": 1.0,  # per degree of tilt past stance + 5 deg
  "punch_driven_back": 10.0,  # per metre pushed away
  "punch_stagger": 20.0,  # tilted past 35 deg
}
# Range shaping (humanoid_boxing/impact.py): the still stage's punches only land
# from 0.8-1.05 m, which random moves from the 1.4 m start rarely find, so
# fight_v5 settled on stepping forward into blue's guard (no punches).
RANGE_REWARDS = {
  "in_range_vs_still": 0.02,  # per step in 0.8-1.05 m, only while blue holds its guard (<= 20 per round)
  "in_range_vs_moving": 0.02,  # the same against a scripted / self-play blue (self-play stood off out of range)
  "punch_from_range": 0.3,  # per punch move started from 0.8-1.05 m (a clean one is worth ~3-8)
  "crowding": -0.01,  # per step inside 0.75 m (smothered: punches arrive at ~0 m/s)
}
STEPS_PER_ITERATION = 24

# The arena's realism penalties stay, so the fight policy learns smooth,
# balanced combos (which moves to chain, and when) rather than ones that jerk,
# slip or saturate motors. They are scaled by REALISM_SCALE: at full weight
# they outweighed clean hits ~5:1 for a normal step-in-and-punch round, which
# would teach it to stand in guard instead of boxing.
REALISM_PENALTIES = ("overspeed", "torque_saturation", "joint_limits", "jerkiness", "foot_slip")
REALISM_SCALE = 0.25


def boxing_fight_env_cfg(
  num_envs: int = 1,
  iterations: int = 6000,
  punch_checkpoint: str = PUNCH_CHECKPOINT,
  footwork_checkpoint: str = FOOTWORK_CHECKPOINT,
  round_s: float = 20.0,
) -> ManagerBasedRlEnvCfg:
  """iterations: planned PPO iterations of the run (sets the curriculum's stages)."""
  cfg = boxing_arena_env_cfg(num_envs=num_envs, gain_scale=1.0, learner="red", round_s=round_s)

  for name in REALISM_PENALTIES:
    cfg.rewards[name].weight *= REALISM_SCALE

  for name in ("clean_hits", "hits_taken", "time_up_points"):
    cfg.rewards.pop(name)
  for name, w in {**PUNCH_REWARDS, **RANGE_REWARDS}.items():
    cfg.rewards[name] = RewardTermCfg(func=getattr(impact, name), weight=w, params={"team": "red"})
  cfg.rewards["time_up_clean"] = RewardTermCfg(func=impact.time_up_clean, weight=10.0, params={"team": "red"})
  cfg.scene.sensors = (*cfg.scene.sensors, *impact.landing_sensors())
  cfg.events["reset_impacts"] = EventTermCfg(func=impact.reset_impacts, mode="reset")

  cfg.actions = {
    team: skills.FighterSkillsCfg(
      entity_name=team,
      role="learner" if team == "red" else "opponent",
      punch_checkpoint=punch_checkpoint,
      footwork_checkpoint=footwork_checkpoint,
    )
    for team in ("red", "blue")
  }

  pcfg = vision.PerceptionCfg()
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={"fight": ObservationTermCfg(func=skills.fight_actor_obs, params={"team": "red"})},
      concatenate_terms=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "fight": ObservationTermCfg(func=skills.fight_actor_obs, params={"team": "red"}),
        "opponent_true": ObservationTermCfg(func=vision.opponent_true, params={"team": "red", "cfg": pcfg}),
        "opponent_move": ObservationTermCfg(func=skills.opponent_move, params={"team": "red"}),
        "opponent_state": ObservationTermCfg(func=skills.own_state, params={"team": "blue"}),
      },
      concatenate_terms=True,
    ),
  }

  cfg.curriculum = {
    "opponent": CurriculumTermCfg(
      func=skills.opponent_curriculum,
      params={"team": "blue", "total_steps": iterations * STEPS_PER_ITERATION, "stages": OPPONENT_STAGES},
    ),
    # not a curriculum: logs clean punches per round, by zone (Curriculum/punches/...)
    "punches": CurriculumTermCfg(func=impact.punch_stats, params={"team": "red"}),
  }
  return cfg
