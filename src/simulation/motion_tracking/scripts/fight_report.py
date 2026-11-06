"""Progress report for a fight training run (Mjlab-Boxing-Fight-Pioneer).

  uv run scripts/fight_report.py --run logs/rsl_rl/pioneer_fight/<run>
      the key training metrics over the run (tensorboard log)

  uv run scripts/fight_report.py --run logs/rsl_rl/pioneer_fight/<run> --behaviour-check
      also loads the run's latest checkpoint and plays 256 rounds against the
      still target: time spent in each move, distance kept, clean punches by
      zone and force (needs the GPU for ~1 min)

  ... --behaviour-check --opponent selfplay
      the same against the checkpoint itself (blue plays it from its side):
      both sides' clean punches and points
"""

import contextlib
import glob
import io
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import tyro  # noqa: E402

METRICS = {
  "Train/mean_reward": "mean reward",
  "Train/mean_episode_length": "round length",
  "Curriculum/opponent/hold": "opp: still",
  "Curriculum/opponent/scripted": "opp: scripted",
  "Curriculum/opponent/selfplay": "opp: self-play",
  "Curriculum/punches/clean_per_round": "clean punches",
  "Curriculum/punches/head": "  head",
  "Curriculum/punches/liver_ribs": "  liver/ribs",
  "Curriculum/punches/solar_plexus": "  solar plexus",
  "Curriculum/punches/chest": "  chest",
  "Curriculum/punches/force_per_punch_N": "force/punch N",
  "Curriculum/punches/taken_per_round": "punches taken",
  "Curriculum/punches/points_per_round": "points dealt",
  "Curriculum/punches/points_taken_per_round": "points taken",
  "Curriculum/punches/mix_per_punch": "mix-up x",
  "Curriculum/punches/blocked_per_round": "blocked",
  "Curriculum/punches/type/jab": "  jab",
  "Curriculum/punches/type/cross": "  cross",
  "Curriculum/punches/type/lead_hook": "  lead hook",
  "Curriculum/punches/type/rear_hook": "  rear hook",
  "Curriculum/punches/type/lead_uppercut": "  lead upper",
  "Curriculum/punches/type/rear_uppercut": "  rear upper",
  "Curriculum/punches/guard_share": "guard share",
  "Curriculum/punches/vs_still/clean_per_round": "still: clean",
  "Curriculum/punches/vs_selfplay/clean_per_round": "self: clean",
  "Curriculum/punches/vs_selfplay/taken_per_round": "self: taken",
  "Curriculum/punches/vs_selfplay/defence_rate": "self: defence",
  "Curriculum/punches/vs_selfplay/blocks_made_per_round": "self: blocks",
  "Curriculum/punches/vs_selfplay/evaded_per_round": "self: evaded",
  "Curriculum/punches/vs_selfplay/guard_share": "self: guard",
  "Curriculum/punches/vs_selfplay/mix_per_punch": "self: mix-up",
  "Episode_Reward/punch_stagger": "stagger",
  "Episode_Reward/knockdown": "knockdown",
  "Episode_Reward/knocked_down": "knocked down",
  "Episode_Reward/stability": "stability",
  "Episode_Reward/rope_contact": "rope contact",
  "Episode_Reward/foot_slip": "foot slip",
  "Episode_Reward/overspeed": "overspeed",
  "Policy/mean_std": "action std",
  "Loss/entropy": "entropy",
}


def curves(run: Path, columns: int = 9) -> str:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  files = sorted(glob.glob(str(run / "events.out.tfevents.*")))
  if not files:
    return f"no tensorboard log in {run}"
  ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
  ea.Reload()
  tags = ea.Tags()["scalars"]
  ev = {t: {x.step: x.value for x in ea.Scalars(t)} for t in METRICS if t in tags}
  if not ev:
    return "no metrics logged yet"
  first = min(min(d) for d in ev.values())
  last = max(max(d) for d in ev.values())
  pts = sorted(set(np.linspace(first, last, columns).astype(int)))

  def at(d, p):  # mean over +-5 iterations
    v = [d[k] for k in d if abs(k - p) <= 5]
    return f"{np.mean(v):8.2f}" if v else f"{'-':>8s}"

  lines = [f"iterations {first} -> {last}", f"{'':16s}" + "".join(f"{p:>8d}" for p in pts)]
  for t, label in METRICS.items():
    if t in ev:
      lines.append(f"{label:16s}" + "".join(at(ev[t], p) for p in pts))
  lines.append("(Episode_Reward/* are points per second of the round: x20 for points per round)")
  return "\n".join(lines)


def behaviour(run: Path, rounds: int = 256, opponent: str = "hold") -> str:
  import torch

  import humanoid_boxing  # noqa: F401
  import humanoid_tracking  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from humanoid_boxing import impact, skills
  from humanoid_boxing.rl_cfg import FightRunner

  ckpts = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  if not ckpts:
    return "no checkpoint yet"
  task = "Mjlab-Boxing-Fight-Pioneer"
  with contextlib.redirect_stdout(io.StringIO()):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = rounds
    cfg.curriculum = {}  # blue holds its guard (a still target)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
    runner = FightRunner(env, asdict(load_rl_cfg(task)), device="cuda:0")
    runner.load(str(ckpts[-1]), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  u = env.unwrapped
  red, me, them = u.action_manager.get_term("red"), u.scene["red"], u.scene["blue"]
  if opponent == "selfplay":  # blue plays this checkpoint from its own side
    blue = u.action_manager.get_term("blue")
    blue.learner_actor = runner.alg.actor
    blue.mode_probs = torch.tensor([0.0, 0.0, 1.0], device="cuda:0")
  obs, _ = env.reset()
  moves, gaps, steps = Counter(), [], 950  # stop before the 1000-step bell resets the counts
  for _ in range(steps):
    with torch.no_grad():
      a = runner.alg.actor(obs, stochastic_output=True)
    obs, *_ = env.step(a)
    moves.update(skills.MOVES[m] for m in red.player.move.tolist())
    gaps.append(float(torch.norm(me.data.root_link_pos_w[:, :2] - them.data.root_link_pos_w[:, :2], dim=-1).mean()))
  st = impact.punch_stats(u, torch.arange(rounds, device="cuda:0"), "red")
  total = rounds * steps
  vs = "vs_selfplay" if opponent == "selfplay" else "vs_still"
  share = ", ".join(f"{m} {c / total * 100:.0f}%" for m, c in moves.most_common(8))
  zones = ", ".join(f"{z} {float(st[z]):.1f}" for z in impact.ZONES)
  return (
    f"checkpoint {ckpts[-1].name}, {rounds} rounds vs " + ("itself" if opponent == "selfplay" else "the still target") + "\n"
    f"time in move: {share}\n"
    f"distance to blue: start {gaps[0]:.2f} m, 5 s {gaps[250]:.2f} m, end {gaps[-1]:.2f} m"
    f" (punches land cleanly from ~0.8-1.05 m; guard gloves touch at ~0.7 m)\n"
    f"clean punches per round: {float(st['clean_per_round']):.1f} ({zones}), "
    f"force per punch {float(st['force_per_punch_N']):.0f} N, mix-up x{float(st['mix_per_punch']):.2f} per punch, "
    f"{float(st['blocked_per_round']):.1f} blocked punches per round\n"
    f"clean by type: " + ", ".join(f"{m} {float(st['type/' + m]):.1f}" for m in skills.PUNCHES) + "\n"
    f"clean by level: " + ", ".join(f"{lv} {float(st['level/' + lv]):.1f}" for lv in impact.LEVELS) + "\n"
    f"punches taken per round: {float(st['taken_per_round']):.1f}; points dealt {float(st['points_per_round']):.1f},"
    f" taken {float(st['points_taken_per_round']):.1f} (x15 / x10 in the reward)\n"
    f"defence: blue threw {float(st[vs + '/opp_thrown_per_round']):.1f} punches per round,"
    f" red blocked {float(st[vs + '/blocks_made_per_round']):.1f},"
    f" evaded {float(st[vs + '/evaded_per_round']):.1f}"
    f" (defence rate {float(st[vs + '/defence_rate']):.0%});"
    f" red in guard {float(st['guard_share']):.0%} of the time"
  )


def main(run: str, behaviour_check: bool = False, opponent: str = "hold"):
  """Args:
    run: the run's log folder, logs/rsl_rl/pioneer_fight/<run>.
    behaviour_check: also play the latest checkpoint.
    opponent: "hold" (the still target) or "selfplay" (the checkpoint itself).
  """
  path = Path(run)
  print(curves(path))
  if behaviour_check:
    print()
    print(behaviour(path, opponent=opponent))


if __name__ == "__main__":
  tyro.cli(main)
