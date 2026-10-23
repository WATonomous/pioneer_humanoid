"""Measure each punch's momentum baseline for the power task's reward.

Runs a policy (normally the tracking policy, before any power training) in
Mjlab-Tracking-Flat-Wato-Power and records, for every punch, the peak
momentum it gives the target (along the punch's line) and the target's mass.
Momentum grows with the target's mass, so the baseline is stored per punch
as the median momentum in log-spaced mass bins. The power reward then scores
each punch against its own baseline (1.0 = as good as this policy), so a jab
improving by half counts as much as a rear hook improving by half.

Writes <motion file>_baseline.yaml next to the motion file, which the power
task loads (humanoid_tracking/power.py).

  uv run scripts/measure_power_baseline.py \\
      --checkpoint-file logs/rsl_rl/wato_tracking/<tracking run>/model_<n>.pt
"""

import contextlib
import io
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import humanoid_tracking  # noqa: E402,F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: E402

from humanoid_tracking import power  # noqa: E402

TASK = "Mjlab-Tracking-Flat-Wato-Power"


def main(
  checkpoint_file: str,
  motion_file: str = "data/motions/punch_training.npz",
  num_envs: int = 1024,
  steps: int = 3000,
  bins: int = 6,
  device: str = "cuda:0",
):
  """Args:
    checkpoint_file: policy to measure (the tracking policy before power training).
    motion_file: training clip with its .yaml impact labels.
    num_envs: parallel robots (each has its own target mass, 2-40 kg).
    steps: env steps to run (50 per second).
    bins: log-spaced target-mass bins per punch.
    device: torch / MuJoCo Warp device.
  """
  with contextlib.redirect_stdout(io.StringIO()):
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = num_envs
    cfg.commands["motion"].motion_file = motion_file
    cfg.events.pop("push_robot", None)  # measure the punches, not recoveries from pushes
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))
    runner = MotionTrackingOnPolicyRunner(env, asdict(load_rl_cfg(TASK)), device=device)
    runner.load(checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
  u = env.unwrapped
  cmd = u.command_manager.get_term("motion")
  segments = yaml.safe_load(Path(motion_file).with_suffix(".yaml").read_text())["segments"]
  mass = cmd.target_mass.clone()

  moves, momenta, masses = [], [], []
  obs = env.get_observations()
  peak = torch.zeros(num_envs, device=device)  # raw peak momentum (kg m/s) of the current punch
  for _ in range(steps):
    with torch.no_grad():
      obs, *_ = env.step(policy(obs))
    peak = torch.where(cmd.window(power.MOMENTUM), torch.maximum(peak, cmd.ball_momentum_now()), peak)
    done = cmd.punching & (cmd.offset == power.MOMENTUM[1] + 1)
    for e in done.nonzero().flatten().tolist():
      moves.append(segments[int(cmd.seg[cmd.time_steps[e]])]["move"])
      momenta.append(float(peak[e]))
      masses.append(float(mass[e]))
    peak = torch.where(done | ~cmd.punching, torch.zeros_like(peak), peak)

  moves, momenta, masses = np.array(moves), np.array(momenta), np.array(masses)
  lo, hi = np.log(masses.min()), np.log(masses.max())
  edges = np.linspace(lo, hi, bins + 1)
  centres = (edges[:-1] + edges[1:]) / 2
  out = {"checkpoint": checkpoint_file, "mass_log_bin_centres": [round(float(c), 4) for c in centres], "punches": {}}
  print(f"{'punch':14s} {'n':>5s}  median momentum per mass bin (kg m/s) at {np.round(np.exp(centres), 1).tolist()} kg")
  for move in sorted(set(moves)):
    m = moves == move
    med = []
    for a, b in zip(edges[:-1], edges[1:]):
      sel = m & (np.log(masses) >= a) & (np.log(masses) <= b)
      med.append(float(np.median(momenta[sel])) if sel.sum() >= 5 else np.nan)
    med = np.array(med)
    # fill empty bins from neighbours; never below a small floor
    good = ~np.isnan(med)
    med = np.interp(centres, centres[good], med[good]) if good.any() else np.full(bins, 1.0)
    med = np.maximum(med, 0.2)
    out["punches"][str(move)] = [round(float(x), 4) for x in med]
    print(f"{move:14s} {m.sum():5d}  {np.round(med, 2).tolist()}")

  path = Path(motion_file).with_name(Path(motion_file).stem + "_baseline.yaml")
  path.write_text(yaml.safe_dump(out, sort_keys=False))
  print(f"wrote {path}")


if __name__ == "__main__":
  tyro.cli(main)
