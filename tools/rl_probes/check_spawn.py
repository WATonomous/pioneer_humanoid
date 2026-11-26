"""Sanity-check the multi-slot course spawn and the push ramp on the TRAIN task.

- resets 64 envs several times and prints where they spawned (should be the six
  gap slots) and whether they are standing;
- steps ~120 control steps and reports how many are still upright;
- prints the push ramp scale from the curriculum log value.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "src/simulation/humanoid_rl/humanoid_rl/scripts"))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_rl_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Locomotion-StairsStraightFoothold-PioneerHumanoid-v0"
N = 64

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.log_dir = "/workspace/humanoid/models/pioneer_humanoid_stairs_foothold"
env = gym.make(TASK, cfg=cfg).unwrapped

robot = env.scene["robot"]
env.reset()
ys = robot.data.root_pos_w[:, 1].cpu()
zs = robot.data.root_pos_w[:, 2].cpu()
print("[SPAWN] y min/max", float(ys.min()), float(ys.max()))
for slot in (-1.549, 1.58, 4.08, 7.08, 10.68, 14.83):
    n = int(((ys - slot).abs() < 0.12).sum())
    print(f"[SPAWN]   slot {slot:+.2f}: {n}/{N} envs")
print(f"[SPAWN] z min/max {float(zs.min()):.2f} {float(zs.max()):.2f}")

for _ in range(120):
    env.step(torch.zeros(N, env.action_space.shape[-1], device=env.device))
zs = robot.data.root_pos_w[:, 2]
upright = int((zs > 0.5).sum())
print(f"[SPAWN] after 120 steps: {upright}/{N} upright, z min {float(zs.min()):.2f}")

cur = env.curriculum_manager
print(f"[SPAWN] curriculum terms {list(cur.active_terms)}")
print("[SPAWN] push velocity range now:",
      env.event_manager.get_term_cfg("push_robot").params.get("velocity_range"))
print("[SPAWN] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
