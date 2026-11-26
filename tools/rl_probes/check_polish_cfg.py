"""Verify the polish reward set builds and the new terms are registered."""

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

TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=2)
r = cfg.rewards
print("[PC] posture", r.flat_orientation_l2.weight)
print("[PC] flight", r.feet_flight.weight, r.feet_flight.params["sensor_cfg"].body_names)
print("[PC] radial", r.spiral_radial_error.weight, "face", r.spiral_face_tangent.weight)
print("[PC] overspeed", r.speed_over_command_l2.weight, "caps", r.spiral_progress.params.get("cap"))
print("[PC] bonus", r.spiral_riser_bonus.weight, r.spiral_riser_bonus.params)
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()
a = torch.zeros(2, env.action_space.shape[-1], device=env.device)
for _ in range(40):
    env.step(a)
sums = env.reward_manager._episode_sums
print("[PC] stationary: flight", round(float(sums["feet_flight"].mean()), 2),
      "| posture", round(float(sums["flat_orientation_l2"].mean()), 2),
      "| bonus", round(float(sums["spiral_riser_bonus"].mean()), 2))
print("[PC] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
