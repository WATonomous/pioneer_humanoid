"""Verify the simplified spiral recipe."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "src/simulation/humanoid_rl/humanoid_rl/scripts"))
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app
import gymnasium as gym, humanoid_rl_tasks, torch
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg("Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-v0", device="cuda:0", num_envs=2)
r = cfg.rewards
print("[R] bonus", r.spiral_riser_bonus.weight, r.spiral_riser_bonus.params)
print("[R] flight", r.feet_flight.weight, r.feet_flight.params["sensor_cfg"].body_names)
print("[R] caps", r.spiral_progress.params.get("cap"), r.spiral_path_progress.params.get("cap"),
      "weights", r.spiral_progress.weight, r.spiral_path_progress.weight)
print("[R] cmd", cfg.commands.base_velocity.ranges.lin_vel_x, "| radial", r.spiral_radial_error.weight,
      "flat", r.flat_orientation_l2.weight, "slide", r.feet_slide.weight)
env = gym.make("Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-v0", cfg=cfg).unwrapped
env.reset()
a = torch.zeros(2, env.action_space.shape[-1], device=env.device)
for _ in range(60):
    env.step(a)
sums = env.reward_manager._episode_sums
print("[R] stationary 60 steps: bonus", round(float(sums["spiral_riser_bonus"].mean()), 2),
      "flight", round(float(sums["feet_flight"].mean()), 2))
print("[R] OK")
simulation_app.close()
