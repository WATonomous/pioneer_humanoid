"""Verify the 3 s clock, the foot-land tread bonus and the mismatch term."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "src/simulation/humanoid_rl/humanoid_rl/scripts"))
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app
import gymnasium as gym, humanoid_rl_tasks, torch
from isaaclab_tasks.utils import parse_env_cfg
TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=3)
print("[GT] clock period", cfg.observations.policy.gait_clock.params)
print("[GT] arm period", cfg.actions.arm_swing.period)
print("[GT] mismatch", cfg.rewards.contact_mismatch.weight, "tread", cfg.rewards.tread_step.weight)
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()
a = torch.zeros(3, env.action_space.shape[-1], device=env.device)
for _ in range(80):
    env.step(a)
s = env.reward_manager._episode_sums
print("[GT] sums: mismatch %.3f tread %.3f (80 stationary steps)" % (float(s["contact_mismatch"].mean()), float(s["tread_step"].mean())))
obs = env.observation_manager.compute()["policy"]
print("[GT] policy obs", tuple(obs.shape))
print("[GT] OK")
simulation_app.close()
