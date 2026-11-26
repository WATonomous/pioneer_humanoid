"""Verify the spiral command is now the tangent-following one."""

import math
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

TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
cfg.observations.policy.enable_corruption = False
env = gym.make(TASK, cfg=cfg).unwrapped
cmd = env.command_manager.get_term("base_velocity")
print("[CMD] term class", type(cmd).__name__, "| cfg class", type(cmd.cfg).__name__)
robot = env.scene["robot"]
env.reset()
zero = torch.zeros(1, env.action_space.shape[-1], device=env.device)
for step in range(120):
    env.step(zero)
    if step % 20 == 0:
        p = robot.data.root_pos_w[0]
        tangent = math.atan2(float(p[1]), float(p[0])) + math.pi / 2
        print(f"[CMD] {step:3d} yaw {math.degrees(float(robot.data.heading_w[0])):7.1f} "
              f"target {math.degrees(float(cmd.heading_target[0])):7.1f} "
              f"tangent {math.degrees(tangent):7.1f} | cmd vx {float(cmd.vel_command_b[0,0]):+.2f} "
              f"wz {float(cmd.vel_command_b[0,2]):+.2f} | radius {math.hypot(float(p[0]), float(p[1])):.2f} "
              f"z {float(p[2]):.2f}")
print("[CMD] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
