"""Per-step trace of command vs motion on the spiral (model_900, PLAY)."""

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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import cli_args  # noqa: E402

TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0"
CKPT = "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_900.pt"

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
cfg.observations.policy.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped


class A:
    pass


a = A()
for k, v in dict(task=TASK, num_envs=1, checkpoint=None, device="cuda:0", distributed=False,
                 seed=3, max_iterations=None, resume=False, run_name=None, logger=None,
                 load_run=None, load_checkpoint=None, video=False, video_length=200).items():
    setattr(a, k, v)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, a)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)
robot = u.scene["robot"]
cmd = u.command_manager.get_term("base_velocity")

obs = env.get_observations()
print("[TR] step  cmd_vx cmd_wz | vx    vy    wz   | yaw°  head_target° head° | radial  z")
with torch.inference_mode():
    for step in range(400):
        obs, *_ = env.step(policy(obs))
        if step % 20 == 0:
            p = robot.data.root_pos_w[0]
            vb = robot.data.root_lin_vel_b[0]
            r = math.hypot(float(p[0]), float(p[1]))
            print(f"[TR] {step:4d}  {float(cmd.vel_command_b[0, 0]):.2f}   {float(cmd.vel_command_b[0, 2]):+.2f} | "
                  f"{float(vb[0]):+.2f} {float(vb[1]):+.2f} {float(robot.data.root_ang_vel_b[0, 2]):+.2f} | "
                  f"{math.degrees(float(robot.data.heading_w[0])):6.1f} "
                  f"{math.degrees(float(cmd.heading_target[0])):8.1f} "
                  f"{math.degrees(math.atan2(float(p[1]), float(p[0])) + math.pi / 2):8.1f} | "
                  f"{r:.2f} {float(p[2]):.2f}")
print("[TR] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
