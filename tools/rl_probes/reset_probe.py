"""Minimal check: does this env survive being reset twice in one process?

Two probes that reset the environment inside a sweep (smooth_probe, slew_probe)
hung at ~100% CPU with no output, while every probe that resets once
(target_probe, sat_probe, leg_quality) ran normally. This isolates that:
one reset, a few steps, a second reset, a few steps, with unbuffered progress
prints so a stall is visible immediately.
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import cli_args  # noqa: E402

TASK = os.environ.get("TASK_OVERRIDE", "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")
CKPT = os.environ.get("CKPT_OVERRIDE",
                      "/workspace/humanoid/outputs/rl/pioneer_humanoid_spiral_foothold/2026-11-16_22-26-35/model_299.pt")
N = 2

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
print("[R] building env", flush=True)
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
print("[R] env built", flush=True)


class A:
    pass


a = A()
for k, v in dict(task=TASK, num_envs=N, checkpoint=None, device="cuda:0", distributed=False,
                 seed=5, max_iterations=None, resume=False, run_name=None, logger=None,
                 load_run=None, load_checkpoint=None, video=False, video_length=200).items():
    setattr(a, k, v)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, a)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)
print("[R] policy loaded", flush=True)

for cycle in range(2):
    print(f"[R] cycle {cycle}: reset", flush=True)
    obs, _ = env.reset()
    print(f"[R] cycle {cycle}: reset done", flush=True)
    with torch.inference_mode():
        for step in range(60):
            act = policy(obs)
            obs, *_ = env.step(act)
            if step % 20 == 0:
                print(f"[R] cycle {cycle} step {step} |a| {float(act.abs().mean()):.2f}", flush=True)
    print(f"[R] cycle {cycle}: done", flush=True)

print("[R] OK", flush=True)
env.close()
simulation_app.close()
