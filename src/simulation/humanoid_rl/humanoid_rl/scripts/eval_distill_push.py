# Copyright (c) 2022-2025, The Isaac Lab Project Developers / WATonomous.
# SPDX-License-Identifier: BSD-3-Clause

"""Success rate of a distilled student over many episodes, in the training env (same DR, observation noise and render mode).

It reads the same success test the training logs (``Curriculum/success/success_rate``: block at rest on the floor, evaluated on
each episode's final state), with a window big enough to hold every episode, so the number compares directly with the training
curve. Nothing is trained.

Example:
  PYTHONPATH=$(pwd) $ISAACLAB/isaaclab.sh -p $RL_RUNNERS/eval_distill_push.py --headless \\
    --checkpoint outputs/rl/push_distill/<run>/nn/student_125.pt --episodes 1024
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a distilled push-block student.")
parser.add_argument("--task", type=str, default="Isaac-Bimanual-Push-Block-Distill-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=1024, help="Stop after at least this many finished episodes.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import traceback

import gymnasium as gym
import torch

import humanoid_rl_tasks  # noqa: F401
from humanoid_rl_tasks.push_block.student_policy import FrameStack, StudentVisionPolicy, student_images
from isaaclab_tasks.utils import parse_env_cfg


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.curriculum.success.params["window"] = args_cli.episodes + 4 * args_cli.num_envs   # keep every episode
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    device = u.device
    cm = u.curriculum_manager
    logger = cm._term_cfgs[cm._term_names.index("success")].func

    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    student = StudentVisionPolicy(
        proprio_dim=ckpt["proprio_dim"],
        num_actions=ckpt["num_actions"],
        image_shapes={name: tuple(shape) for name, shape in ckpt["image_shapes"].items()},
    ).to(device)
    student.load_state_dict(ckpt["student"])
    student.eval()
    frame_stack = FrameStack(ckpt.get("history", 1))

    obs_dict, _ = env.reset()
    prev_dones = None
    step = 0
    while logger._count < args_cli.episodes:
        with torch.no_grad():
            actions = student(frame_stack(student_images(obs_dict), prev_dones), obs_dict["student"])
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        prev_dones = terminated | truncated
        step += 1
        if step % 250 == 0:
            n = logger._count
            print(f"[eval] step {step}: {n} episodes, success {float(logger._buf[:n].mean()) * 100:.2f}%", flush=True)

    n = logger._count
    k = int(logger._buf[:n].sum().item())
    lo, hi = wilson(k, n)
    print(f"[eval] RESULT {os.path.basename(args_cli.checkpoint)}: {k}/{n} = {100 * k / n:.2f}%  (95% interval {100 * lo:.1f}-{100 * hi:.1f}%)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    os._exit(0)   # Kit otherwise hangs in shutdown holding the GPU
