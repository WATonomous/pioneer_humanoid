# Copyright (c) 2022-2025, The Isaac Lab Project Developers / WATonomous.
# SPDX-License-Identifier: BSD-3-Clause

"""Play a distilled vision student on the push-block task.

Example:
  PYTHONPATH=$(pwd) $ISAACLAB/isaaclab.sh -p $RL_RUNNERS/play_distill_push.py \\
    --task Isaac-Bimanual-Push-Block-Distill-Play-v0 --enable_cameras \\
    --checkpoint outputs/rl/push_distill/<run>/nn/student_1000.pt --num_envs 16
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play distilled push-block student.")
parser.add_argument("--task", type=str, default="Isaac-Bimanual-Push-Block-Distill-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str, required=True, help="Student .pt from distill_push.py")
parser.add_argument("--num_steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import humanoid_rl_tasks  # noqa: F401
from humanoid_rl_tasks.push_block.student_policy import FrameStack, StudentVisionPolicy, student_images
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    student = StudentVisionPolicy(
        proprio_dim=ckpt["proprio_dim"],
        num_actions=ckpt["num_actions"],
        image_shapes={name: tuple(shape) for name, shape in ckpt["image_shapes"].items()},
    ).to(device)
    student.load_state_dict(ckpt["student"])
    student.eval()
    print(f"[INFO] Loaded student from {args_cli.checkpoint}")

    frame_stack = FrameStack(ckpt.get("history", 1))
    prev_dones = None
    obs_dict, _ = env.reset()
    for step in range(args_cli.num_steps):
        with torch.no_grad():
            actions = student(frame_stack(student_images(obs_dict), prev_dones), obs_dict["student"])
        obs_dict, rew, terminated, truncated, _ = env.step(actions)
        prev_dones = terminated | truncated
        if step % 50 == 0:
            print(f"step={step} rew_mean={rew.mean().item():.3f} "
                  f"done={(terminated | truncated).float().mean().item():.2f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
