# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a follow-camera video of a stair policy on a *-Play course task.

The default headless viewport never follows the robot, so plain --video clips
show empty ground. This script re-points the viewport camera at env 0's robot
every ``follow_interval`` steps and records via gym's RecordVideo, exiting
after ``video_length`` steps.

Example:
    isaaclab.sh -p record_follow.py \\
        --task Isaac-Velocity-Stairs-PioneerHumanoid-Play-v0 \\
        --checkpoint <model.pt> --video_length 3000 --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record a follow-camera rollout.")
parser.add_argument("--task", type=str, default=None, help="Name of the *-Play task.")
parser.add_argument("--video_length", type=int, default=3000, help="Rollout steps to record (50 Hz).")
parser.add_argument("--follow_interval", type=int, default=50, help="Re-aim camera every N steps.")
parser.add_argument("--seed", type=int, default=None, help="Env seed override.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import humanoid_rl_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed

    if not args_cli.checkpoint or not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(f"--checkpoint is required and must exist (got {args_cli.checkpoint})")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    log_dir = os.path.dirname(args_cli.checkpoint)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=os.path.join(log_dir, "videos", "follow"),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins
    sim = env.unwrapped.sim

    def aim_camera():
        root = (robot.data.root_pos_w[0] - origins[0]).tolist()
        base = origins[0].tolist()
        # World-space anchor: env origin + local root offset.
        cx, cy, cz = base[0] + root[0], base[1] + root[1], base[2] + root[2]
        eye = (cx - 2.5, cy - 2.2, cz + 1.6)
        target = (cx + 1.5, cy, cz + 0.3)
        sim.set_camera_view(eye, target)

    obs = env.get_observations()
    aim_camera()
    timestep = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            runner.alg.policy.reset(dones)
            timestep += 1
            if timestep % 500 == 0:
                root = robot.data.root_pos_w[0] - origins[0]
                tilt = torch.linalg.vector_norm(robot.data.projected_gravity_b[0, :2]).item()
                print(f"[FOLLOW] step {timestep}: root_x={root[0].item():.2f} "
                      f"root_z={root[2].item():.2f} tilt={tilt:.3f}", flush=True)
            if timestep % args_cli.follow_interval == 0:
                aim_camera()
            if timestep == args_cli.video_length:
                break
    print(f"[FOLLOW] recorded {timestep} steps beside {args_cli.checkpoint}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
