# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open the Onshape stair course in the Isaac Sim GUI for visual inspection (no policy).

Builds the Onshape straight-course *-Play task, parks the viewport camera
side-on to the robot's spawn, and steps zero actions until the window closes.
No checkpoint, no training, no recording.

Example:
    isaaclab.sh -p view_course.py \\
        --task Isaac-Velocity-OnshapeCourse-PioneerHumanoid-Play-v0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="View the Onshape stair course in the GUI.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-OnshapeCourse-PioneerHumanoid-Play-v0",
                    help="Name of the Onshape *-Play task.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

import humanoid_rl_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env.reset()
    robot = env.unwrapped.scene["robot"]
    pos = robot.data.root_pos_w[0].tolist()

    # Side profile: the straight course runs +Y from the spawn, so look at the
    # first flights from the robot's left side.
    eye = (pos[0] - 5.0, pos[1] + 4.0, pos[2] + 2.2)
    target = (pos[0], pos[1] + 4.0, pos[2] + 0.5)
    env.unwrapped.sim.set_camera_view(eye, target)
    print(f"[VIEW] task={args_cli.task} camera eye={eye} target={target}")

    zero = torch.zeros(env.unwrapped.action_manager.total_action_dim,
                       device=env.unwrapped.device).unsqueeze(0)
    print("[VIEW] holding zero actions until the window closes")
    while simulation_app.is_running():
        env.step(zero)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
