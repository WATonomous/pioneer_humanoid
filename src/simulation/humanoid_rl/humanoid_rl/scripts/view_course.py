# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open a static course in the Isaac Sim GUI for visual inspection (no policy).

Builds one *-Play course task (default: the full 12-up/walkway/12-down
stairs at fixed 8 cm risers), parks the viewport camera side-on so the
flights read as silhouettes, and steps zero actions until the window closes.
No checkpoint, no training, no recording.

Example:
    isaaclab.sh -p view_course.py \\
        --task Isaac-Velocity-Stairs-PioneerHumanoid-Play-v0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="View a stair course in the GUI.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Stairs-PioneerHumanoid-Play-v0",
                    help="Name of the *-Play task (stairs-only tiles guaranteed). "
                         "Pass the train task id to see the live curriculum map.")
parser.add_argument("--rows", type=int, default=2, help="Terrain rows to generate (small = fast, legible map).")
parser.add_argument("--cols", type=int, default=3, help="Terrain cols to generate.")
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
    env_cfg.scene.terrain.terrain_generator.num_rows = args_cli.rows
    env_cfg.scene.terrain.terrain_generator.num_cols = args_cli.cols
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Course layout summary from the live config (local tile frame, metres).
    gen = env_cfg.scene.terrain.terrain_generator
    course = gen.sub_terrains["stair_course"]
    print(f"[VIEW] task={args_cli.task} num_envs=1 "
          f"families={list(gen.sub_terrains)} rows={gen.num_rows} cols={gen.num_cols}")
    print(f"[VIEW] stair_course: steps={course.num_steps} risers={course.step_height_range} "
          f"tread={course.step_width} approach={course.approach_length} "
          f"walkway={course.walkway_length} spawn_x={course.spawn_x}")

    # Side-profile camera, close: course runs +X, so park on -Y and look at
    # mid-course. Narrow tiles + mini-map keep the subject large in frame.
    origin = env.unwrapped.scene.env_origins[0].tolist()
    eye = (origin[0] + 5.0, origin[1] - 6.0, origin[2] + 2.2)
    target = (origin[0] + 5.0, origin[1], origin[2] + 0.5)
    env.unwrapped.sim.set_camera_view(eye, target)
    print(f"[VIEW] camera eye={eye} target={target} (side profile, orbit with mouse)")

    env.reset()
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
