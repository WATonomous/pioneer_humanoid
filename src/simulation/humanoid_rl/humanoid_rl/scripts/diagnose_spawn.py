# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Small spawn/contact diagnostic for Isaac Lab locomotion tasks."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect robot spawn pose, body heights, and contact forces.")
parser.add_argument("--task", type=str, required=True, help="Name of the Isaac Lab task to inspect.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=20, help="Number of zero-action policy steps to simulate.")
parser.add_argument("--root_z", type=float, default=None, help="Override the robot default root height.")
parser.add_argument(
    "--root_rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="Override the robot default root quaternion.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_rl_tasks  # noqa: F401  (registers every RL task)
from isaaclab_tasks.utils import parse_env_cfg


def _fmt(values: torch.Tensor) -> str:
    return ", ".join(f"{value:.4f}" for value in values.detach().cpu().tolist())


def _print_pose_snapshot(env, label: str) -> None:
    robot = env.scene["robot"]
    body_pos_w = robot.data.body_pos_w[0]

    print(f"\n[{label}]")
    print(f"root_pos_w: [{_fmt(robot.data.root_pos_w[0])}]")
    print(f"root_quat_w: [{_fmt(robot.data.root_quat_w[0])}]")
    print("body_z:")
    for name, pos in zip(robot.body_names, body_pos_w):
        print(f"  {name}: {pos[2].item():.4f}")

    contact_sensor = env.scene.sensors.get("contact_forces")
    if contact_sensor is None:
        print("contact_forces: not configured")
        return

    forces = contact_sensor.data.net_forces_w[0]
    norms = torch.norm(forces, dim=-1)
    print("contact_force_norms:")
    for name, norm in zip(contact_sensor.body_names, norms):
        if norm.item() > 1.0:
            print(f"  {name}: {norm.item():.2f}")


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.root_z is not None:
        x, y, _ = env_cfg.scene.robot.init_state.pos
        env_cfg.scene.robot.init_state.pos = (x, y, args_cli.root_z)
    if args_cli.root_rot is not None:
        env_cfg.scene.robot.init_state.rot = tuple(args_cli.root_rot)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    print(f"task: {args_cli.task}")
    print(f"num_envs: {base_env.num_envs}")
    print(f"action_dim: {base_env.single_action_space.shape[0]}")
    print(f"body_names: {robot.body_names}")
    print(f"joint_names: {robot.joint_names}")

    env.reset()
    _print_pose_snapshot(base_env, "after reset")

    zero_actions = torch.zeros(
        (base_env.num_envs, base_env.single_action_space.shape[0]),
        device=base_env.device,
        dtype=torch.float32,
    )
    sample_steps = {1, 2, 5, 10, args_cli.steps}

    for step_idx in range(1, args_cli.steps + 1):
        _, reward, terminated, truncated, _ = env.step(zero_actions)
        if step_idx in sample_steps:
            root_z = robot.data.root_pos_w[:, 2].detach().cpu().tolist()
            reward_values = reward.detach().cpu().tolist()
            terminated_values = terminated.detach().cpu().tolist()
            truncated_values = truncated.detach().cpu().tolist()
            print(
                f"step={step_idx} root_z={root_z} reward={reward_values} "
                f"terminated={terminated_values} truncated={truncated_values}"
            )

    _print_pose_snapshot(base_env, f"after {args_cli.steps} zero-action steps")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
