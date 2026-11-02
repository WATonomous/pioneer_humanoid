# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantitative course evaluation for stair-climbing policies (no camera needed).

Rolls out a trained RSL-RL checkpoint deterministically on a *-Play course task
and reports tensor-measured traversal stats: goal-reach rate while upright,
falls, distance covered, and foot-height evidence of real stepping. This is the
acceptance gate for each stair-curriculum stage: curves and completion counters
from training are not trusted on their own.

NOTE (2026-10-12): currently hangs pre-rollout (~100% CPU, no output) in two
attempts on a StepCount-Play task; cause unknown. Do not use without fixing.

Example:
    isaaclab.sh -p eval_course.py --task Isaac-Velocity-StepCount-PioneerHumanoid-Play-v0 \\
        --checkpoint <model.pt> --num_envs 32 --steps 1500 --seed 123
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a stair policy quantitatively.")
parser.add_argument("--task", type=str, default=None, help="Name of the *-Play task.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel eval environments.")
parser.add_argument("--steps", type=int, default=1500, help="Rollout steps per env (50 Hz policy steps).")
parser.add_argument("--seed", type=int, default=None, help="Env seed override (default: agent config seed).")
parser.add_argument("--goal_distance", type=float, default=None,
                    help="Override goal X (default: read from the course_complete termination).")
parser.add_argument("--out", type=str, default=None, help="Optional JSON summary path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import humanoid_rl_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed

    if not args_cli.checkpoint or not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(f"--checkpoint is required and must exist (got {args_cli.checkpoint})")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    foot_ids, _ = robot.find_bodies("Foot_.*")
    foot_ids = torch.tensor(foot_ids, device=env.unwrapped.device)
    origins = env.unwrapped.scene.env_origins

    if args_cli.goal_distance is not None:
        goal = float(args_cli.goal_distance)
    else:
        term = env.unwrapped.termination_manager.get_term("course_complete")
        goal = float(term.params["goal_distance"])

    n = env.unwrapped.num_envs
    max_x = torch.full((n,), -1e6, device=env.unwrapped.device)
    foot_apex = torch.zeros(n, device=env.unwrapped.device)
    step_rate_num = torch.zeros(n, device=env.unwrapped.device)
    fell = torch.zeros(n, dtype=torch.bool, device=env.unwrapped.device)
    ever_stepped = torch.zeros(n, dtype=torch.bool, device=env.unwrapped.device)

    obs = env.get_observations()
    import time as _time
    _t0 = _time.time()
    print(f"[EVAL] rollout start: {args_cli.steps} steps x {n} envs", flush=True)
    with torch.inference_mode():
        for i in range(args_cli.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            local_x = (robot.data.root_pos_w - origins)[:, 0]
            tilt = torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1)
            feet_z = robot.data.body_pos_w[:, foot_ids, 2]
            max_x = torch.maximum(max_x, local_x)
            foot_apex = torch.maximum(foot_apex, feet_z.max(dim=1).values)
            stepping = (feet_z[:, 0] - feet_z[:, 1]).abs() > 0.05
            step_rate_num += stepping.float()
            ever_stepped |= stepping
            fell |= (tilt > 0.8) | (~torch.isfinite(local_x))
            if (i + 1) % 100 == 0:
                print(f"[EVAL] step {i + 1}/{args_cli.steps} "
                      f"({(i + 1) / (_time.time() - _t0):.1f} steps/s)", flush=True)

    upright_reach = (max_x >= goal) & ~fell
    summary = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "num_envs": n,
        "steps": args_cli.steps,
        "seed": args_cli.seed,
        "goal_distance": goal,
        "frac_reached_goal_upright": float(upright_reach.float().mean()),
        "frac_ever_fell": float(fell.float().mean()),
        "frac_ever_stepped": float(ever_stepped.float().mean()),
        "mean_stepping_rate": float((step_rate_num / args_cli.steps).mean()),
        "mean_max_x": float(max_x.mean()),
        "min_max_x": float(max_x.min()),
        "mean_foot_apex": float(foot_apex.mean()),
        "min_foot_apex": float(foot_apex.min()),
    }
    print("[EVAL] " + json.dumps(summary, indent=2))
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[EVAL] summary written to {args_cli.out}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
