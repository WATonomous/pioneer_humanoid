# Copyright (c) 2022-2025, The Isaac Lab Project Developers / WATonomous.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal online distillation: privileged push teacher -> vision student.

Loads an RSL-RL PPO teacher checkpoint (actor MLP on 37-D privileged obs) and
trains a CNN+MLP student that sees RGB + proprio only, via behavior cloning on
teacher actions while the student rolls out in the env (DAgger-style).

Example:
  PYTHONPATH=$(pwd) $ISAACLAB/isaaclab.sh -p $RL_RUNNERS/distill_push.py \\
    --task Isaac-Bimanual-Push-Block-Distill-v0 --headless --enable_cameras \\
    --teacher models/push/model_1499.pt \\
    --num_envs 64 --max_iterations 2000
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal vision distillation for push-block.")
parser.add_argument("--task", type=str, default="Isaac-Bimanual-Push-Block-Distill-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--teacher", type=str, required=True, help="Path to RSL-RL PPO .pt checkpoint.")
parser.add_argument("--student_checkpoint", type=str, default=None, help="Optional student resume .pt")
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--horizon", type=int, default=24, help="Env steps collected per update.")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--beta_teacher", type=float, default=0.0,
                    help="Probability of stepping with teacher actions (1=pure BC data, 0=DAgger).")
parser.add_argument("--run_name", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from torch.utils.tensorboard import SummaryWriter

import humanoid_rl_tasks  # noqa: F401
from humanoid_rl_tasks.push_block.student_policy import StudentVisionPolicy
from isaaclab_tasks.utils import parse_env_cfg


def _load_teacher_actor(checkpoint: str, obs_dim: int, act_dim: int, device: str) -> nn.Module:
    """Build an ActorCritic matching BimanualPushBlockPPORunnerCfg and load actor weights.

    rsl_rl >=4.0 replaced the (num_actor_obs, num_critic_obs) constructor with
    (obs: TensorDict, obs_groups: dict[str, list[str]]); it only uses these to
    size the actor/critic MLPs, and act_inference() still reduces to a plain
    torch.cat + MLP forward internally, so ac.actor is usable identically to
    the old API once constructed. Detect via signature instead of version
    string so this works unmodified in either rsl_rl generation.
    """
    import inspect

    if "obs_groups" in inspect.signature(ActorCritic.__init__).parameters:
        from tensordict import TensorDict

        dummy_obs = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
        ac = ActorCritic(
            obs=dummy_obs,
            obs_groups={"policy": ["policy"], "critic": ["policy"]},
            num_actions=act_dim,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
            init_noise_std=1.0,
        ).to(device)
    else:
        ac = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=act_dim,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
            init_noise_std=1.0,
        ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    own = ac.state_dict()
    mismatch = {k: (tuple(v.shape), tuple(own[k].shape)) for k, v in state.items() if k in own and v.shape != own[k].shape}
    if mismatch:
        raise SystemExit(f"{checkpoint}: teacher shapes {mismatch} (checkpoint, env) don't match the env's {obs_dim} obs / "
                         f"{act_dim} actions: a teacher trained on the old 6-DoF arm can't drive pioneer_bimanual_arm_8dof")
    ac.load_state_dict(state, strict=False)
    ac.eval()
    for p in ac.parameters():
        p.requires_grad_(False)
    print(f"[INFO] Loaded teacher actor from {checkpoint}")
    return ac.actor


def _obs_group(obs: dict, key: str):
    if key not in obs:
        raise KeyError(f"Missing obs group '{key}'. Available: {list(obs.keys())}")
    return obs[key]


def _rgb_tensor(obs: dict) -> torch.Tensor:
    """Return NHWC RGB from the student_rgb group (dict or tensor)."""
    rgb_group = _obs_group(obs, "student_rgb")
    if isinstance(rgb_group, dict):
        rgb = rgb_group["rgb"]
    else:
        rgb = rgb_group
    if rgb.dim() != 4:
        raise RuntimeError(f"Expected 4D RGB tensor, got shape {tuple(rgb.shape)}")
    return rgb


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device

    _out_root = os.environ.get("HUMANOID_ROOT") or os.getcwd()
    log_root = os.path.abspath(os.path.join(_out_root, "outputs", "rl", "push_distill"))
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        run_dir += f"_{args_cli.run_name}"
    log_dir = os.path.join(log_root, run_dir)
    os.makedirs(os.path.join(log_dir, "nn"), exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, "summaries"))
    print(f"[INFO] Logging to {log_dir}")

    obs_dict, _ = env.reset()
    teacher_obs = _obs_group(obs_dict, "teacher")
    student_obs = _obs_group(obs_dict, "student")
    student_rgb = _rgb_tensor(obs_dict)

    num_actions = unwrapped.action_manager.total_action_dim
    teacher = _load_teacher_actor(
        args_cli.teacher, teacher_obs.shape[-1], num_actions, device
    )

    # Expect NHWC from isaaclab image term
    if student_rgb.shape[-1] not in (1, 3, 4):
        raise RuntimeError(f"Expected NHWC RGB, got {tuple(student_rgb.shape)}")
    h, w, c = student_rgb.shape[1], student_rgb.shape[2], min(student_rgb.shape[3], 3)
    image_shape = (h, w, c)

    student = StudentVisionPolicy(
        proprio_dim=student_obs.shape[-1],
        num_actions=num_actions,
        image_shape=image_shape,
    ).to(device)
    if args_cli.student_checkpoint:
        student.load_state_dict(
            torch.load(args_cli.student_checkpoint, map_location=device, weights_only=False)["student"]
        )
        print(f"[INFO] Resumed student from {args_cli.student_checkpoint}")

    optimizer = torch.optim.Adam(student.parameters(), lr=args_cli.lr)
    loss_fn = nn.MSELoss()

    obs_dict, _ = env.reset()
    global_step = 0
    for iteration in range(args_cli.max_iterations):
        student.train()
        total_loss = 0.0
        n_updates = 0
        ep_rew = torch.zeros(unwrapped.num_envs, device=device)

        for _ in range(args_cli.horizon):
            teacher_o = _obs_group(obs_dict, "teacher")
            student_o = _obs_group(obs_dict, "student")
            rgb = _rgb_tensor(obs_dict)

            with torch.no_grad():
                teacher_actions = teacher(teacher_o)
            student_actions = student(rgb, student_o)

            loss = loss_fn(student_actions, teacher_actions)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_updates += 1

            if torch.rand(1).item() < args_cli.beta_teacher:
                step_actions = teacher_actions
            else:
                step_actions = student_actions.detach()

            obs_dict, rew, terminated, truncated, extras = env.step(step_actions)
            ep_rew += rew
            dones = terminated | truncated
            global_step += unwrapped.num_envs
            if dones.any():
                # episode returns are already handled by env reset; keep rolling mean via extras if present
                pass

        mean_loss = total_loss / max(n_updates, 1)
        writer.add_scalar("distill/behavior_loss", mean_loss, iteration)
        writer.add_scalar("distill/ep_rew_mean", ep_rew.mean().item(), iteration)
        if iteration % 10 == 0:
            print(
                f"iter={iteration:5d}  loss={mean_loss:.6f}  "
                f"rew_mean={ep_rew.mean().item():.3f}  steps={global_step}"
            )

        if iteration % args_cli.save_interval == 0 or iteration == args_cli.max_iterations - 1:
            path = os.path.join(log_dir, "nn", f"student_{iteration}.pt")
            torch.save(
                {
                    "student": student.state_dict(),
                    "iteration": iteration,
                    "proprio_dim": student_obs.shape[-1],
                    "num_actions": num_actions,
                    "image_shape": image_shape,
                },
                path,
            )
            print(f"[INFO] Saved {path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
