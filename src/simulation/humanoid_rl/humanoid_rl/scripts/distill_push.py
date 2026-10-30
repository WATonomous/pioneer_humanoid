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
import time
from collections import deque
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
parser.add_argument("--history", type=int, default=1, help="Frames the student sees per camera (current + previous ones, stacked on channels).")
parser.add_argument("--no_image_aug", action="store_true", help="Train on clean images (default: random shift/contrast/colour/noise).")
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
from humanoid_rl_tasks.push_block.student_policy import FrameStack, StudentVisionPolicy, augment_rgb, student_images
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


def _image_shapes(images: dict) -> dict:
    """(H, W, C) per camera from NHWC tensors."""
    shapes = {}
    for name, img in images.items():
        if img.dim() != 4 or img.shape[-1] not in (1, 3, 4):
            raise RuntimeError(f"Expected NHWC RGB for camera '{name}', got {tuple(img.shape)}")
        shapes[name] = (img.shape[1], img.shape[2], min(img.shape[3], 3))
    return shapes


def augment_rgb_dict(images: dict) -> dict:
    return {name: augment_rgb(img) for name, img in images.items()}


def _sync_time(device) -> float:
    """perf_counter after waiting for queued GPU work, so per-phase timings are honest."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _num(x: float, spec: str = ".4f") -> str:
    return "n/a" if x != x else format(x, spec)


def _format_report(it: int, total_its: int, horizon: int, num_envs: int, stats: dict, env_log: dict, elapsed: float, tot_timesteps: int) -> str:
    """One iteration's block, laid out like the teacher's RSL-RL printout (rewards swapped for the behavior loss)."""
    width, pad = 80, 35
    fps = int(horizon * num_envs / max(stats["collect_s"] + stats["learn_s"], 1e-9))
    rows = [
        ("Computation:", f"{fps:.0f} steps/s (collection: {stats['collect_s']:.3f}s, learning {stats['learn_s']:.3f}s)"),
        ("Mean behavior loss (MSE):", _num(stats["loss"])),
        ("Last-step behavior loss:", _num(stats["loss_last"])),
        ("Mean action error (L1):", _num(stats["l1"])),
        ("Mean gradient norm (pre-clip):", _num(stats["grad"])),
        ("Learning rate:", format(stats["lr"], ".2e")),
        ("Teacher-driven steps:", f"{stats['teacher_frac'] * 100:.0f}%"),
        ("Mean episode length:", _num(stats["ep_len"], ".2f")),
    ]
    rows += [(f"{k}:", _num(env_log[k])) for k in sorted(env_log) if k.startswith("Curriculum/")]
    rows += [(f"{k}:", _num(env_log[k])) for k in sorted(env_log) if k.startswith("Episode_Termination/")]
    header = f" \033[1m Student distillation iteration {it}/{total_its} \033[0m "
    out = [f"{'#' * width}", header.center(width, " "), ""]
    out += [f"{label:>{pad}} {value}" for label, value in rows]
    iter_time = stats["collect_s"] + stats["learn_s"]
    eta = elapsed / max(it + 1, 1) * (total_its - it - 1)
    out += [
        "-" * width,
        f"{'Total timesteps:':>{pad}} {tot_timesteps}",
        f"{'Iteration time:':>{pad}} {iter_time:.2f}s",
        f"{'Time elapsed:':>{pad}} {_hms(elapsed)}",
        f"{'ETA:':>{pad}} {_hms(eta)}",
        "",
    ]
    return "\n".join(out)


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
    student_rgb = student_images(obs_dict)

    num_actions = unwrapped.action_manager.total_action_dim
    teacher = _load_teacher_actor(
        args_cli.teacher, teacher_obs.shape[-1], num_actions, device
    )

    image_shapes = _image_shapes(student_rgb)
    image_shapes = {name: (h, w, c * args_cli.history) for name, (h, w, c) in image_shapes.items()}
    student = StudentVisionPolicy(
        proprio_dim=student_obs.shape[-1],
        num_actions=num_actions,
        image_shapes=image_shapes,
    ).to(device)
    if args_cli.student_checkpoint:
        student.load_state_dict(
            torch.load(args_cli.student_checkpoint, map_location=device, weights_only=False)["student"]
        )
        print(f"[INFO] Resumed student from {args_cli.student_checkpoint}")

    optimizer = torch.optim.Adam(student.parameters(), lr=args_cli.lr)
    loss_fn = nn.MSELoss()

    obs_dict, _ = env.reset()
    num_envs = unwrapped.num_envs
    frame_stack = FrameStack(args_cli.history)
    prev_dones = None
    tot_timesteps = 0
    ep_len = torch.zeros(num_envs, device=device)
    len_buffer = deque(maxlen=100)  # lengths of the last 100 finished episodes
    env_log: dict = {}  # latest Curriculum/* and Episode_Termination/* values from the env
    start_time = time.time()
    for iteration in range(args_cli.max_iterations):
        student.train()
        losses, l1_errors, grad_norms = [], [], []
        teacher_steps = 0
        collect_s = learn_s = 0.0

        for _ in range(args_cli.horizon):
            t0 = _sync_time(device)
            teacher_o = _obs_group(obs_dict, "teacher")
            student_o = _obs_group(obs_dict, "student")
            rgb = frame_stack(student_images(obs_dict), prev_dones)

            with torch.no_grad():
                teacher_actions = teacher(teacher_o)
            if args_cli.no_image_aug:
                student_actions = student(rgb, student_o)
                rollout_actions = student_actions.detach()
            else:
                # learn from augmented images, but act on clean ones so the logged success is the deployed student's
                student_actions = student(augment_rgb_dict(rgb), student_o)
                with torch.no_grad():
                    rollout_actions = student(rgb, student_o)

            loss = loss_fn(student_actions, teacher_actions)
            optimizer.zero_grad()
            loss.backward()
            grad_norms.append(float(nn.utils.clip_grad_norm_(student.parameters(), 1.0)))
            optimizer.step()
            losses.append(loss.item())
            l1_errors.append((student_actions - teacher_actions).abs().mean().item())
            t1 = _sync_time(device)

            if torch.rand(1).item() < args_cli.beta_teacher:
                step_actions = teacher_actions
                teacher_steps += 1
            else:
                step_actions = rollout_actions

            obs_dict, rew, terminated, truncated, extras = env.step(step_actions)
            t2 = _sync_time(device)
            learn_s += t1 - t0
            collect_s += t2 - t1

            dones = terminated | truncated
            prev_dones = dones
            ep_len += 1
            if dones.any():
                len_buffer.extend(ep_len[dones].tolist())
                ep_len[dones] = 0
            tot_timesteps += num_envs
            for key, value in extras.get("log", {}).items():
                if key.startswith(("Curriculum/", "Episode_Termination/")):
                    env_log[key] = float(value)

        stats = dict(
            collect_s=collect_s,
            learn_s=learn_s,
            loss=sum(losses) / len(losses),
            loss_last=losses[-1],
            l1=sum(l1_errors) / len(l1_errors),
            grad=sum(grad_norms) / len(grad_norms),
            lr=optimizer.param_groups[0]["lr"],
            teacher_frac=teacher_steps / args_cli.horizon,
            ep_len=sum(len_buffer) / len(len_buffer) if len_buffer else float("nan"),
        )
        print(
            _format_report(iteration, args_cli.max_iterations, args_cli.horizon, num_envs, stats, env_log,
                           time.time() - start_time, tot_timesteps),
            flush=True,
        )
        writer.add_scalar("Loss/behavior", stats["loss"], iteration)
        writer.add_scalar("Loss/behavior_last_step", stats["loss_last"], iteration)
        writer.add_scalar("Loss/action_error_l1", stats["l1"], iteration)
        writer.add_scalar("Loss/grad_norm", stats["grad"], iteration)
        writer.add_scalar("Perf/collection_time", collect_s, iteration)
        writer.add_scalar("Perf/learning_time", learn_s, iteration)
        writer.add_scalar("Train/teacher_driven_fraction", stats["teacher_frac"], iteration)
        if len_buffer:
            writer.add_scalar("Train/mean_episode_length", stats["ep_len"], iteration)
        for key, value in env_log.items():
            writer.add_scalar(key, value, iteration)

        if iteration % args_cli.save_interval == 0 or iteration == args_cli.max_iterations - 1:
            path = os.path.join(log_dir, "nn", f"student_{iteration}.pt")
            torch.save(
                {
                    "student": student.state_dict(),
                    "iteration": iteration,
                    "proprio_dim": student_obs.shape[-1],
                    "num_actions": num_actions,
                    "image_shapes": image_shapes,
                    "history": args_cli.history,
                },
                path,
            )
            print(f"[INFO] Saved {path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
