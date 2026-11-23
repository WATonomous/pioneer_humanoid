# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=200,
                    help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False,
                    help="Run in real-time, if possible.")
parser.add_argument(
    "--diagnostics",
    action="store_true",
    default=False,
    help="Print the gait/camera/reward probe ([PLAYARC], [PLAYLEAD], [PLAYDBG], ...). "
         "Also enables the PLAY_INIT_OVERRIDE, PLAY_SPAWN_DY, PLAY_MARKER and "
         "PLAY_FOLLOW_ROBOT/PLAY_REAR_CAM environment variables.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib.metadata as metadata
import os
import time
import torch

from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import humanoid_rl_tasks  # noqa: F401  (registers every RL task)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from play_diagnostics import (
    PlayDiagnostics,
    add_spawn_marker,
    apply_init_override,
    apply_spawn_offsets,
    pin_course_camera,
)

installed_rsl_rl_version = metadata.version("rsl-rl-lib")


def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.diagnostics:
        apply_init_override(env_cfg)

    # specify directory for logging experiments
    _out_root = os.environ.get("HUMANOID_ROOT") or os.getcwd()
    log_root_path = os.path.abspath(os.path.join(_out_root, "outputs", "rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # Where a recording lands. PLAY_VIDEO_DIR / PLAY_VIDEO_PREFIX let a second
    # angle be recorded next to the first instead of overwriting it.
    video_dir = os.environ.get("PLAY_VIDEO_DIR") or os.path.join(log_dir, "videos", "play")
    video_prefix = os.environ.get("PLAY_VIDEO_PREFIX", "rl-video")
    if args_cli.video:
        print(f"[PLAY] video -> {video_dir} (prefix {video_prefix})", flush=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.diagnostics:
        # Spawn/camera setup that makes an Onshape probe reproducible.
        apply_spawn_offsets(env)
        pin_course_camera(env)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": video_dir,
            "name_prefix": video_prefix,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording a video during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if args_cli.diagnostics:
        add_spawn_marker(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    diagnostics = PlayDiagnostics(env) if args_cli.diagnostics else None

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_rsl_rl_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        policy_nn = runner.alg.policy if version.parse(installed_rsl_rl_version) >= version.parse("2.3.0") else runner.alg.actor_critic
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    step_index = 0
    # Hard step cap for headless probes (PLAY_MAX_STEPS=N): exits cleanly
    # without the --video recorder (which is otherwise the only bound on the loop).
    max_steps = int(os.environ.get("PLAY_MAX_STEPS", "0") or 0)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            step_index += 1
            if diagnostics is not None:
                diagnostics.on_step(actions, dones, step_index)
            if version.parse(installed_rsl_rl_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            elif policy_nn is not None:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if max_steps and step_index >= max_steps:
            print(f"[PLAY] step cap reached ({max_steps}) -- exiting", flush=True)
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

# From Isaac Lab scripts/reinforcement_learning/rsl_rl/play.py
