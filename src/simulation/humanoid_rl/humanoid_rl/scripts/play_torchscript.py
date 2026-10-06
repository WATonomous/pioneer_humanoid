#!/usr/bin/env python3
"""Play an exported Pioneer actor without constructing a versioned RSL-RL runner.

Export with export_policy.py first. Isaac Lab must launch before simulation
imports; artifact checks deliberately run before opening the application.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

from humanoid_rl.policy_artifacts import (
    DEFAULT_TASK,
    canonical_task,
    configure_flat_terrain,
    load_metadata,
    sha256,
    validate_assets,
    validate_env_config,
    validate_runtime_layout,
)


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    policy_argument = parser.add_argument("--policy", help="Exported TorchScript actor, not a training checkpoint.")
    metadata_argument = parser.add_argument("--metadata", help="Version-3 JSON manifest from export_policy.py.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Matching registered Pioneer Play task.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flat_terrain", action="store_true", help="Flatten the rough-no-stairs generator only.")
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "YAW_RATE"))
    parser.add_argument("--max_steps", type=int, default=0, help="0: watch until Q, Escape, or window close.")
    parser.add_argument("--no_real_time", action="store_true", help="Do not pace policy steps to simulation time.")
    parser.add_argument("--video", action="store_true", help="Record an MP4, then close automatically.")
    parser.add_argument("--video_seconds", type=float, default=20.0, help="Requested simulation seconds of video.")
    parser.add_argument("--video_folder", default="outputs/native_viewer/videos")
    parser.add_argument("--disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    # AppLauncher pre-parses with help temporarily disabled. Enforce required
    # paths afterwards, so --help works without creating an application.
    policy_argument.required = metadata_argument.required = True
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_steps < 0:
        parser.error("--num_envs must be positive and --max_steps must be nonnegative.")
    if not math.isfinite(args.video_seconds) or args.video_seconds <= 0:
        parser.error("--video_seconds must be finite and positive.")
    if not all(math.isfinite(value) for value in args.command):
        parser.error("--command values must be finite.")
    if args.flat_terrain and "-Stairs-" in canonical_task(args.task):
        parser.error("--flat_terrain is for RoughNoStairs; it does not replace the stairs course.")
    if args.video:
        args.enable_cameras = True
    return args, AppLauncher


def _cached_usd_spawn(env_cfg, usd_path):
    """Avoid native URDF re-conversion, retaining task-specific pair filtering."""
    import isaaclab.sim as sim_utils
    from pioneer_humanoid.selective_self_collision import (
        spawn_from_urdf_with_selective_self_collision,
        spawn_from_usd_with_selective_self_collision,
    )

    original = env_cfg.scene.robot.spawn
    if original.func is not spawn_from_urdf_with_selective_self_collision:
        raise RuntimeError("Supported viewer tasks must use the selective-self-collision URDF spawner.")
    options = {
        field.name: getattr(original, field.name)
        for field in fields(sim_utils.UsdFileCfg)
        if field.name not in {"func", "usd_path", "variants"} and hasattr(original, field.name)
    }
    spawn = sim_utils.UsdFileCfg(usd_path=str(usd_path), **options)
    spawn.func = spawn_from_usd_with_selective_self_collision
    env_cfg.scene.robot.spawn = spawn


def _fixed_command(env_cfg, command):
    vx, vy, yaw_rate = command
    config = env_cfg.commands.base_velocity
    config.ranges.lin_vel_x = (vx, vx)
    config.ranges.lin_vel_y = (vy, vy)
    config.ranges.ang_vel_z = (yaw_rate, yaw_rate)
    config.debug_vis = False
    config.heading_command = False
    config.rel_standing_envs = 0.0
    config.rel_heading_envs = 0.0


def _camera_config(env_cfg, num_envs):
    if num_envs == 1:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (3.0, -3.0, 1.6)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.45)
    else:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (10.0, -10.0, 10.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)


def _overview_camera(env):
    import torch

    origins = env.scene.env_origins.detach().cpu()
    minimum, maximum = origins.min(dim=0).values, origins.max(dim=0).values
    center = 0.5 * (minimum + maximum)
    scale = max(10.0, 1.35 * float(torch.linalg.vector_norm(maximum[:2] - minimum[:2])))
    lookat = (float(center[0]), float(center[1]), float(center[2]) + 0.5)
    eye = (lookat[0] + 0.45 * scale, lookat[1] - 0.75 * scale, lookat[2] + 0.70 * scale)
    env.sim.set_camera_view(eye, lookat)


def _exit_key(headless):
    """Return exit state and a cleanup callback; no detached Windows process."""
    state = {"requested": False}
    if headless:
        return state, lambda: None
    import carb.input
    import omni.appwindow

    interface = carb.input.acquire_input_interface()
    window = omni.appwindow.get_default_app_window()
    if window is None:
        print("[viewer_warning] No keyboard window; close the app or use --max_steps.", flush=True)
        return state, lambda: None
    keyboard = window.get_keyboard()

    def on_event(event, *_unused):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            name = getattr(event.input, "name", str(event.input)).upper()
            if name in {"Q", "ESCAPE"}:
                state["requested"] = True
        return True

    subscription = interface.subscribe_to_keyboard_events(keyboard, on_event)
    return state, lambda: interface.unsubscribe_to_keyboard_events(keyboard, subscription)


def _check_probe(policy, metadata, device):
    import torch

    probe = metadata["probe"]
    observation = torch.tensor([probe["observation"]], dtype=torch.float32, device=device)
    expected = torch.tensor([probe["expected_action"]], dtype=torch.float32, device=device)
    with torch.inference_mode():
        actual = policy(observation)
    torch.testing.assert_close(actual, expected, rtol=probe["rtol"], atol=probe["atol"])
    return float(torch.max(torch.abs(actual - expected)))


def _play(args, app, policy_path, metadata, usd_path):
    import gymnasium as gym
    import torch

    import humanoid_rl_tasks  # noqa: F401 -- register tasks after app launch
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed
    validate_env_config(env_cfg, metadata)
    if args.flat_terrain:
        configure_flat_terrain(env_cfg)
    _fixed_command(env_cfg, args.command)
    _cached_usd_spawn(env_cfg, usd_path)
    _camera_config(env_cfg, args.num_envs)

    env = None
    unsubscribe = lambda: None
    video_folder = Path(args.video_folder).expanduser().absolute()
    video_prefix = "pioneer_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    step_count = 0
    try:
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        base_env = env.unwrapped
        validate_runtime_layout(base_env, metadata)
        step_dt = float(base_env.step_dt)
        video_steps = max(1, round(args.video_seconds / step_dt))
        if args.video:
            # Playback uses simulated time, even if rendering is slower than real time.
            env.metadata["render_fps"] = round(1.0 / step_dt)
            env = gym.wrappers.RecordVideo(
                env, video_folder=str(video_folder), step_trigger=lambda step: step == 0,
                video_length=video_steps, name_prefix=video_prefix, disable_logger=True,
            )

        policy = torch.jit.load(str(policy_path), map_location=base_env.device).eval()
        error = _check_probe(policy, metadata, base_env.device)
        observations, _ = env.reset()
        if args.num_envs > 1:
            _overview_camera(base_env)
        policy_obs = observations["policy"]
        expected_obs_shape = (args.num_envs, metadata["observation_dim"])
        expected_action_shape = (args.num_envs, metadata["action_dim"])
        exit_state, unsubscribe = _exit_key(args.headless)
        print(f"[native_viewer_ready] task={args.task} envs={args.num_envs} probe_error={error:.3g}", flush=True)
        print(f"cached_asset: {usd_path}\npolicy: {policy_path}", flush=True)
        print(f"fixed_command: {list(args.command)}; flat_override: {args.flat_terrain}", flush=True)
        print("Click the viewport and press Q or Escape to exit; closing the window also exits.", flush=True)
        if args.video:
            print(f"Recording {args.video_seconds:g} simulation seconds to {video_folder}; then exiting.", flush=True)
        reset_counts = [0] * args.num_envs
        deadline = time.perf_counter()
        while app.is_running() and not exit_state["requested"]:
            if (args.max_steps and step_count >= args.max_steps) or (args.video and step_count >= video_steps):
                break
            with torch.inference_mode():
                if tuple(policy_obs.shape) != expected_obs_shape or not torch.isfinite(policy_obs).all():
                    raise RuntimeError(f"Invalid policy observations at step {step_count}: {tuple(policy_obs.shape)}.")
                inferred = policy(policy_obs)
                if tuple(inferred.shape) != expected_action_shape or not torch.isfinite(inferred).all():
                    raise RuntimeError(f"Invalid policy actions at step {step_count}: {tuple(inferred.shape)}.")
            # Isaac writes action buffers; clone outside inference mode. Do not normalize
            # twice or clip raw actions: the registered task applies scale/offset/clip.
            observations, _, terminated, truncated, _ = env.step(inferred.clone())
            policy_obs = observations["policy"]
            for env_id in torch.nonzero(terminated | truncated, as_tuple=False).flatten().tolist():
                reset_counts[env_id] += 1
            step_count += 1
            if not args.no_real_time:
                deadline += step_dt
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -5 * step_dt:
                    deadline = time.perf_counter()
        print(f"[viewer_stopped] policy_steps={step_count}; resets_per_env={reset_counts}", flush=True)
    except KeyboardInterrupt:
        print("[viewer_stopped] Keyboard interrupt received.", flush=True)
    finally:
        try:
            unsubscribe()
        finally:
            if env is not None:
                env.close()
    if args.video:
        clips = sorted(video_folder.glob(f"{video_prefix}*.mp4"))
        if not clips:
            raise RuntimeError(f"No MP4 was produced in {video_folder}.")
        print(f"[video_recording_complete] {clips[-1]} ({step_count * metadata['step_dt']:.2f}s simulated)", flush=True)


def main():
    args, launcher_type = _parse_args()
    metadata = load_metadata(Path(args.metadata).expanduser(), args.task)
    policy_path = Path(args.policy).expanduser().absolute()
    if sha256(policy_path) != metadata["policy_sha256"]:
        raise ValueError("Policy hash does not match the manifest; re-export this checkpoint.")
    # absolute(), deliberately NOT resolve(): retain pushd's mapped-drive spelling
    # on Windows, since Isaac Sim cannot resolve relative USD payloads via WSL UNC.
    repository_root = Path(__file__).absolute().parents[5]
    usd_path = validate_assets(repository_root, metadata)
    # AppLauncher mutates its input dictionary while resolving Kit settings.
    launcher = launcher_type(vars(args).copy())
    try:
        _play(args, launcher.app, policy_path, metadata, usd_path)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
