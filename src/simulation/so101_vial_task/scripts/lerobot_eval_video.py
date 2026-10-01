# Video-capturing variant of scripts/lerobot_eval.py. Saves an mp4 per
# episode; keeps only successful ones (unless --keep_all).
import argparse
import json
import os
import random
from tqdm import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval")
parser.add_argument("--seed", type=int, default=1984)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--policy_type", type=str, choices=("lerobot", "groot"), default="lerobot")
parser.add_argument("--policy_path", type=str, default=None)
parser.add_argument("--rename_map", type=str, default=None)
parser.add_argument("--policy_host", type=str, default="localhost")
parser.add_argument("--policy_port", type=int, default=5555)
parser.add_argument("--action_horizon", type=int, default=16)
parser.add_argument("--lang_instruction", type=str, default="Pick up the vial and place it in the rack")
parser.add_argument("--rerun", action="store_true", default=False)
parser.add_argument("--video_dir", type=str, default="/workspace/humanoid/outputs/videos")
parser.add_argument("--video_camera", type=str, default="external_D455")
parser.add_argument("--keep_all", action="store_true", default=False, help="save videos for failed episodes too")
parser.add_argument("--video_prefix", type=str, default="ep")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import imageio

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import humanoid_so101_vial_task.tasks  # noqa: F401
from humanoid_so101_vial_task.utils.keyboard import KeyboardControl
from humanoid_so101_vial_task.utils.lerobot_interface import (
    LeRobotSO101Interface,
    GR00TRemotePolicy,
    LocalLeRobotPolicy,
)


def main():
    os.makedirs(args_cli.video_dir, exist_ok=True)
    keyboard_control = KeyboardControl()

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = args_cli.seed

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": camera_cfg.height, "width": camera_cfg.width}
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")

    rename_map = json.loads(args_cli.rename_map) if args_cli.rename_map else None
    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device, port=None, id="leader_arm_1", cameras=cameras, fps=30, kind="follower",
        rename_map=rename_map,
    )
    robot_iface.init_device(visualize=args_cli.rerun)

    if args_cli.policy_type == "groot":
        policy = GR00TRemotePolicy(
            robot_iface=robot_iface, host=args_cli.policy_host, port=args_cli.policy_port,
            action_horizon=args_cli.action_horizon, lang_instruction=args_cli.lang_instruction,
        )
        policy.connect()
    else:
        if not args_cli.policy_path:
            raise ValueError("--policy_path is required when --policy_type lerobot")
        policy = LocalLeRobotPolicy(robot_iface=robot_iface, policy_path=args_cli.policy_path, task_description=args_cli.lang_instruction)
        policy.connect()

    obs, _ = env.reset()
    policy.reset()

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    initial_action = torch.tensor([-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465], device=env.unwrapped.device)

    step = 0
    num_episodes = 0
    num_successes = 0
    success_rate = 0.0
    frames = []
    pbar = None

    def grab_frame():
        vis = obs.get("visual", {})
        if args_cli.video_camera in vis:
            img = vis[args_cli.video_camera][0]
        elif len(vis) > 0:
            img = list(vis.values())[0][0]
        else:
            return
        arr = img.detach().cpu().numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        frames.append(arr)

    while simulation_app.is_running():
        with torch.inference_mode():
            if step == 0:
                pbar = tqdm(total=env.unwrapped.max_episode_length, desc=f"Rollout (ep {num_episodes + 1}, success: {success_rate:.1f}%)", unit="step")
                frames = []

            if step < 10:
                actions[:] = initial_action
            else:
                joint_positions = obs["policy"]["joint_pos_obs"][0].clone()
                actions[:] = policy.get_action(joint_positions, obs["visual"], log=True)

            obs, rewards, terminated, truncated, info = env.step(actions)
            grab_frame()
            step += 1
            if pbar is not None:
                pbar.update(1)

            is_terminated = terminated.item() if terminated.numel() == 1 else terminated.any().item()
            is_truncated = truncated.item() if truncated.numel() == 1 else truncated.any().item()

            if is_terminated or is_truncated:
                if pbar is not None:
                    pbar.close()
                    pbar = None
                num_episodes += 1
                success = is_terminated and not is_truncated
                if success:
                    num_successes += 1
                success_rate = (num_successes / num_episodes) * 100

                if frames and (success or args_cli.keep_all):
                    tag = "success" if success else "fail"
                    path = os.path.join(args_cli.video_dir, f"{args_cli.video_prefix}_{num_episodes:03d}_{tag}.mp4")
                    imageio.mimwrite(path, frames, fps=30, quality=8)
                    print(f"[VIDEO] saved {path} ({len(frames)} frames)")

                obs, _ = env.reset()
                policy.reset()
                step = 0
                continue

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                if pbar is not None:
                    pbar.close()
                    pbar = None
                obs, _ = env.reset()
                policy.reset()
                step = 0
                continue

            if num_episodes >= args_cli.num_episodes:
                if pbar is not None:
                    pbar.close()
                    pbar = None
                print(f"[INFO]: Evaluated {args_cli.num_episodes} episodes")
                print(f"[INFO]: Success Rate: {num_successes}/{args_cli.num_episodes} ({success_rate:.1f}%)")
                env.close()
                simulation_app.close()

    env.close()


if __name__ == "__main__":
    main()
    while True:
        simulation_app.update()
    simulation_app.close()
