# Real-Time Chunking (RTC) eval variant. Drives a flow-matching policy
# (SmolVLA/pi0/pi0.5) via RTC's proper interface: predict_action_chunk()
# with prefix-guided replanning every `execution_horizon` steps, instead of
# the naive truncate-and-replace approach select_action() uses.
#
# The RTC driving logic itself lives in humanoid_il.rtc_driver.RTCDrivenPolicy
# (embodiment-agnostic); SO101RTCPolicy below is the thin, robot-specific
# adapter around it.
import argparse
import json
import random
from copy import copy
from tqdm import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval")
parser.add_argument("--seed", type=int, default=1984)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--policy_path", type=str, required=True)
parser.add_argument("--execution_horizon", type=int, default=10)
parser.add_argument("--lang_instruction", type=str, default="Pick up the vial and place it in the rack")
parser.add_argument("--video_dir", type=str, default=None)
parser.add_argument("--video_camera", type=str, default="external_D455")
parser.add_argument("--video_prefix", type=str, default="ep")
parser.add_argument("--keep_all", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import humanoid_so101_vial_task.tasks  # noqa: F401
from humanoid_so101_vial_task.utils.keyboard import KeyboardControl
from humanoid_so101_vial_task.utils.lerobot_interface import LeRobotSO101Interface

from humanoid_il.rtc_driver import RTCDrivenPolicy


class SO101RTCPolicy:
    """Thin SO101-specific adapter around the embodiment-agnostic RTCDrivenPolicy.

    Owns exactly the two conversions that are actually robot-specific
    (sim-state -> LeRobot obs_frame, and model action -> sim command); the
    RTC replanning/guidance logic itself lives entirely in RTCDrivenPolicy
    and does not change per embodiment.
    """

    def __init__(self, robot_iface: LeRobotSO101Interface, policy_path: str, task_description: str, execution_horizon: int):
        self._iface = robot_iface
        self._iface.task_description = task_description
        self._iface.make_policy(policy_path)
        self._driver = RTCDrivenPolicy(
            policy=self._iface.policy,
            preprocessor=self._iface.preprocessor,
            postprocessor=self._iface.postprocessor,
            task_description=task_description,
            robot_type=self._iface.robot.robot_type,
            execution_horizon=execution_horizon,
        )

    def reset(self):
        self._driver.reset()

    def get_action(self, joint_positions: torch.Tensor, visual_obs: dict, log: bool = False) -> torch.Tensor:
        obs_frame = self._iface.sim_obs_to_policy_processor(joint_positions, visual_obs)
        action_values = self._driver.get_action(obs_frame)
        return self._iface.prediction_to_sim_processor(action_values, obs_frame, log=log)


def main():
    import os
    import imageio
    import numpy as np

    if args_cli.video_dir:
        os.makedirs(args_cli.video_dir, exist_ok=True)

    keyboard_control = KeyboardControl()

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
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

    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device, port=None, id="leader_arm_1", cameras=cameras, fps=30, kind="follower", rename_map=None,
    )
    robot_iface.init_device(visualize=False)

    policy = SO101RTCPolicy(robot_iface, args_cli.policy_path, args_cli.lang_instruction, args_cli.execution_horizon)

    obs, _ = env.reset()
    policy.reset()

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    initial_action = torch.tensor([-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465], device=env.unwrapped.device)

    step = 0
    num_episodes = 0
    num_successes = 0
    success_rate = 0.0
    pbar = None
    frames = []

    def grab_frame():
        if not args_cli.video_dir:
            return
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
        # torch.no_grad(), not inference_mode(): RTC's guidance step needs a
        # real (locally re-enabled) autograd pass; inference_mode tensors can
        # never support that even under a nested enable_grad().
        with torch.no_grad():
            if step == 0:
                pbar = tqdm(total=env.unwrapped.max_episode_length, desc=f"Rollout (ep {num_episodes + 1}, success: {success_rate:.1f}%)", unit="step")
                frames = []

            if step < 10:
                actions[:] = initial_action
            else:
                joint_positions = obs["policy"]["joint_pos_obs"][0].clone()
                actions[:] = policy.get_action(joint_positions, obs["visual"], log=False)

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

                if args_cli.video_dir and frames and (success or args_cli.keep_all):
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
