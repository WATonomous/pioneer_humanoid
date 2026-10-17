# Remote variant of lerobot_eval_rtc.py: the policy runs in rtc_policy_server.py (Python 3.12 venv)
# and this Isaac (Python 3.10) process talks to it over a local socket.
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
parser.add_argument("--server_host", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=5556)
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

import pickle
import socket
import struct


class SO101RTCPolicy:
    """Client for rtc_policy_server.py. Does the sim<->raw-unit conversions only."""

    def __init__(self, robot_iface: LeRobotSO101Interface, host: str, port: int):
        self._iface = robot_iface
        self._sock = socket.create_connection((host, port))
        self._call({"cmd": "ping"})
        print(f"[INFO]: Connected to RTC policy server at {host}:{port}")

    def _call(self, req):
        data = pickle.dumps(req, protocol=4)
        self._sock.sendall(struct.pack("!Q", len(data)) + data)
        header = b""
        while len(header) < 8:
            header += self._sock.recv(8 - len(header))
        (n,) = struct.unpack("!Q", header)
        buf = bytearray()
        while len(buf) < n:
            buf.extend(self._sock.recv(min(1 << 20, n - len(buf))))
        resp = pickle.loads(bytes(buf))
        if not resp["ok"]:
            raise RuntimeError(f"policy server error: {resp['error']}")
        return resp

    def reset(self):
        self._call({"cmd": "reset"})

    def get_action(self, joint_positions: torch.Tensor, visual_obs: dict, log: bool = False) -> torch.Tensor:
        state = self._iface.get_raw_actions_from_radians(joint_positions).cpu().numpy().astype(np.float32)
        images = {}
        for camera in self._iface.cameras.keys():
            img = visual_obs[f"rgb_{camera}"][0].detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            images[camera] = img[..., :3]
        action = self._call({"cmd": "act", "state": state, "images": images})["action"]
        raw = torch.tensor(action, dtype=torch.float32, device=self._iface.device)
        return self._iface.get_mapped_actions_vectorized(raw)


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

    policy = SO101RTCPolicy(robot_iface, args_cli.server_host, args_cli.server_port)

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

                if num_episodes >= args_cli.num_episodes:
                    print(f"[INFO]: Evaluated {args_cli.num_episodes} episodes", flush=True)
                    print(f"[INFO]: Success Rate: {num_successes}/{args_cli.num_episodes} ({success_rate:.1f}%)", flush=True)
                    env.close()
                    simulation_app.close()
                    return

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
                print(f"[INFO]: Evaluated {args_cli.num_episodes} episodes", flush=True)
                print(f"[INFO]: Success Rate: {num_successes}/{args_cli.num_episodes} ({success_rate:.1f}%)", flush=True)
                env.close()
                simulation_app.close()

    env.close()


if __name__ == "__main__":
    main()
    while True:
        simulation_app.update()
    simulation_app.close()
