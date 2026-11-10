# Copyright (c) 2022-2025, The Isaac Lab Project Developers / WATonomous.
# SPDX-License-Identifier: BSD-3-Clause

"""Demo video of a distilled student: 4 parallel envs on one screen, ego camera on top and wrist camera below.

Each column is one env in its own lab room (envs are 4 m apart) with its own randomization: tray shift/turn, cube and tray
colour, cube spawn, tube and room light, back-wall photo; only the faint dome light is shared. The student acts from its
own 160x120 cameras; the video shows high-resolution cameras mounted as children of those, so they share the same pose and
jitter. Rendering uses the student env's quality preset (ray-traced reflections, bounce light).

Example:
  PYTHONPATH=$(pwd) $ISAACLAB/isaaclab.sh -p $RL_RUNNERS/demo_distill_push.py --headless \\
    --checkpoint outputs/rl/push_distill/<run>/nn/student_700.pt --seconds 30 --out outputs/demo/student.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demo video of a distilled push-block student.")
parser.add_argument("--task", type=str, default="Isaac-Bimanual-Push-Block-Distill-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Student .pt from distill_push.py")
parser.add_argument("--seconds", type=float, default=30.0, help="Length of the video (simulated time).")
parser.add_argument("--fps", type=int, default=25, help="Video frame rate; the policy runs at 50 Hz, so 25 shows every 2nd step.")
parser.add_argument("--scale", type=float, default=0.75, help="Panel size against the 640x480 render (0.75 gives a 1920-wide video).")
parser.add_argument("--out", type=str, default="outputs/demo/student.mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import humanoid_rl_tasks  # noqa: F401
import omni.usd
from isaaclab.sensors import CameraCfg
from pioneer_humanoid.bimanual_arm import make_camera_cfg
from humanoid_rl_tasks import workcell
from humanoid_rl_tasks.push_block.scene import TRAY_PIVOT
from humanoid_rl_tasks.push_block.mdp.utils import block_center_w
from humanoid_rl_tasks.push_block.student_policy import FrameStack, StudentVisionPolicy, student_images
from isaaclab_tasks.utils import parse_env_cfg
from pxr import Usd

NUM_ENVS = 4
PANEL_W, PANEL_H = 640, 480


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _display_camera(name: str) -> CameraCfg:
    """A 640x480 camera with the student camera's lens, mounted as a child of it so it follows its pose and DR jitter."""
    cam = make_camera_cfg(name, PANEL_H, PANEL_W)
    cam.prim_path = cam.prim_path + "/demo"
    cam.offset = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl")
    return cam


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=NUM_ENVS)
    env_cfg.scene.demo_ego = _display_camera("ego")
    env_cfg.scene.demo_wrist = _display_camera("wrist_left")
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    device = u.device

    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    student = StudentVisionPolicy(
        proprio_dim=ckpt["proprio_dim"],
        num_actions=ckpt["num_actions"],
        image_shapes={name: tuple(shape) for name, shape in ckpt["image_shapes"].items()},
    ).to(device)
    student.load_state_dict(ckpt["student"])
    student.eval()
    frame_stack = FrameStack(ckpt.get("history", 1))

    stage = omni.usd.get_context().get_stage()
    params = u.cfg.curriculum.success.params   # the success test the training logs, evaluated on the tray's own frame
    pivot = torch.tensor(TRAY_PIVOT, device=device)

    def solved() -> torch.Tensor:
        c = (block_center_w(u.scene["object"]) - u.scene.env_origins)[:, :2]
        tray = u.__dict__.get("_tray_pose", torch.zeros(NUM_ENVS, 3, device=device))
        rel = c - pivot - tray[:, :2]
        ct, st = torch.cos(-tray[:, 2]), torch.sin(-tray[:, 2])
        c_nom = pivot + torch.stack([ct * rel[:, 0] - st * rel[:, 1], st * rel[:, 0] + ct * rel[:, 1]], dim=-1)
        z = block_center_w(u.scene["object"])[:, 2] - u.scene.env_origins[:, 2]
        from humanoid_rl_tasks.push_block.mdp.utils import BLOCK_HALF_SIZE
        ok = (c_nom[:, 0] > params["x_min"]) & (c_nom[:, 0] < params["x_max"]) & (c_nom[:, 1].abs() < params["y_half"])
        ok &= (z - (params["floor_z"] + BLOCK_HALF_SIZE)).abs() < params["z_tol"]
        ok &= torch.norm(u.scene["object"].data.root_lin_vel_w, dim=1) < params["rest_speed"]
        return ok

    def shader_color(root: str):
        prim = stage.GetPrimAtPath(root)
        for name in ("inputs:diffuse_tint", "inputs:diffuseColor", "inputs:diffuse_color_constant"):
            for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                a = p.GetAttribute(name)
                if a.IsValid() and a.Get() is not None:
                    return tuple(int(255 * min(max(float(c), 0.0), 1.0)) for c in a.Get())
        return (128, 128, 128)

    def light(path: str):
        p = stage.GetPrimAtPath(path)
        col = p.GetAttribute("inputs:color").Get()
        return float(p.GetAttribute("inputs:intensity").Get()), tuple(int(255 * min(float(c), 1.0)) for c in col)

    nominal = {"TubeLight": workcell.TUBE_INTENSITY, "light": workcell.DOME_INTENSITY, "RoomLight": workcell.ROOM_LIGHT_INTENSITY}

    # layout
    s = args_cli.scale
    pw, ph = int(PANEL_W * s) // 2 * 2, int(PANEL_H * s) // 2 * 2
    top_h, head_h = 74, 150
    W, H = pw * NUM_ENVS, top_h + head_h + ph * 2
    f_big, f_med, f_small = _font(26), _font(20), _font(17)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args_cli.fps), "-i", "-",
         "-c:v", "libopenh264", "-b:v", "30M", "-pix_fmt", "yuv420p", args_cli.out],
        stdin=subprocess.PIPE,
    )

    n_steps = int(args_cli.seconds * 50)
    stride = max(1, round(50 / args_cli.fps))
    episodes = np.zeros(NUM_ENVS, dtype=int)
    wins = np.zeros(NUM_ENVS, dtype=int)
    obs_dict, _ = env.reset()
    prev_dones = None
    poster = None
    for step in range(n_steps):
        ok_before = solved()
        with torch.no_grad():
            actions = student(frame_stack(student_images(obs_dict), prev_dones), obs_dict["student"])
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        prev_dones = terminated | truncated
        done = prev_dones.cpu().numpy()
        episodes += done
        wins += done & ok_before.cpu().numpy()
        if step % stride:
            continue

        frame = Image.new("RGB", (W, H), (16, 16, 18))
        d = ImageDraw.Draw(frame)
        # top bar: the scene-wide lights and back wall
        d.text((14, 8), "Student policy  |  4 parallel envs  |  domain randomization, quality render", fill=(255, 255, 255), font=f_big)
        inten, col = light("/World/light")
        d.rectangle((14, 44, 36, 66), fill=col, outline=(200, 200, 200))
        d.text((44, 45), f"dome (shared) x{inten / nominal['light']:.2f}", fill=(220, 220, 220), font=f_med)
        d.text((330, 45), "every env is its own lab room: its own tube, room light and back-wall photo", fill=(180, 180, 180), font=f_med)
        d.text((W - 360, 12), "ego camera (top)   wrist camera (bottom)", fill=(180, 180, 180), font=f_small)

        tray = u._tray_pose.cpu().numpy()
        for e in range(NUM_ENVS):
            x0 = e * pw
            d.text((x0 + 12, top_h + 6), f"ENV {e + 1}", fill=(255, 255, 255), font=f_big)
            d.text((x0 + 12, top_h + 38), f"tray {tray[e, 0] * 1000:+.1f}, {tray[e, 1] * 1000:+.1f} mm   {np.degrees(tray[e, 2]):+.1f} deg", fill=(220, 220, 220), font=f_small)
            d.rectangle((x0 + 12, top_h + 66, x0 + 34, top_h + 88), fill=shader_color(f"/World/envs/env_{e}/Object"), outline=(200, 200, 200))
            d.text((x0 + 40, top_h + 66), "cube", fill=(220, 220, 220), font=f_small)
            d.rectangle((x0 + 100, top_h + 66, x0 + 122, top_h + 88), fill=shader_color(f"/World/envs/env_{e}/Box"), outline=(200, 200, 200))
            d.text((x0 + 128, top_h + 66), "tray", fill=(220, 220, 220), font=f_small)
            for k, (label, name) in enumerate((("tube", "TubeLight"), ("room", "RoomLight"))):
                inten, col = light(f"/World/envs/env_{e}/{name}")
                yy = top_h + 94 + 26 * k
                d.rectangle((x0 + 12, yy, x0 + 34, yy + 20), fill=col, outline=(200, 200, 200))
                d.text((x0 + 40, yy), f"{label} x{inten / nominal[name]:.2f}", fill=(220, 220, 220), font=f_small)
            photo = stage.GetPrimAtPath(f"/World/envs/env_{e}/LabRoom/Looks/GlassView/tex").GetAttribute("inputs:file").Get()
            d.text((x0 + 190, top_h + 94), f"wall: {os.path.basename(str(photo.path))}", fill=(220, 220, 220), font=f_small)
            d.text((x0 + pw - 190, top_h + 12), f"episode {episodes[e] + 1}", fill=(220, 220, 220), font=f_med)
            d.text((x0 + pw - 190, top_h + 40), f"solved {wins[e]}/{episodes[e]}", fill=(120, 230, 140), font=f_med)
            for r, cam in enumerate(("demo_ego", "demo_wrist")):
                img = u.scene[cam].data.output["rgb"][e].cpu().numpy()[..., :3].astype(np.uint8)
                frame.paste(Image.fromarray(img).resize((pw, ph), Image.LANCZOS), (x0, top_h + head_h + r * ph))
            d.rectangle((x0, top_h, x0 + pw - 1, H - 1), outline=(60, 60, 64))
        if poster is None and step > 25:
            poster = frame.copy()
        ff.stdin.write(frame.tobytes())
        if step % 250 == 0:
            print(f"[demo] step {step}/{n_steps}  solved {wins.tolist()} of {episodes.tolist()} episodes", flush=True)

    ff.stdin.close()
    ff.wait()
    if poster is not None:
        poster.save(os.path.splitext(args_cli.out)[0] + "_poster.png")
    total, won = int(episodes.sum()), int(wins.sum())
    print(f"[demo] wrote {args_cli.out}: {args_cli.seconds:.0f} s, {W}x{H}; finished episodes {total}, solved {won}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)  # Kit otherwise hangs in shutdown holding the GPU
