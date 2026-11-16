"""Labelled rollout videos (OpenCV drawing + ffmpeg H.264)."""
from __future__ import annotations

import subprocess

import cv2
import numpy as np

import common as C

FFMPEG = C.ffmpeg_exe()


def depth_color(d, dmax=1.4):
    x = np.clip(d / dmax, 0, 1)
    x = (x * 255).astype(np.uint8)
    c = cv2.applyColorMap(x, cv2.COLORMAP_TURBO)[:, :, ::-1]
    c[d <= 0] = 0
    return c


def _tile(obs, status, title):
    views = []
    for cam, lab in (("ego", "ego (D455)"), ("wrist_left", "wrist")):
        if ("rgb", cam) in obs:
            img = obs[("rgb", cam)]
            kind = "RGB"
        else:
            img = depth_color(obs[("depth", cam)], 1.4 if cam == "ego" else 0.6)
            kind = "depth"
        img = np.ascontiguousarray(img)
        cv2.putText(img, f"{lab} {kind}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{lab} {kind}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        views.append(img)
    if "demo" in obs:   # big third-person view, the policy's two cameras stacked on its right
        demo = np.ascontiguousarray(obs["demo"])
        if status["success"]:
            cv2.rectangle(demo, (0, 0), (demo.shape[1] - 1, demo.shape[0] - 1), (60, 200, 80), 8)
            cv2.putText(demo, "INSERTED" if C.SCENE == "peg_insert" else "SUCCESS", (18, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (40, 170, 60), 3, cv2.LINE_AA)
        side = np.concatenate(views, 0)
        if side.shape[0] != demo.shape[0]:
            side = cv2.resize(side, (int(side.shape[1] * demo.shape[0] / side.shape[0]), demo.shape[0]))
        row = np.concatenate([demo, side], 1)
    else:
        row = np.concatenate(views, 1)
    bar = np.full((34, row.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, title, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    flags = ([("grasped", status["grasped"]), ("over hole", status["aligned"]), ("INSERTED", status["success"])]
             if C.SCENE == "peg_insert" else [("SUCCESS", status["success"])])
    x = 6
    txt = f"t={status['t']:5.2f}s"
    cv2.putText(bar, txt, (x, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    x += 90
    for name, on in flags:
        col = (80, 220, 80) if on else (120, 120, 120)
        cv2.putText(bar, ("[x] " if on else "[ ] ") + name, (x, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        x += 120
    return np.concatenate([bar, row], 0)


def rollout_video(frames, path, tag, fps=12.5, title=None):
    """frames: list of (env_idx, iteration, obs dict, status dict)."""
    by_env: dict[int, dict[int, tuple]] = {}
    for vi, it, obs, st in frames:
        by_env.setdefault(vi, {})[it] = (obs, st)
    n_it = max(it for _, it, _, _ in frames) + 1
    envs = sorted(by_env)
    last = {}
    out = []
    for it in range(n_it):
        tiles = []
        for vi in envs:
            if it in by_env[vi]:
                last[vi] = by_env[vi][it]
            obs, st = last[vi]
            tiles.append(_tile(obs, st, title(vi) if title else f"policy: {tag}   eval episode {vi}"))
        if len(tiles) == 1:
            grid = tiles[0]
        else:
            while len(tiles) < 4:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:4], 1)], 0)
        out.append(grid)
    h, w = out[0].shape[:2]
    p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(path)],
                         stdin=subprocess.PIPE)
    for f in out:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    print("saved", path)
