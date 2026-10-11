"""Render a qpos record (30 Hz, from matcha_demo) to MP4, in parallel chunks (CPU rendering is ~1.5 s/frame).

    MUJOCO_GL=egl python -m humanoid_hands_sim.render matcha.npy matcha.mp4 [--size 720,1280] [--fps 30]
        [--shadow] [--jobs 4] [--cam '{"lookat": [...], "distance": 1.5, "azimuth": 205, "elevation": -18}']
    MUJOCO_GL=egl python -m humanoid_hands_sim.render matcha.npy still.png --frame 52.0   # one frame at t=52 s
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool

import mujoco
import numpy as np

from .matcha_hands import CAMERA


def _renderer(H, W, camd):
    from .robot import make_model
    m = make_model()
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    for k, v in camd.items():
        setattr(cam, k, v)
    return m, mujoco.MjData(m), r, cam


def _frame(m, d, r, cam, q, shadow):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    r.update_scene(d, cam)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
    return r.render()


def _chunk(job):
    qs, out, H, W, fps, shadow, camd = job
    m, d, r, cam = _renderer(H, W, camd)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "18", out],
                          stdin=subprocess.PIPE)
    for q in qs:
        ff.stdin.write(_frame(m, d, r, cam, q, shadow).tobytes())
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--size", default="360,640", help="H,W")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--cam", default=None, help="JSON free-camera fields (default: the scene's CAMERA)")
    ap.add_argument("--frame", type=float, default=None, help="write one PNG at this time (s) instead of a video")
    a = ap.parse_args()
    H, W = map(int, a.size.split(","))
    camd = json.loads(a.cam) if a.cam else CAMERA
    traj = np.load(a.traj)
    if a.frame is not None:
        from PIL import Image
        m, d, r, cam = _renderer(H, W, camd)
        Image.fromarray(_frame(m, d, r, cam, traj[min(len(traj) - 1, int(a.frame * 30))], int(a.shadow))).save(a.out)
        return
    traj = traj[:: max(1, round(30 / a.fps))]
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
    jobs = [(p, f"{tmp}/{i:02d}.mp4", H, W, a.fps, int(a.shadow), camd)
            for i, p in enumerate(np.array_split(traj, a.jobs)) if len(p)]
    with Pool(len(jobs)) as pool:
        pool.map(_chunk, jobs)
    with open(f"{tmp}/list.txt", "w") as f:
        f.write("".join(f"file '{j[1]}'\n" for j in jobs))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{tmp}/list.txt",
                    "-c", "copy", a.out], check=True)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
