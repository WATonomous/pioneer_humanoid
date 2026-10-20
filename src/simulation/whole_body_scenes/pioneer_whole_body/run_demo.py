"""Run a scene's scripted demo, record its tape, and (optionally) render the video.

    MUJOCO_GL=osmesa python -m pioneer_whole_body.run_demo --scene bearing_block --out out/bearing_block
"""
from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import numpy as np

from . import entry, make_model
from .controller import WholeBody
from .skills import TaskFailed
from .video import Tape


def run(scene: str, out: str, fps: float = 30.0, seed: int = 0) -> dict:
    e = entry(scene)
    m = make_model(scene)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if e.reset is not None:
        e.reset(m, d, np.random.default_rng(seed))
        mujoco.mj_forward(m, d)
    tape = Tape(fps)

    def caption():
        if e.progress is None:
            return ("", "")
        k, n, text = e.progress(m, d)
        return (f"step {min(k + 1, n)}/{n}", f"{'done' if k >= n else f'step {k + 1}/{n}: {text}'}")

    tape.caption_fn = caption
    tape.focus_fn = lambda: (wb.gaze_target() if wb.gaze_target is not None else None)
    wb = WholeBody(m, d, scene_step=e.step, on_frame=tape.frame, fps=fps)
    notes: dict = {"scene": scene, "seed": seed}
    t0 = time.time()
    try:
        e.plan(wb, notes)
        notes["failed"] = None
    except TaskFailed as err:
        notes["failed"] = str(err)
    k, n, _ = e.progress(m, d) if e.progress else (0, 0, "")
    notes.update(steps_done=k, steps=n, success=bool(e.progress and k >= n), sim_time=round(wb.t, 2),
                 wall_time=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    tape.save(out + ".npz")
    with open(out + ".json", "w") as f:
        json.dump(notes, f, indent=1)
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, help="path prefix: writes OUT.npz (tape), OUT.json, OUT.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    notes = run(a.scene, a.out, seed=a.seed)
    print(json.dumps(notes))
    if a.render:
        from .shots import render
        w, h = map(int, a.size.split("x"))
        render(a.scene, a.out + ".npz", a.out + ".mp4", size=(w, h), workers=a.workers)


if __name__ == "__main__":
    main()
