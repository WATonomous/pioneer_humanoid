"""Run a scene's scripted demo, record its tape, and (optionally) render the video.

    MUJOCO_GL=osmesa python -m pioneer_whole_body.run_demo --scene bearing_block --out out/bearing_block
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import mujoco
import numpy as np

from . import entry, make_model
from .controller import WholeBody
from .skills import TaskFailed
from .video import Tape


def jitter(m: mujoco.MjModel, d: mujoco.MjData, rng: np.random.Generator, xy: float = 0.002,
           yaw: float = math.radians(3)) -> None:
    """Every loose part (free joint) a little off where the scene put it: up to ``xy`` m and ``yaw`` rad (not the
    ones welded to something at the start, e.g. captive screws: they go with their part)."""
    welded = {int(m.eq_obj2id[i]) for i in range(m.neq)
              if m.eq_type[i] == mujoco.mjtEq.mjEQ_WELD and d.eq_active[i]}
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE or int(m.jnt_bodyid[j]) in welded:
            continue
        a = m.jnt_qposadr[j]
        d.qpos[a:a + 2] += rng.uniform(-xy, xy, 2)
        q = np.zeros(4)
        dq = rng.uniform(-yaw, yaw)
        mujoco.mju_mulQuat(q, np.array([math.cos(dq / 2), 0, 0, math.sin(dq / 2)]), d.qpos[a + 3:a + 7])
        d.qpos[a + 3:a + 7] = q


def run(scene: str, out: str | None, fps: float = 30.0, seed: int = 0) -> dict:
    """Run the demo; seed 0 is the scene as built (the videos), other seeds jitter the parts (``jitter``) unless the
    scene has its own ``reset``."""
    e = entry(scene)
    m = make_model(scene)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if e.reset is not None:
        e.reset(m, d, np.random.default_rng(seed))
        mujoco.mj_forward(m, d)
    elif seed:
        jitter(m, d, np.random.default_rng(seed))
        mujoco.mj_forward(m, d)
    tape = Tape(fps, seed=seed)

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
    # done: every step through and the plan's own final checks passed
    notes.update(steps_done=k, steps=n, success=bool(e.progress and k >= n and notes["failed"] is None),
                 sim_time=round(wb.t, 2),
                 wall_time=round(time.time() - t0, 1))
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        tape.save(out + ".npz")
        with open(out + ".json", "w") as f:
            json.dump(notes, f, indent=1, default=str)
    return notes


def _one(args):
    scene, seed = args
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    return run(scene, None, seed=seed)


def sweep(scene: str, seeds, workers: int = 4) -> dict:
    """Run the demo on many seeds (parts jittered), in parallel: the success rate and how the failures failed."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_one, [(scene, s) for s in seeds]))
    ok = [r["seed"] for r in res if r["success"]]
    return dict(scene=scene, runs=len(res), success=len(ok), rate=round(len(ok) / max(1, len(res)), 2),
                failures={r["seed"]: f"step {r['steps_done'] + 1}/{r['steps']}: {r['failed']}" for r in res
                          if not r["success"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", help="path prefix: writes OUT.npz (tape), OUT.json, OUT.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=0, help="instead: run seeds 1..N (parts jittered), print the rate")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.seeds:
        print(json.dumps(sweep(a.scene, range(1, a.seeds + 1), a.workers), indent=1))
        return
    if not a.out:
        ap.error("--out is required (or --seeds)")
    notes = run(a.scene, a.out, seed=a.seed)
    print(json.dumps(notes, default=str))
    if a.render:
        from .shots import render
        w, h = map(int, a.size.split("x"))
        render(a.scene, a.out + ".npz", a.out + ".mp4", size=(w, h), workers=a.workers)


if __name__ == "__main__":
    main()
