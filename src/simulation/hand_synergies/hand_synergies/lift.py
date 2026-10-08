"""Grasp and lift: the hand comes down palm-first over an object resting on a table, closes, and lifts it.

One trial: a random tall object (a bottle-like cylinder or carton-like box, 12-18 cm) stands on the table;
the hand (on a mocap wrist, palm down) is placed with its palm just above it, in a pre-shape from a
sampler -- a claw grab from above: the stock thumb hangs ~10.6 cm below the palm, so it would hit the
table beside anything shorter; the
power autograsp closes (grasp_gen.GraspGen.close, gravity on, nothing pinned), the wrist lifts LIFT_M
over LIFT_S and holds for HOLD_S. Success: the object ends at least SUCCESS_RISE_M higher.

Samplers (pre-shapes) are search.py's: ``prior`` (grasp_gen's random pre-shape) or ``synergy-k`` /
``gauss-20`` (a fitted grasp posture opened to the prior's aperture).

    python -m hand_synergies.lift --trials 600 --samplers prior,synergy-3 --thumbs stock,near36 --finger-collisions
    MUJOCO_GL=osmesa python -m hand_synergies.lift --gif out/lift.gif --trials 8 --samplers prior
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np
from pioneer_humanoid.mujoco_hand import PALM_BODY, hand_spec

from .scene import (MAX_SIZE, PALM_SURFACE_Z, TIMESTEP, _GEOM_TYPES, set_object,
                    stiffen_contacts)

LIFT_M = 0.10
LIFT_S = 1.0
HOLD_S = 1.0
SETTLE_S = 0.3
SUCCESS_RISE_M = 0.08
PALM_POINT = np.array([0.015, 0.105])   # hand-frame xy over the object's centre (between thumb and fingers)
PALM_JITTER = 0.015                      # m, uniform +- on that placement
GAP = (0.0, 0.01)                        # m between the palm surface and the object's top
OBJECT_HALF_HEIGHT = (0.06, 0.09)        # m
OBJECT_HALF_WIDTH = (0.012, 0.035)       # m (cylinder radius, box half sides)


def make_model(thumb: str = "stock", finger_collisions: bool = False, width: int = 480,
               height: int = 360) -> mujoco.MjModel:
    from .thumb import thumb_kwargs

    spec = mujoco.MjSpec()
    spec.modelname = "hand_lift"
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.worldbody.add_light(pos=[0.3, -0.3, 0.8], dir=[-0.3, 0.3, -1], diffuse=[0.8, 0.8, 0.8])
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.5, 0.5, 0.01],
                            rgba=[0.55, 0.5, 0.45, 1], friction=[1.0, 0.01, 0.001])
    wrist = spec.worldbody.add_body(name="wrist", mocap=True, pos=[0, 0, 0.3])
    wrist.add_frame().attach_body(
        hand_spec(**thumb_kwargs(thumb), finger_collisions=finger_collisions).body(PALM_BODY), "", "")
    obj = spec.worldbody.add_body(name="object", pos=[0, 0.1, 0.05])
    obj.add_freejoint(name="object")
    for name, gtype in _GEOM_TYPES.items():
        obj.add_geom(name=name, type=gtype, size=MAX_SIZE[name], rgba=[0.9, 0.55, 0.2, 1],
                     friction=[1.0, 0.01, 0.001], density=0)
    obj.mass = 0.05
    obj.inertia = [1e-5, 1e-5, 1e-5]
    obj.explicitinertial = True
    # The inertial frame must be set too: left alone it took the body's spawn position, putting the
    # centre of mass ~11 cm from the object's centre.
    obj.ipos = [0.0, 0.0, 0.0]
    obj.iquat = [1.0, 0.0, 0.0, 0.0]
    stiffen_contacts(spec)
    return spec.compile()


class Lifter:
    def __init__(self, thumb: str, finger_collisions: bool):
        from .grasp_gen import GraspGen

        self.model = make_model(thumb, finger_collisions)
        self.gen = GraspGen(model=self.model)  # reuses its autograsp on this model
        self.data = self.gen.data
        self.wrist = self.model.body("wrist").mocapid[0]
        self.table = self.model.geom("table").id

    def _object(self, rng):
        """Tall object standing on the table: (kind, size, pos, quat, half height)."""
        kind = ("cylinder", "box")[rng.integers(2)]
        yaw = rng.uniform(-np.pi, np.pi)
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        half_h = rng.uniform(*OBJECT_HALF_HEIGHT)
        if kind == "cylinder":
            size = np.array([rng.uniform(*OBJECT_HALF_WIDTH), half_h])
        else:
            size = np.array([rng.uniform(*OBJECT_HALF_WIDTH), rng.uniform(*OBJECT_HALF_WIDTH), half_h])
        return kind, size, np.array([0.0, 0.0, half_h]), quat, half_h

    def _start_clear(self) -> bool:
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        return not any(c.dist < 0 and m.geom_bodyid[c.geom1] != m.geom_bodyid[c.geom2]
                       for c in d.contact[: d.ncon])

    def trial(self, seed: int, sampler: str, pca: dict | None, frames: list | None = None) -> bool:
        from .grasp_gen import GRASP_TYPES
        from .search import OPEN_SCALE, OPEN_STEPS, _sample

        rng = np.random.default_rng(seed)  # same object and placement for every sampler / thumb
        m, d, gen = self.model, self.data, self.gen
        kind, size, pos, quat, half_h = self._object(rng)
        offset = PALM_POINT + rng.uniform(-PALM_JITTER, PALM_JITTER, size=2)
        gap = rng.uniform(*GAP)
        wrist0 = np.array([pos[0] - offset[0], pos[1] - offset[1], 2 * half_h + gap - PALM_SURFACE_Z])

        q = _sample(sampler, np.random.default_rng([seed, 1]), pca, gen)
        if sampler != "prior":
            q = q.copy()
            q[gen.close_j] *= OPEN_SCALE
        for delta in (0.0, *OPEN_STEPS):  # open until the hand clears the object and the table
            q0 = q.copy()
            q0[gen.close_j] -= gen.close_sign * delta
            q0 = np.clip(q0, *gen.ranges.T)
            mujoco.mj_resetData(m, d)
            set_object(m, gen.idx, kind, size)
            d.mocap_pos[self.wrist] = wrist0
            d.qpos[gen.idx.qpos] = q0
            d.ctrl[gen.idx.act] = q0
            d.qpos[gen.idx.obj_qpos:gen.idx.obj_qpos + 3] = pos
            d.qpos[gen.idx.obj_qpos + 3:gen.idx.obj_qpos + 7] = quat
            if self._start_clear():
                break
        else:
            return False

        def record():
            if frames is not None and len(frames) < 10_000 and int(d.time / TIMESTEP) % 40 == 0:
                frames.append(d.time)

        mujoco.mj_step(m, d, nstep=int(0.1 / TIMESTEP))  # object settles on the table
        z0 = d.qpos[gen.idx.obj_qpos + 2]
        ctrl = gen.close(q0, GRASP_TYPES["power"][0], after_step=record)
        for _ in range(int(SETTLE_S / TIMESTEP)):
            mujoco.mj_step(m, d)
            record()
        n = int(LIFT_S / TIMESTEP)
        for i in range(n + int(HOLD_S / TIMESTEP)):
            d.mocap_pos[self.wrist] = wrist0 + [0, 0, LIFT_M * min(1.0, (i + 1) / n)]
            d.ctrl[gen.idx.act] = ctrl
            mujoco.mj_step(m, d)
            record()
        return bool(d.qpos[gen.idx.obj_qpos + 2] - z0 >= SUCCESS_RISE_M)


_LIFTERS = {}


def _work(args):
    seed, thumb, sampler, fc, pca_path = args
    key = (thumb, fc)
    if key not in _LIFTERS:
        _LIFTERS[key] = Lifter(thumb, fc)
    pca = dict(np.load(pca_path)) if sampler != "prior" else None
    return thumb, sampler, _LIFTERS[key].trial(seed, sampler, pca)


def render_gif(path: str, thumb: str, sampler: str, fc: bool, pca: dict | None, seeds) -> None:
    """Re-run the first successful seed, rendering every 40 ms."""
    from PIL import Image

    lifter = Lifter(thumb, fc)
    for seed in seeds:
        if lifter.trial(seed, sampler, pca):
            break
    else:
        raise SystemExit("no successful lift to render")
    m, d = lifter.model, lifter.data
    renderer = mujoco.Renderer(m, 360, 480)
    imgs = []

    class Recorder(list):
        def append(self, t):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = d.qpos[lifter.gen.idx.obj_qpos:lifter.gen.idx.obj_qpos + 3]
            cam.distance, cam.azimuth, cam.elevation = 0.35, 135, -20
            renderer.update_scene(d, cam)
            imgs.append(Image.fromarray(renderer.render()))
            super().append(t)

    lifter.trial(seed, sampler, pca, frames=Recorder())
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=40, loop=0)
    renderer.close()
    print(f"wrote {path} (seed {seed}, {len(imgs)} frames)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trials", type=int, default=600)
    p.add_argument("--samplers", default="prior,synergy-3")
    p.add_argument("--thumbs", default="stock")
    p.add_argument("--finger-collisions", action="store_true")
    p.add_argument("--synergies", default="out/synergies.npz")
    p.add_argument("--gif", default=None, help="render one successful lift (first thumb and sampler) instead")
    p.add_argument("--out", default=None, help="save per-trial results (npz)")
    args = p.parse_args()
    samplers, thumbs = args.samplers.split(","), args.thumbs.split(",")
    if args.gif:
        pca = dict(np.load(args.synergies)) if samplers[0] != "prior" else None
        render_gif(args.gif, thumbs[0], samplers[0], args.finger_collisions, pca, range(args.trials))
        return
    jobs = [(s, t, sm, args.finger_collisions, args.synergies) for t in thumbs for sm in samplers
            for s in range(args.trials)]
    results = {}
    with mp.Pool() as pool:
        for thumb, sampler, ok in pool.imap_unordered(_work, jobs, chunksize=8):
            results.setdefault((thumb, sampler), []).append(ok)
    for (thumb, sampler), oks in sorted(results.items()):
        print(f"thumb={thumb:8s} sampler={sampler:10s} lifted {np.mean(oks):5.1%} of {len(oks)}")
    if args.out:
        np.savez(args.out, **{f"{t}|{s}": np.array(v) for (t, s), v in results.items()})


if __name__ == "__main__":
    main()
