"""Grasp, release, lift: the hand (palm down, on a wrist that slides vertically) closes on an object held by
a fixture, the fixture (a weld) lets go, and the wrist lifts LIFT_M over LIFT_S and holds for HOLD_S.

The objects and their placement under the palm are grasp_gen's (sample_object). Success: the object rose at least SUCCESS_RISE_M, ended within HELD_M of where it would be had it
not moved in the hand, and no step went unstable (MuJoCo silently resets an unstable simulation; a squeezed
cylinder can spin up). Unlike grasp_gen's shake test this carries the object, under real gravity, while the
hand accelerates.

(A first version lifted tall objects off a table: the stock thumb hangs ~10.6 cm below the palm, so only
tall objects fit, and none of them was lifted.)

Samplers (pre-shapes) are search.py's: ``prior`` (grasp_gen's random pre-shape) or ``synergy-k`` /
``gauss-20`` (a fitted grasp posture opened to the prior's aperture).

    python -m hand_synergies.lift --trials 1000 --samplers prior,synergy-3 --thumbs stock,near36 --finger-collisions
    MUJOCO_GL=osmesa python -m hand_synergies.lift --gif out/lift.gif --trials 40 --samplers synergy-3
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np
from pioneer_humanoid.mujoco_hand import PALM_BODY, hand_spec

from .scene import (MAX_SIZE, TIMESTEP, _GEOM_TYPES, set_object,
                    stiffen_contacts)

LIFT_M = 0.10
LIFT_S = 1.0
HOLD_S = 1.0
SETTLE_S = 0.3
SUCCESS_RISE_M = 0.08
HELD_M = 0.03               # and end within this of where it would be had it not moved in the hand
WRIST0 = np.array([0.0, 0.0, 0.35])
RELEASE_S = 0.15            # gravity-free settle after the fixture lets go


def make_model(thumb: str = "stock", finger_collisions: bool = False, width: int = 480,
               height: int = 360, torque_limit: float | None = None) -> mujoco.MjModel:
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
    # The wrist rides a vertical slide joint driven by a stiff position servo. Teleporting a mocap wrist
    # instead gives the hand zero velocity as far as contact friction is concerned, and the object slips.
    wrist = spec.worldbody.add_body(name="wrist", pos=WRIST0.tolist())
    wrist.add_joint(name="lift", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.2, 0.3],
                    damping=0.0, armature=0.0)
    lift = spec.add_actuator(name="lift", target="lift", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    lift.set_to_position(kp=20000.0, kv=600.0)
    wrist.add_frame().attach_body(
        hand_spec(**thumb_kwargs(thumb), finger_collisions=finger_collisions,
                  torque_limit=torque_limit).body(PALM_BODY), "", "")
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
    # Fixture: a mocap body the object is welded to until release. A weld lets the solver balance the
    # squeezing fingers; resetting the object's pose every step (grasp_gen's pin) lets them sink in, and
    # the stored overlap fires a small object off at tens of m/s when it's let go.
    spec.worldbody.add_body(name="fixture", mocap=True, pos=[0, 0.1, 0.05])
    weld = spec.add_equality(name="fixture", type=mujoco.mjtEq.mjEQ_WELD, name1="object", name2="fixture",
                             objtype=mujoco.mjtObj.mjOBJ_BODY)
    weld.solref = [0.004, 1.0]
    stiffen_contacts(spec)
    return spec.compile()


class Lifter:
    def __init__(self, thumb: str, finger_collisions: bool, torque_limit: float | None = None):
        from .grasp_gen import GraspGen

        self.model = make_model(thumb, finger_collisions, torque_limit=torque_limit)
        self.gen = GraspGen(model=self.model)  # reuses its autograsp on this model
        self.data = self.gen.data
        self.lift_q = self.model.joint("lift").qposadr[0]
        self.lift_act = self.model.actuator("lift").id
        self.fixture = self.model.body("fixture").mocapid[0]
        self.weld = self.model.equality("fixture").id
        # weld the object's frame onto the fixture's frame (anchor at the origin, identity relative pose)
        self.model.eq_data[self.weld, :10] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        self.table = self.model.geom("table").id

    def wrist_pos(self) -> np.ndarray:
        return self.data.xpos[self.model.body("wrist").id].copy()

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
        kind, size, pos_h, quat = gen.sample_object(rng)  # grasp_gen's objects, in the hand frame
        pos = WRIST0 + pos_h  # the wrist frame is the hand frame (palm down)

        q = _sample(sampler, np.random.default_rng([seed, 1]), pca, gen)
        if sampler != "prior":
            q = q.copy()
            q[gen.close_j] *= OPEN_SCALE
        for delta in (0.0, *OPEN_STEPS):  # open until the hand clears the object
            q0 = q.copy()
            q0[gen.close_j] -= gen.close_sign * delta
            q0 = np.clip(q0, *gen.ranges.T)
            mujoco.mj_resetData(m, d)
            set_object(m, gen.idx, kind, size)
            d.qpos[gen.idx.qpos] = q0
            d.ctrl[gen.idx.act] = q0
            d.qpos[gen.idx.obj_qpos:gen.idx.obj_qpos + 3] = pos
            d.qpos[gen.idx.obj_qpos + 3:gen.idx.obj_qpos + 7] = quat
            d.mocap_pos[self.fixture] = pos
            d.mocap_quat[self.fixture] = quat
            d.eq_active[self.weld] = 1
            if self._start_clear():
                break
        else:
            return False

        def record():
            if frames is not None and int(round(d.time / TIMESTEP)) % 40 == 0:
                frames.append(d.time)

        # Close on the object while the fixture holds it, squeeze, settle, then let go.
        ctrl = gen.close(q0, GRASP_TYPES["power"][0], after_step=record)
        for _ in range(int(SETTLE_S / TIMESTEP)):
            mujoco.mj_step(m, d)
            record()
        return self._release_and_lift(ctrl, record)

    def lift_grasp(self, rec: dict, q_target: np.ndarray | None = None) -> bool:
        """Carry a stored grasp_gen grasp: hand at its posture, object at its pose (welded), targets
        ``q_target`` (default: its own commanded targets), release, settle without gravity, lift."""
        m, d, gen = self.model, self.data, self.gen
        kind = str(rec["kind"])
        size = rec["size"][: {"sphere": 1, "cylinder": 2, "box": 3}[kind]]
        pos = WRIST0 + rec["obj_pos"]
        mujoco.mj_resetData(m, d)
        set_object(m, gen.idx, kind, size)
        d.qpos[gen.idx.qpos] = rec["q"]
        ctrl = rec["ctrl"] if q_target is None else np.clip(q_target + (rec["ctrl"] - rec["q"]), *gen.ranges.T)
        d.ctrl[gen.idx.act] = ctrl
        d.qpos[gen.idx.obj_qpos:gen.idx.obj_qpos + 3] = pos
        d.qpos[gen.idx.obj_qpos + 3:gen.idx.obj_qpos + 7] = rec["obj_quat"]
        d.mocap_pos[self.fixture] = pos
        d.mocap_quat[self.fixture] = rec["obj_quat"]
        d.eq_active[self.weld] = 1
        mujoco.mj_step(m, d, nstep=int(SETTLE_S / TIMESTEP))
        return self._release_and_lift(ctrl)

    def _release_and_lift(self, ctrl: np.ndarray, record=lambda: None) -> bool:
        m, d, gen = self.model, self.data, self.gen
        d.eq_active[self.weld] = 0
        g = m.opt.gravity.copy()
        m.opt.gravity[:] = 0.0
        for _ in range(int(RELEASE_S / TIMESTEP)):
            mujoco.mj_step(m, d)
            record()
        m.opt.gravity[:] = g
        z0 = d.qpos[gen.idx.obj_qpos + 2]
        pos_h = d.qpos[gen.idx.obj_qpos:gen.idx.obj_qpos + 3] - self.wrist_pos()  # where it settled
        n = int(LIFT_S / TIMESTEP)
        for i in range(n + int(HOLD_S / TIMESTEP)):
            d.ctrl[self.lift_act] = LIFT_M * min(1.0, (i + 1) / n)
            d.ctrl[gen.idx.act] = ctrl
            mujoco.mj_step(m, d)
            record()
        if gen.unstable():
            return False  # MuJoCo reset an unstable step: whatever happened after is not a lift
        obj = d.qpos[gen.idx.obj_qpos:gen.idx.obj_qpos + 3]
        carried = self.wrist_pos() + pos_h
        return bool(obj[2] - z0 >= SUCCESS_RISE_M and np.linalg.norm(obj - carried) < HELD_M)


_LIFTERS = {}


def _work(args):
    seed, thumb, sampler, fc, pca_path = args
    key = (thumb, fc)
    if key not in _LIFTERS:
        _LIFTERS[key] = Lifter(thumb, fc)
    pca = dict(np.load(pca_path)) if sampler != "prior" else None
    return thumb, sampler, _LIFTERS[key].trial(seed, sampler, pca)


def _carry(args):
    i, k, grasps_path, pca_path = args
    from .synergies import project

    if "stored" not in _LIFTERS:
        _LIFTERS["stored"] = (Lifter("stock", False), dict(np.load(grasps_path)), dict(np.load(pca_path)))
    lifter, grasps, pca = _LIFTERS["stored"]
    rec = {f: grasps[f][i] for f in ("q", "ctrl", "kind", "size", "obj_pos", "obj_quat")}
    return k, lifter.lift_grasp(rec, None if k == 20 else project(grasps["q"][i][None], pca, k)[0])


def carry_stored(grasps_path: str, pca_path: str, n: int) -> None:
    """Can grasp_gen's shake-test-stable grasps be carried, as is and with postures rebuilt from k PCs?"""
    count = len(np.load(grasps_path)["q"])
    pick = np.random.default_rng(0).choice(count, size=min(n, count), replace=False)
    ks = (20, 10, 5, 3, 1)
    held = {k: [] for k in ks}
    with mp.Pool() as pool:
        for k, ok in pool.imap_unordered(_carry, [(int(i), k, grasps_path, pca_path) for k in ks for i in pick],
                                         chunksize=8):
            held[k].append(ok)
    for k in ks:
        label = "as stored" if k == 20 else f"{k}-PC posture"
        print(f"stable grasps carried, {label:13s}: {np.mean(held[k]):5.1%} of {len(held[k])}")


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
    p.add_argument("--stored", type=int, default=0,
                   help="instead: carry this many of --grasps' stable grasps, rebuilt from k PCs (k=20: as is)")
    p.add_argument("--grasps", default="out/grasps.npz")
    args = p.parse_args()
    if args.stored:
        carry_stored(args.grasps, args.synergies, args.stored)
        return
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
