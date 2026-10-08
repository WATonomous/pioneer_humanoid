"""Pick an object up off the table, the way the hand will once it's on the arm: reach above, descend,
close, lift.

The wrist stands in for the arm's end effector: x / y / z slide joints and a yaw hinge, each driven by a
stiff position servo (moving joints, not a teleported mocap body, so contact friction sees the hand's
real velocity). The object rests on the table under gravity the whole time -- no fixture.

The geometry decides the approach: palm down, the stock thumb hangs ~10.6 cm below the palm, so the palm
stops ~11 cm above the table; the object sits under the palm between the hanging thumb and the fingers,
which curl down over it (a claw grasp). The thumb's MCP_A is kept low on the way down so it doesn't sweep
into the object, then the autograsp closes from the pre-shape (grasp_gen.GraspGen.close).

One trial: grasp_gen's object (sphere / box / cylinder lying on its side, random yaw, <= 8 cm tall) on
the table; the hand's palm point lands over it with +-JITTER_M; reach -> descend -> close -> settle ->
lift LIFT_M -> hold. Success: the object rose >= SUCCESS_RISE_M, ended within HELD_M of where it sat in
the hand, and no step went unstable.

    python -m hand_synergies.pick --trials 1000 --samplers prior,synergy-3 --finger-collisions
    MUJOCO_GL=osmesa python -m hand_synergies.pick --gif out/pick.gif --samplers synergy-3
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import mujoco
import numpy as np
from pioneer_humanoid.mujoco_hand import PALM_BODY, hand_spec

from .scene import MAX_SIZE, TIMESTEP, _GEOM_TYPES, set_object, stiffen_contacts

JITTER_M = 0.01
CLEARANCE_M = 0.004        # the hand's lowest point (bounding boxes) stays this far above the table
ABOVE_M = 0.08             # start this far above the grasp height
DESCEND_S = 0.6
THUMB_DESCENT_MAX = 0.3    # rad, MCP_A_thumb cap while descending
SETTLE_S = 0.2
LIFT_M = 0.10
LIFT_S = 1.0
HOLD_S = 1.0
SUCCESS_RISE_M = 0.08
HELD_M = 0.03
WRIST_AXES = ("x", "y", "z")
PEDESTAL_HALF = [0.035, 0.035, 0.025]
BUTTON_TRAVEL = 0.01       # m
BUTTON_SPRING = 300.0      # N/m: ~3 N to bottom it out
# Keyboard: a big one -- this index fingertip is ~2 cm wide, a laptop key pitch is 19 mm.
KEY_PITCH = 0.024
KEY_CAP = 0.018
KEY_H = 0.008
KEY_BASE_Z = 0.012
KEY_TRAVEL = 0.006
KEY_SPRING = 150.0         # N/m: ~1 N at the bottom


def make_model(thumb: str = "stock", finger_collisions: bool = False, torque_limit: float | None = None,
               width: int = 480, height: int = 360, extras: bool = False, keyboard: bool = False) -> mujoco.MjModel:
    """``extras``: also a box to stack on ("pedestal", mocap) and a spring-loaded push button
    ("button_base" mocap + "button" on a slide joint), parked out of the way (see actions.py).
    ``keyboard``: a 26-key QWERTY keyboard of spring keys ("key_a" ... bodies and joints) at the origin."""
    from .thumb import thumb_kwargs

    spec = mujoco.MjSpec()
    spec.modelname = "hand_pick"
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.worldbody.add_light(pos=[0.3, -0.3, 0.8], dir=[-0.3, 0.3, -1], diffuse=[0.8, 0.8, 0.8])
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.5, 0.5, 0.01],
                            rgba=[0.55, 0.5, 0.45, 1], friction=[1.0, 0.01, 0.001])
    # Arm stand-in: x, y, z slides and a yaw hinge with stiff position servos.
    wrist = spec.worldbody.add_body(name="wrist")
    for axis, vec in zip(WRIST_AXES, np.eye(3)):
        wrist.add_joint(name=f"wrist_{axis}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=vec.tolist(),
                        range=[-1.0, 1.0])
        act = spec.add_actuator(name=f"wrist_{axis}", target=f"wrist_{axis}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=20000.0, kv=600.0)
    for name, axis in (("wrist_yaw", [0, 0, 1]), ("wrist_pitch", [1, 0, 0])):
        # the spec compiler reads hinge ranges in degrees (a range of +-3.2 was +-3.2 deg, and the soft
        # limit fought the servo: "-30 deg" wasn't)
        wrist.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis, range=[-180.0, 180.0])
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=500.0, kv=20.0)
    wrist.add_frame().attach_body(
        hand_spec(**thumb_kwargs(thumb), finger_collisions=finger_collisions,
                  torque_limit=torque_limit).body(PALM_BODY), "", "")
    obj = spec.worldbody.add_body(name="object", pos=[0.3, 0.3, 0.05])
    obj.add_freejoint(name="object")
    for name, gtype in _GEOM_TYPES.items():
        obj.add_geom(name=name, type=gtype, size=MAX_SIZE[name], rgba=[0.9, 0.55, 0.2, 1],
                     friction=[1.0, 0.01, 0.001], density=0)
    obj.mass = 0.05
    obj.inertia = [1e-5, 1e-5, 1e-5]
    obj.explicitinertial = True
    obj.ipos = [0.0, 0.0, 0.0]  # see scene.make_model: left alone it takes the spawn position
    obj.iquat = [1.0, 0.0, 0.0, 0.0]
    if extras:
        ped = spec.worldbody.add_body(name="pedestal", mocap=True, pos=[2.0, 0.0, PEDESTAL_HALF[2]])
        ped.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=PEDESTAL_HALF,
                     rgba=[0.35, 0.45, 0.6, 1], friction=[1.0, 0.01, 0.001])
        base = spec.worldbody.add_body(name="button_base", mocap=True, pos=[2.0, 1.0, 0.0])
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.01], pos=[0, 0, 0.01],
                      rgba=[0.3, 0.3, 0.32, 1])
        button = base.add_body(name="button", pos=[0, 0, 0.02])
        button.add_joint(name="button", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                         range=[-BUTTON_TRAVEL, 0.0], stiffness=BUTTON_SPRING, damping=1.0)
        button.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0.006, 0], pos=[0, 0, 0.006],
                        rgba=[0.85, 0.15, 0.15, 1], density=300)
    if keyboard:
        spec.worldbody.add_geom(name="keyboard", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[0.15, 0.045, KEY_BASE_Z / 2], pos=[0, -KEY_PITCH, KEY_BASE_Z / 2],
                                rgba=[0.15, 0.15, 0.17, 1])
        for c, (x, y) in key_layout().items():
            key = spec.worldbody.add_body(name=f"key_{c}", pos=[x, y, KEY_BASE_Z])
            key.add_joint(name=f"key_{c}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                          range=[-KEY_TRAVEL, 0.0], stiffness=KEY_SPRING, damping=0.5)
            # keys collide with the hand only (contype bit 2), not with the plate they sit on
            key.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[KEY_CAP / 2, KEY_CAP / 2, KEY_H / 2],
                         pos=[0, 0, KEY_H / 2], rgba=[0.88, 0.88, 0.9, 1], density=300, contype=0, conaffinity=2)
    stiffen_contacts(spec)
    return spec.compile()


def key_layout() -> dict[str, tuple[float, float]]:
    """Key centres (x, y) of a staggered QWERTY layout, top row at y = 0."""
    out = {}
    for row, (keys, shift) in enumerate((("qwertyuiop", 0.0), ("asdfghjkl", 0.25), ("zxcvbnm", 0.75))):
        for i, c in enumerate(keys):
            out[c] = ((i + shift - 4.5) * KEY_PITCH, -row * KEY_PITCH)
    return out


class Picker:
    def __init__(self, thumb: str = "stock", finger_collisions: bool = False, torque_limit: float | None = None,
                 pitch_deg: float = 0.0, extras: bool = False, keyboard: bool = False):
        from .grasp_gen import GraspGen

        self.pitch = np.radians(pitch_deg)
        self.frames = None
        self.model = make_model(thumb, finger_collisions, torque_limit, extras=extras, keyboard=keyboard)
        self.gen = GraspGen(model=self.model)  # its autograsp, on this model
        self.data = self.gen.data
        m = self.model
        self.wrist_q = np.array([m.joint(f"wrist_{a}").qposadr[0] for a in WRIST_AXES])
        self.wrist_act = np.array([m.actuator(f"wrist_{a}").id for a in WRIST_AXES])
        self.pitch_q = m.joint("wrist_pitch").qposadr[0]
        self.pitch_act = m.actuator("wrist_pitch").id
        self.hand_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] not in
                           (0, self.gen.idx.obj_body)]
        self.tips = (m.site("tip_thumb_distal").id, m.site("tip_distal_2").id)

    def _object(self, rng):
        """grasp_gen's object, resting on the table: (kind, size, pos, quat)."""
        kind, size, _, _ = self.gen.sample_object(rng)
        yaw = rng.uniform(-np.pi, np.pi)
        qyaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        if kind == "cylinder":  # on its side: axis along x before the yaw
            quat = np.empty(4)
            mujoco.mju_mulQuat(quat, qyaw, np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]))
            half_h = size[0]
        else:
            quat = qyaw
            half_h = size[0] if kind == "sphere" else size[2]
        return kind, size, np.array([0.0, 0.0, half_h]), quat, half_h

    def _place(self, q0, descend_q, obj_pos, half_h, jitter) -> np.ndarray:
        """Wrist xyz for the grasp: the thumb-tip / middle-fingertip midpoint (pre-shape ``q0``) over the
        object's centre (+ ``jitter``) at its height, raised if the hand (``descend_q``) would go below the
        table."""
        m, d, gen = self.model, self.data, self.gen
        mujoco.mj_resetData(m, d)
        d.qpos[self.pitch_q] = self.pitch
        d.qpos[gen.idx.qpos] = q0
        mujoco.mj_kinematics(m, d)
        mid = (d.site_xpos[self.tips[0]] + d.site_xpos[self.tips[1]]) / 2  # wrist at the origin
        wrist = np.array([obj_pos[0] + jitter[0], obj_pos[1] + jitter[1], half_h]) - mid
        d.qpos[gen.idx.qpos] = descend_q
        mujoco.mj_kinematics(m, d)
        low = min(self._geom_bottom(g) for g in self.hand_geoms)  # hand's lowest point, wrist at the origin
        wrist[2] = max(wrist[2], CLEARANCE_M - low)
        return wrist

    def _geom_bottom(self, g: int) -> float:
        """Lowest world z of geom g's bounding box (current kinematics)."""
        m, d = self.model, self.data
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        rot = d.geom_xmat[g].reshape(3, 3)
        return float(d.geom_xpos[g][2] + rot[2] @ c - np.abs(rot[2]) @ h)

    def _set_wrist(self, target: np.ndarray) -> None:
        self.data.ctrl[self.wrist_act] = target

    def wrist_pos(self) -> np.ndarray:
        return self.data.xpos[self.model.body("wrist").id].copy()

    # --- motion helpers (the actions in actions.py chain these) ---------------------------------------
    def record(self) -> None:
        if self.frames is not None and int(round(self.data.time / TIMESTEP)) % 40 == 0:
            self.frames.append(self.data.time)

    def step(self, seconds: float, hand: np.ndarray | None = None) -> None:
        for _ in range(int(seconds / TIMESTEP)):
            if hand is not None:
                self.data.ctrl[self.gen.idx.act] = hand
            mujoco.mj_step(self.model, self.data)
            self.record()

    def move(self, a, b, seconds: float, hand: np.ndarray | None = None) -> None:
        """Wrist from xyz ``a`` to ``b`` in a straight line over ``seconds`` (hand targets held)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = int(seconds / TIMESTEP)
        for i in range(n):
            self._set_wrist(a + (b - a) * (i + 1) / n)
            if hand is not None:
                self.data.ctrl[self.gen.idx.act] = hand
            mujoco.mj_step(self.model, self.data)
            self.record()

    def obj_pos(self) -> np.ndarray:
        a = self.gen.idx.obj_qpos
        return self.data.qpos[a:a + 3].copy()

    def grasp_and_lift(self, seed: int, sampler: str, pca: dict | None, frames: list | None = None,
                       obj_xy=(0.0, 0.0), setup=None) -> dict:
        """Reach, descend, close, lift LIFT_M, hold. Returns the grasp's details and whether it held.

        ``setup(start_pos)``: called right after the reset, e.g. to place other things in the scene."""
        from .grasp_gen import GRASP_TYPES
        from .search import OPEN_SCALE, _sample

        self.frames = frames
        rng = np.random.default_rng(seed)  # same object and placement for every sampler / thumb
        m, d, gen = self.model, self.data, self.gen
        kind, size, pos, quat, half_h = self._object(rng)
        pos[:2] += obj_xy
        jitter = rng.uniform(-JITTER_M, JITTER_M, size=2)

        q0 = _sample(sampler, np.random.default_rng([seed, 1]), pca, gen)
        if sampler != "prior":
            q0 = q0.copy()
            q0[gen.close_j] *= OPEN_SCALE
        q0 = np.clip(q0, *gen.ranges.T)
        descend_q = q0.copy()
        descend_q[1] = min(descend_q[1], THUMB_DESCENT_MAX)  # MCP_A_thumb (JOINT_NAMES[1])
        grasp = self._place(q0, descend_q, pos, half_h, jitter)
        start = grasp + [0, 0, ABOVE_M]

        mujoco.mj_resetData(m, d)
        d.qpos[self.pitch_q] = self.pitch
        d.ctrl[self.pitch_act] = self.pitch
        set_object(m, gen.idx, kind, size)
        d.qpos[self.wrist_q] = start
        self._set_wrist(start)
        d.qpos[gen.idx.qpos] = descend_q
        d.ctrl[gen.idx.act] = descend_q
        d.qpos[gen.idx.obj_qpos:gen.idx.obj_qpos + 3] = pos
        d.qpos[gen.idx.obj_qpos + 3:gen.idx.obj_qpos + 7] = quat
        if setup is not None:
            setup(pos)
        mujoco.mj_forward(m, d)

        self.step(0.2)  # object settles on the table
        self.move(start, grasp, DESCEND_S)
        ctrl = gen.close(q0, GRASP_TYPES["power"][0], after_step=self.record)
        self.step(SETTLE_S)
        z0 = self.obj_pos()[2]
        in_hand = self.obj_pos() - self.wrist_pos()
        lifted = grasp + [0, 0, LIFT_M]
        self.move(grasp, lifted, LIFT_S, hand=ctrl)
        self.step(HOLD_S, hand=ctrl)
        obj = self.obj_pos()
        ok = (not gen.unstable() and obj[2] - z0 >= SUCCESS_RISE_M
              and np.linalg.norm(obj - self.wrist_pos() - in_hand) < HELD_M)
        return dict(ok=bool(ok), ctrl=ctrl, grasp=grasp, lifted=lifted, kind=kind, size=size, half_h=half_h,
                    start_pos=pos, in_hand=in_hand)

    def trial(self, seed: int, sampler: str, pca: dict | None, frames: list | None = None) -> bool:
        return self.grasp_and_lift(seed, sampler, pca, frames)["ok"]


_PICKERS = {}


def _work(args):
    seed, thumb, sampler, fc, pca_path, pitch = args
    key = (thumb, fc, pitch)
    if key not in _PICKERS:
        _PICKERS[key] = Picker(thumb, fc, pitch_deg=pitch)
    pca = dict(np.load(pca_path)) if sampler != "prior" else None
    return thumb, sampler, pitch, _PICKERS[key].trial(seed, sampler, pca)


def render_gif(path: str, thumb: str, sampler: str, fc: bool, pca: dict | None, seeds, pitch: float) -> None:
    """Re-run the first successful seed with a fixed camera, a frame every 40 ms."""
    from PIL import Image

    picker = Picker(thumb, fc, pitch_deg=pitch)
    for seed in seeds:
        if picker.trial(seed, sampler, pca):
            break
    else:
        raise SystemExit("no successful pick to render")
    m, d = picker.model, picker.data
    renderer = mujoco.Renderer(m, 360, 480)
    imgs = []

    class Recorder(list):
        def append(self, t):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.0, 0.0, 0.08]
            cam.distance, cam.azimuth, cam.elevation = 0.45, 135, -15
            renderer.update_scene(d, cam)
            imgs.append(Image.fromarray(renderer.render()))
            super().append(t)

    picker.trial(seed, sampler, pca, frames=Recorder())
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=40, loop=0)
    renderer.close()
    print(f"wrote {path} (seed {seed}, {len(imgs)} frames)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--samplers", default="prior,synergy-3")
    p.add_argument("--thumbs", default="stock")
    p.add_argument("--finger-collisions", action="store_true")
    p.add_argument("--synergies", default="out/synergies.npz")
    p.add_argument("--pitch", default="0", help="wrist pitch(es) about world X, deg, comma-separated")
    p.add_argument("--gif", default=None, help="render one successful pick (first thumb, sampler, pitch) instead")
    args = p.parse_args()
    samplers, thumbs = args.samplers.split(","), args.thumbs.split(",")
    pitches = [float(x) for x in args.pitch.split(",")]
    if args.gif:
        pca = dict(np.load(args.synergies)) if samplers[0] != "prior" else None
        render_gif(args.gif, thumbs[0], samplers[0], args.finger_collisions, pca, range(args.trials), pitches[0])
        return
    jobs = [(s, t, sm, args.finger_collisions, args.synergies, pt) for t in thumbs for sm in samplers
            for pt in pitches for s in range(args.trials)]
    results = {}
    with mp.Pool() as pool:
        for thumb, sampler, pitch, ok in pool.imap_unordered(_work, jobs, chunksize=8):
            results.setdefault((thumb, sampler, pitch), []).append(ok)
    for (thumb, sampler, pitch), oks in sorted(results.items()):
        print(f"thumb={thumb:8s} sampler={sampler:10s} pitch={pitch:+4.0f} picked up {np.mean(oks):5.1%} of {len(oks)}")


if __name__ == "__main__":
    main()
