"""Generate stable grasps of random objects with the pioneer hand (CPU, multiprocess).

One trial:
  1. Random pre-shape (finger fan, thumb circumduction/opposition, slight curl), random object
     (sphere / cylinder / box, random size and orientation) under the palm, gravity off.
  2. GraspIt-style autograsp: the closing joints of the participating digits ramp toward the
     palm; when a link touches the object, every closing joint between that link and the palm
     stops (distal joints keep curling to wrap around it). The object is pinned in place while
     the hand closes (as GraspIt plans against a static object); otherwise the first finger
     to arrive pushes it away.
  3. Squeeze: stopped joints get SQUEEZE_RAD more target to press on the object.
  4. Shake test: from the settled grasp, gravity along +-X, +-Y, +-Z for HOLD_TEST_S each. The
     grasp is kept only if the object moves < MAX_DRIFT_M and turns < MAX_TURN_RAD in all six.

The sampling choices here (fan/curl ranges, grasp types) are the prior the synergies inherit;
they are all in SAMPLING below.

    python -m hand_synergies.grasp_gen --trials 4000 --out grasps.npz
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import mujoco
import numpy as np
from pioneer_humanoid.mujoco_hand import JOINT_NAMES, PALM_BODY, joint_ranges

from .scene import OBJECT_TYPES, PALM_SURFACE_Z, HandIndex, half_height, make_model, set_object

# Closing joints per digit, proximal to distal, with the sign that curls them toward the palm.
# The thumb "closes" by opposing (MCP_A_thumb sweeps it under the palm) then curling.
DIGITS = {
    "thumb": (("MCP_A_thumb", +1), ("PIP_thumb", -1), ("DIP_thumb", -1)),
    "index": (("MCP_1", +1), ("PIP_1", -1), ("DIP_1", +1)),
    "middle": (("MCP_2", +1), ("PIP_2", -1), ("DIP_2", +1)),
    "ring": (("MCP_3", -1), ("PIP_3", -1), ("DIP_3", +1)),
    "pinky": (("MCP_4", +1), ("PIP_4", -1), ("DIP_4", +1)),
}
CONTACT_GROUPS = ("palm", *DIGITS)

GRASP_TYPES = {  # name: (digits that close, sampling weight)
    "power": (("thumb", "index", "middle", "ring", "pinky"), 0.5),
    "tripod": (("thumb", "index", "middle"), 0.25),
    "pinch": (("thumb", "index"), 0.25),
}

SAMPLING = dict(
    fan=(-0.1, 0.47),          # rad; MCP_A_1 = -fan, MCP_A_2 = -fan/3, MCP_A_3 = +fan/3, MCP_A_4 = +fan
    fan_noise=0.08,            # rad, per-finger N(0, .) on top of the fan
    thumb_oppose=(0.0, 0.8),   # rad, initial MCP_A_thumb
    precurl=(0.0, 0.25),       # fraction of each flex joint's range
    sphere_r=(0.012, 0.04),
    cylinder_r=(0.006, 0.035),
    cylinder_half_h=(0.03, 0.08),
    box_half=(0.008, 0.035),
    obj_x=(-0.02, 0.06),       # hand frame; index is at -x, pinky at +x
    obj_y=(0.05, 0.17),        # palm root ~0.0 .. fingertips ~0.21
    obj_gap=(0.0, 0.015),      # object top below the palm surface
)

CTRL_EVERY = 5               # physics steps per control update (5 ms)
CLOSE_SPEED = 1.5            # rad/s
CLOSE_TIMEOUT_S = 1.5
SQUEEZE_RAD = 0.08
SETTLE_S = 0.3
HOLD_TEST_S = 0.5
MAX_DRIFT_M = 0.015
MAX_TURN_RAD = 0.3
GRAVITY = 9.81


class GraspGen:
    def __init__(self, thumb: str = "stock", finger_collisions: bool = False, model: mujoco.MjModel | None = None):
        """``model``: use this compiled model (hand + a free body "object") instead of scene.make_model."""
        self.model = model if model is not None else make_model(thumb=thumb, finger_collisions=finger_collisions)
        self.data = mujoco.MjData(self.model)
        self.idx = HandIndex.of(self.model)
        self.ranges = joint_ranges(self.model)
        m = self.model
        self.palm = m.body(PALM_BODY).id
        jid = {n: i for i, n in enumerate(JOINT_NAMES)}
        # closing joints: index into JOINT_NAMES, sign, digit
        self.close_j, self.close_sign, self.close_digit = [], [], []
        for digit, chain in DIGITS.items():
            for name, sign in chain:
                self.close_j.append(jid[name])
                self.close_sign.append(sign)
                self.close_digit.append(digit)
        self.close_j = np.array(self.close_j)
        self.close_sign = np.array(self.close_sign, float)
        self.close_digit = np.array(self.close_digit)
        # closed limit of each closing joint
        self.close_limit = np.where(self.close_sign > 0, self.ranges[self.close_j, 1], self.ranges[self.close_j, 0])
        # per hand body: which closing joints lie between it and the palm, and its contact group
        joint_of_body = {m.jnt_bodyid[m.joint(n).id]: i for i, n in enumerate(JOINT_NAMES)}
        close_pos = {j: k for k, j in enumerate(self.close_j)}
        self.body_stops: dict[int, list[int]] = {}
        self.body_group: dict[int, str] = {self.palm: "palm"}
        for b in range(m.nbody):
            if b in (0, self.palm, self.idx.obj_body):
                continue
            stops, cur, group = [], b, None
            while cur != self.palm and cur != 0:
                j = joint_of_body.get(cur)
                if j is not None:
                    if j in close_pos:
                        stops.append(close_pos[j])
                    name = JOINT_NAMES[j]
                    group = group or self._digit_of(name)
                cur = m.body_parentid[cur]
            if cur == self.palm:
                self.body_stops[b] = stops
                self.body_group[b] = group

    @staticmethod
    def _digit_of(joint_name: str) -> str:
        if joint_name.endswith("thumb") or joint_name == "circumduction":
            return "thumb"
        return ("index", "middle", "ring", "pinky")[int(joint_name[-1]) - 1]

    # --- sampling -------------------------------------------------------------------------
    def sample_preshape(self, rng: np.random.Generator) -> np.ndarray:
        s = SAMPLING
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        q = np.zeros(len(JOINT_NAMES))
        jid = {n: i for i, n in enumerate(JOINT_NAMES)}
        fan = rng.uniform(*s["fan"])
        for name, k in (("MCP_A_1", -1), ("MCP_A_2", -1 / 3), ("MCP_A_3", 1 / 3), ("MCP_A_4", 1)):
            q[jid[name]] = k * fan + rng.normal(0, s["fan_noise"])
        q[jid["circumduction"]] = rng.uniform(lo[0], hi[0])
        q[jid["MCP_A_thumb"]] = rng.uniform(*s["thumb_oppose"])
        for k, j in enumerate(self.close_j):
            if JOINT_NAMES[j] == "MCP_A_thumb":
                continue
            span = hi[j] - lo[j]
            q[j] = self.close_sign[k] * rng.uniform(*s["precurl"]) * span
        return np.clip(q, lo, hi)

    def sample_object(self, rng: np.random.Generator):
        s = SAMPLING
        kind = OBJECT_TYPES[rng.integers(len(OBJECT_TYPES))]
        if kind == "sphere":
            size = np.array([rng.uniform(*s["sphere_r"])])
        elif kind == "cylinder":
            size = np.array([rng.uniform(*s["cylinder_r"]), rng.uniform(*s["cylinder_half_h"])])
        else:
            size = rng.uniform(*s["box_half"], size=3)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        pos = np.array([
            rng.uniform(*s["obj_x"]),
            rng.uniform(*s["obj_y"]),
            PALM_SURFACE_Z - half_height(kind, size, quat) - rng.uniform(*s["obj_gap"]),
        ])
        return kind, size, pos, quat

    # --- simulation -----------------------------------------------------------------------
    def _contacts(self):
        """(bodies of the hand touching the object, contact groups touching it)."""
        m, d, ob = self.model, self.data, self.idx.obj_body
        bodies = set()
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if ob in (b1, b2) and c.dist < 0.0005:
                bodies.add(b2 if b1 == ob else b1)
        return bodies, {self.body_group.get(b) for b in bodies} - {None}

    def _pin_object(self, pos: np.ndarray, quat: np.ndarray) -> None:
        a, v = self.idx.obj_qpos, self.idx.obj_dof
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = quat
        self.data.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _object_pose(self):
        a = self.idx.obj_qpos
        return self.data.qpos[a:a + 3].copy(), self.data.qpos[a + 3:a + 7].copy()

    def trial(self, seed: int) -> dict | None:
        rng = np.random.default_rng(seed)
        grasp_type = rng.choice(list(GRASP_TYPES), p=[w for _, w in GRASP_TYPES.values()])
        q0 = self.sample_preshape(rng)
        kind, size, pos, quat = self.sample_object(rng)
        result = self.grasp(q0, kind, size, pos, quat, GRASP_TYPES[grasp_type][0])
        if result is not None:
            result.update(grasp_type=grasp_type, seed=seed)
        return result

    def close(self, q0: np.ndarray, digits, after_step=None) -> np.ndarray:
        """Autograsp from the current state with targets ``q0``, then squeeze; returns the final targets.

        ``after_step``: called after every control step (e.g. to pin the object).
        """
        m, d, idx = self.model, self.data, self.idx
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        active = np.isin(self.close_digit, digits)
        moving = active.copy()
        ctrl = q0.copy()
        step = CLOSE_SPEED * CTRL_EVERY * m.opt.timestep
        for _ in range(int(CLOSE_TIMEOUT_S / (CTRL_EVERY * m.opt.timestep))):
            bodies, _ = self._contacts()
            for b in bodies:
                for k in self.body_stops.get(b, ()):
                    moving[k] = False
            if not moving.any():
                break
            j = self.close_j[moving]
            ctrl[j] = np.clip(ctrl[j] + self.close_sign[moving] * step, lo[j], hi[j])
            at_limit = np.isclose(ctrl[self.close_j], self.close_limit)
            moving &= ~at_limit
            d.ctrl[idx.act] = ctrl
            mujoco.mj_step(m, d, nstep=CTRL_EVERY)
            if after_step is not None:
                after_step()

        # Squeeze the joints that stopped on contact (not the ones that ran to their limit).
        pressed = active & ~moving & ~np.isclose(ctrl[self.close_j], self.close_limit)
        j = self.close_j[pressed]
        ctrl[j] = np.clip(ctrl[j] + self.close_sign[pressed] * SQUEEZE_RAD, lo[j], hi[j])
        d.ctrl[idx.act] = ctrl
        return ctrl

    def grasp(self, q0, kind, size, pos, quat, digits) -> dict | None:
        """Pre-shape ``q0``, object at (pos, quat), close ``digits``, squeeze, shake. None if it fails."""
        m, d, idx = self.model, self.data, self.idx
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        mujoco.mj_resetData(m, d)
        set_object(m, idx, kind, size)
        m.opt.gravity[:] = 0.0
        d.qpos[idx.qpos] = q0
        d.ctrl[idx.act] = q0
        d.qpos[idx.obj_qpos:idx.obj_qpos + 3] = pos
        d.qpos[idx.obj_qpos + 3:idx.obj_qpos + 7] = quat
        mujoco.mj_forward(m, d)
        if self._contacts()[0] or any(c.dist < 0 for c in d.contact[: d.ncon]):
            return None  # starts inside the hand

        ctrl = self.close(q0, digits, after_step=lambda: self._pin_object(pos, quat))
        settle = int(SETTLE_S / m.opt.timestep)
        for _ in range(settle // (2 * CTRL_EVERY)):  # first half pinned: the squeeze builds up
            mujoco.mj_step(m, d, nstep=CTRL_EVERY)
            self._pin_object(pos, quat)
        mujoco.mj_step(m, d, nstep=settle // 2)  # then released

        if self.unstable():
            return None  # MuJoCo reset a blown-up step (a squeezed cylinder can spin up); not a grasp
        _, groups = self._contacts()
        if len(groups) < 2:
            return None
        q_grasp = d.qpos[idx.qpos].copy()
        if not self.hold_test(ctrl):
            return None
        obj_pos, obj_quat = self._object_pose()
        return dict(
            q=q_grasp, ctrl=ctrl.copy(), preshape=q0, kind=kind,
            size=np.pad(size, (0, 3 - len(size))), obj_pos=obj_pos, obj_quat=obj_quat,
            contacts=np.array([g in groups for g in CONTACT_GROUPS]),
        )

    def unstable(self) -> bool:
        """True if MuJoCo has flagged (and reset) a bad step since the last mj_resetData."""
        return any(self.data.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQACC,
                                                         mujoco.mjtWarning.mjWARN_BADQVEL,
                                                         mujoco.mjtWarning.mjWARN_BADQPOS))

    def hold_test(self, ctrl: np.ndarray) -> bool:
        """Gravity along each of +-X/Y/Z from the current state; True if the object stays."""
        m, d = self.model, self.data
        snap = mujoco.MjData(m)
        mujoco.mj_copyData(snap, m, d)
        try:
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    mujoco.mj_copyData(d, m, snap)
                    d.ctrl[self.idx.act] = ctrl
                    p0, q0 = self._object_pose()
                    m.opt.gravity[:] = 0.0
                    m.opt.gravity[axis] = sign * GRAVITY
                    mujoco.mj_step(m, d, nstep=int(HOLD_TEST_S / m.opt.timestep))
                    p1, q1 = self._object_pose()
                    turn = 2 * np.arccos(min(1.0, abs(float(q0 @ q1))))
                    if np.linalg.norm(p1 - p0) > MAX_DRIFT_M or turn > MAX_TURN_RAD or self.unstable():
                        return False
            return True
        finally:
            m.opt.gravity[:] = 0.0
            mujoco.mj_copyData(d, m, snap)


_GEN: GraspGen | None = None


def _init_worker(thumb: str, sampling: dict, finger_collisions: bool) -> None:
    global _GEN
    SAMPLING.update(sampling)  # spawn-safe: workers don't inherit the parent's edits
    _GEN = GraspGen(thumb, finger_collisions)


def _worker(seed: int):
    return _GEN.trial(seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trials", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=mp.cpu_count())
    p.add_argument("--out", default="grasps.npz")
    p.add_argument("--thumb", default="stock", help="thumb mount (thumb.THUMB_MOUNTS)")
    p.add_argument("--finger-collisions", action="store_true", help="digits collide with each other")
    p.add_argument("--obj-x", type=float, nargs=2, default=SAMPLING["obj_x"], help="object centre x range (hand frame)")
    p.add_argument("--obj-y", type=float, nargs=2, default=SAMPLING["obj_y"], help="object centre y range (hand frame)")
    args = p.parse_args()
    SAMPLING.update(obj_x=tuple(args.obj_x), obj_y=tuple(args.obj_y))

    seeds = range(args.seed * 10_000_000, args.seed * 10_000_000 + args.trials)
    t0 = time.time()
    results = []
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.thumb, SAMPLING, args.finger_collisions)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, seeds, chunksize=16)):
            if r is not None:
                results.append(r)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{args.trials} trials, {len(results)} grasps, {time.time() - t0:.0f} s", flush=True)
    if not results:
        raise SystemExit("no stable grasps")
    out = {k: np.array([r[k] for r in results]) for k in results[0]}
    np.savez(args.out, joint_names=np.array(JOINT_NAMES), contact_groups=np.array(CONTACT_GROUPS), **out)
    kinds = {k: int((out["grasp_type"] == k).sum()) for k in GRASP_TYPES}
    print(f"{len(results)}/{args.trials} stable grasps ({kinds}) -> {args.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
