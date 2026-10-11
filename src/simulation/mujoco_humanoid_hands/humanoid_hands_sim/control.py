"""Scripted control of the humanoid: per-arm IK to Cartesian targets, finger closure, grasp assist.

Sim.move interpolates a grasp-site target (position + rotation); every 10 ms tick both arms re-solve IK
(warm-started) and send joint targets to the PD actuators, then the physics steps 10 x 1 ms.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from . import matcha_hands as S
from .robot import HANDSHAKE, Arm, make_model

FINGERS = ("", "i_", "r_", "l_")   # middle, index, ring, little (URDF joint-name prefixes)
READY = {"l": np.array([0.12, 0.28, 0.95]), "r": np.array([0.12, -0.28, 0.95])}


def hand_ctrl(side, c, thumb=None):
    """Finger targets for closure c in [0,1] (power grasp); thumb closure defaults to c."""
    t = c if thumb is None else thumb
    out = {}
    for f in FINGERS:
        out[f"{side}_{f}abd"] = 0.0
        out[f"{side}_{f}mcp"] = 1.4 * c
        out[f"{side}_{f}pip"] = 1.5 * c
        out[f"{side}_{f}dip"] = 1.2 * c
    out.update({f"{side}_th_c": 0.87 * t, f"{side}_th_f": 0.9 * t, f"{side}_th_mcp": 0.9 * t, f"{side}_th_ip": 0.9 * t})
    return out


class Sim:
    """The model, its state, both arms' IK targets, and a qpos record at `fps`."""

    def __init__(self, model: mujoco.MjModel | None = None, fps: int = 30):
        self.m = m = model if model is not None else make_model()
        self.d = d = mujoco.MjData(m)
        self.arms = {"l": Arm(m, "l"), "r": Arm(m, "r")}
        self.q, self.tgt = {}, {}
        for s, a in self.arms.items():
            a.rest = np.array([-0.3, 0, 0, -1.6, 0, 0, 0])
            q, _, _ = a.ik(READY[s], HANDSHAKE, a.rest, iters=200)   # start with both hands up, in front
            self.q[s] = q
            d.qpos[a.qadr] = q
            d.ctrl[a.act] = q
            self.tgt[s] = a.fk(q)
        mujoco.mj_forward(m, d)
        self.held = {}                     # side -> tool welded to that palm
        self.fps, self.traj = fps, []

    # -- state
    def site(self, side):
        sid = self.arms[side].site
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def body(self, name):
        b = self.m.body(name).id
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    # -- time
    def tick(self, n=10):
        for s, a in self.arms.items():
            p, R = self.tgt[s]
            q, _, _ = a.ik(p, R, self.q[s], iters=8)
            self.q[s] = q
            self.d.ctrl[a.act] = q
        for _ in range(n):
            S.step(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
        if self.d.time * self.fps >= len(self.traj):
            self.traj.append(self.d.qpos.copy())

    def wait(self, t):
        for _ in range(int(t / 0.01)):
            self.tick()

    def move(self, side, pos, R=None, t=1.0):
        """Smooth (smoothstep) move of the grasp site to pos, slerping to rotation R (default: keep)."""
        p0, R0 = self.tgt[side]
        R = R0 if R is None else R
        sl = Slerp([0, 1], Rot.from_matrix([R0, R]))
        n = max(1, int(t / 0.01))
        for i in range(1, n + 1):
            u = i / n
            u = u * u * (3 - 2 * u)
            self.tgt[side] = (p0 + (np.asarray(pos) - p0) * u, sl(u).as_matrix())
            self.tick()

    # -- hands
    def set_hand(self, side, **kw):
        for k, v in hand_ctrl(side, **kw).items():
            self.d.ctrl[self.m.actuator(k).id] = v

    def ramp_hand(self, side, c, thumb, t=0.6, until=None):
        """Close/open the hand gradually (a step command makes the fingertips strike what they grasp).
        until=tool: stop closing as soon as `tool` is gripped."""
        ids = {k: self.m.actuator(k).id for k in hand_ctrl(side, 0.0)}
        start = {k: self.d.ctrl[i] for k, i in ids.items()}
        goal = hand_ctrl(side, c, thumb)
        n = max(1, int(t / 0.01))
        for j in range(1, n + 1):
            u = j / n
            for k, i in ids.items():
                self.d.ctrl[i] = start[k] + (goal[k] - start[k]) * u
            self.tick()
            if until and self.gripped(side, until):
                return

    def hand_contacts(self, side, tool):
        """Hand links of `side` touching body `tool`."""
        m, d = self.m, self.d
        T, out = m.body(tool).id, set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if T in (b1, b2):
                o = m.body(b2 if b1 == T else b1).name
                if o.startswith(f"{side}_hand"):
                    out.add(o)
        return sorted(out)

    def gripped(self, side, tool):
        """At least one finger and the thumb or the palm touch `tool` (its handle is between them)."""
        links = self.hand_contacts(side, tool)
        return (any(l.split("_hand_")[1][0] in "PID" for l in links)
                and any(l.split("_hand_")[1][0] in "TH" for l in links))

    def attach(self, side, tool):
        """Grasp assist: weld `tool` to the palm where it is now -- only if the fingers physically grip it."""
        links = self.hand_contacts(side, tool)
        print(f"  [{self.d.time:5.1f}s] {side} hand on {tool}: {len(links)} links in contact {links}", flush=True)
        if not self.gripped(side, tool):
            raise RuntimeError(f"grasp of {tool} failed: only {links}")
        m, d = self.m, self.d
        e = m.equality(f"{side}_grasp_{tool}").id
        b1, b2 = m.eq_obj1id[e], m.eq_obj2id[e]
        R1 = d.xmat[b1].reshape(3, 3)
        q1, q2, q1i, rel = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(q1, d.xmat[b1])
        mujoco.mju_mat2Quat(q2, d.xmat[b2])
        mujoco.mju_negQuat(q1i, q1)
        mujoco.mju_mulQuat(rel, q1i, q2)
        m.eq_data[e, :3] = 0                               # anchor
        m.eq_data[e, 3:6] = R1.T @ (d.xpos[b2] - d.xpos[b1])  # tool position in the palm frame
        m.eq_data[e, 6:10] = rel                           # tool orientation in the palm frame
        m.eq_data[e, 10] = 1.0                             # torque scale
        d.eq_active[e] = 1
        self.held[side] = tool

    def detach(self, side):
        tool = self.held.pop(side, None)
        if tool:
            self.d.eq_active[self.m.equality(f"{side}_grasp_{tool}").id] = 0
