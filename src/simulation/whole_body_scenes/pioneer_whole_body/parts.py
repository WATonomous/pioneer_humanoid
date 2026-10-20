"""Shared assembly parts: knob bolts, threaded holes, and the thread engagement the physics can't express by itself.

A real thread is a helix: MuJoCo can't collide one. A threaded hole here is a small body inside its part with a
hinge (turn) and a slide (feed) along the hole's axis, coupled rigidly (``feed = PITCH / 2 pi * turn``: turn it
clockwise seen from above and it goes in; ``Thread.step`` sets the feed from the turn every physics step, as stiff as
a real thread: pushing or pulling on the bolt doesn't move it along). It has thread friction and stops at the
bottom: once there the turn can't go further.

A bolt is a free body. When its tip is in the hole's mouth, upright and centred (``Thread.try_engage``), a weld
switches on between the bolt and the hole body: from then on the bolt can only turn, and feeds in or out as it
turns, like on a thread. Nothing turns it but the hand's grip on the knob (friction at the fingertips).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

STEEL = [0.66, 0.68, 0.72, 1]
DARK = [0.10, 0.11, 0.13, 1]
KNOB = [0.13, 0.14, 0.16, 1]
ORANGE = [0.93, 0.47, 0.16, 1]

PITCH = 0.012                 # m per turn: a quick-acting (multi-start) thread: a bolt seats in under a turn
THREAD_FRICTION = 0.015       # N m: holds a released bolt, far below what a grip can turn
ENGAGE_RADIUS = 0.003         # tip within this of the hole axis
ENGAGE_TILT = math.radians(10)
ENGAGE_DEPTH = 0.004          # tip within this above / below the hole's mouth

KNOB_R, KNOB_H = 0.021, 0.042
SHAFT_R = 0.0045


def quat_z(a: float) -> list[float]:
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def add_knob_bolt(world, name: str, pos, yaw: float = 0.0, shaft: float = 0.05, color=KNOB):
    """A star-knob style bolt standing on its tip: origin at the tip, knob on top (a knurled ring for the fingers)."""
    b = world.add_body(name=name, pos=list(pos), quat=quat_z(yaw))
    b.add_freejoint(name=name)
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    if shaft > 0:                     # (a nut has none: its underside goes onto the thread)
        b.add_geom(name=f"{name}_shaft", type=cyl, size=[SHAFT_R, shaft / 2, 0], pos=[0, 0, shaft / 2], rgba=STEEL,
                   mass=0.02)
        b.add_geom(name=f"{name}_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[SHAFT_R, 0, 0], pos=[0, 0, 0.0005],
                   rgba=STEEL, mass=0.001)
    b.add_geom(name=f"{name}_washer", type=cyl, size=[0.011, 0.0015, 0], pos=[0, 0, shaft + 0.0015], rgba=STEEL,
               mass=0.003)
    b.add_geom(name=f"{name}_knob", type=cyl, size=[KNOB_R, KNOB_H / 2, 0], pos=[0, 0, shaft + 0.003 + KNOB_H / 2],
               rgba=color, mass=0.02, friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):     # knurl ridges (visual)
        a = 2 * math.pi * i / 10
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                   pos=[KNOB_R * math.cos(a), KNOB_R * math.sin(a), shaft + 0.003 + KNOB_H / 2],
                   quat=quat_z(a), rgba=color, contype=0, conaffinity=0, mass=0.0001)
    b.add_geom(type=cyl, size=[KNOB_R * 0.45, 0.0006, 0], pos=[0, 0, shaft + 0.003 + KNOB_H + 0.0006],
               rgba=ORANGE, contype=0, conaffinity=0, mass=0.0001)
    b.add_site(name=f"{name}_tip", pos=[0, 0, 0])
    b.add_site(name=f"{name}_top", pos=[0, 0, shaft + 0.003 + KNOB_H])
    return b


def bolt_length(shaft: float = 0.05) -> float:
    return shaft + 0.003 + KNOB_H


def add_thread_hole(spec: mujoco.MjSpec, part, name: str, pos, travel: float):
    """A threaded hole in ``part`` with its mouth at ``pos`` (part frame): the bolt goes ``travel`` deep."""
    h = part.add_body(name=name, pos=list(pos))
    h.mass = 0.002
    h.gravcomp = 1.0
    h.inertia = [1e-7, 1e-7, 1e-7]
    # heavy armature: MuJoCo scales a constraint's stiffness with the inertia it acts on, and a 2 g nut body lets
    # an arm pull the bolt off its thread limit by centimetres
    h.add_joint(name=f"{name}_feed", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], armature=2.0, damping=5.0)
    h.add_joint(name=f"{name}_turn", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], frictionloss=THREAD_FRICTION,
                armature=1e-4, damping=0.002)
    h.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[SHAFT_R + 0.0008, 0.0002, 0], pos=[0, 0, -0.0001],
               rgba=DARK, contype=0, conaffinity=0, group=1, mass=0)
    return h


def add_weld(spec: mujoco.MjSpec, name: str, body1: str, body2: str, active: bool = True):
    """A weld holding body2 where it is relative to body1 in the model's reference pose (anchor at body2's origin).
    (MjSpec's default equality data is the joint one, "0 1 0 0 0", which as a weld anchor sits a metre off.)"""
    eq = spec.add_equality(name=name, type=mujoco.mjtEq.mjEQ_WELD, name1=body1, name2=body2,
                           objtype=mujoco.mjtObj.mjOBJ_BODY, active=active)
    eq.data[:] = [0.0] * len(eq.data)
    eq.data[10] = 1.0                      # torquescale; relpose quaternion 0 = use the reference pose
    eq.solref = [0.004, 1.0]
    eq.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    return eq


def add_bolt_weld(spec: mujoco.MjSpec, hole: str, bolt: str):
    """The (initially off) weld that holds ``bolt`` on the thread of ``hole`` once engaged."""
    eq = spec.add_equality(name=f"{hole}__{bolt}", type=mujoco.mjtEq.mjEQ_WELD, name1=hole, name2=bolt,
                           objtype=mujoco.mjtObj.mjOBJ_BODY, active=False)
    eq.solref = [0.004, 1.0]
    eq.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    return eq


@dataclass
class Thread:
    """Run-time state of one hole: which bolt (if any) is on it. ``step`` engages a bolt that's in position."""
    hole: str
    bolts: tuple[str, ...]
    travel: float
    radius: float = ENGAGE_RADIUS     # how far off the axis a tip may be and still catch (a nut over an axle: wide)

    def bind(self, model: mujoco.MjModel):
        self.hole_id = model.body(self.hole).id
        self.feed = model.joint(f"{self.hole}_feed").qposadr[0]
        self.turn = model.joint(f"{self.hole}_turn").qposadr[0]
        self.feed_v = model.joint(f"{self.hole}_feed").dofadr[0]
        self.turn_v = model.joint(f"{self.hole}_turn").dofadr[0]
        self.eqs = {b: model.equality(f"{self.hole}__{b}").id for b in self.bolts}
        self.tips = {b: model.site(f"{b}_tip").id for b in self.bolts}
        self.bolt_ids = {b: model.body(b).id for b in self.bolts}
        return self

    def engaged(self, data) -> str | None:
        for b, e in self.eqs.items():
            if data.eq_active[e]:
                return b
        return None

    def depth(self, data) -> float:
        """How far the engaged bolt has gone in (0 .. travel)."""
        return -float(data.qpos[self.feed])

    def seated(self, data, tol: float = 0.0015) -> bool:
        return self.engaged(data) is not None and self.depth(data) > self.travel - tol

    def step(self, data) -> None:
        """The helix: the feed follows the turn (and the turn stops where the bolt bottoms out or backs out).
        A bolt screwed home is tight (its preload's friction): from then on it only turns further in."""
        k = PITCH / (2 * math.pi)
        if self.engaged(data) is None:
            data.qpos[self.feed] = 0.0
            data.qvel[self.feed_v] = 0.0
            return
        lo, hi = -self.travel / k, 0.0005 / k
        if self.seated(data):                  # latch the tightest turn reached: it never comes back out
            self.home = max(lo, min(getattr(self, "home", 0.0), float(data.qpos[self.turn])))
            hi = self.home
        if data.qpos[self.turn] < lo or data.qpos[self.turn] > hi:
            data.qpos[self.turn] = min(max(data.qpos[self.turn], lo), hi)
            data.qvel[self.turn_v] = 0.0
        data.qpos[self.feed] = k * data.qpos[self.turn]
        data.qvel[self.feed_v] = k * data.qvel[self.turn_v]

    def try_engage(self, model, data, busy: set) -> str | None:
        if self.engaged(data) is not None:
            return None
        hp = data.xpos[self.hole_id]
        hr = data.xmat[self.hole_id].reshape(3, 3)
        for b in self.bolts:
            if b in busy:
                continue
            br = data.xmat[self.bolt_ids[b]].reshape(3, 3)
            rel = hr.T @ (data.site_xpos[self.tips[b]] - hp)
            tilt = math.acos(min(1.0, float(hr[:, 2] @ br[:, 2])))
            if math.hypot(rel[0], rel[1]) < self.radius and abs(rel[2]) < ENGAGE_DEPTH and tilt < ENGAGE_TILT:
                self.home = 0.0
                # the thread body is where its joints put it (feed 0 at the mouth); the bolt goes on it upright at
                # its current turn angle, so the weld only straightens it
                data.qpos[self.feed] = 0.0
                data.qpos[self.turn] = 0.0
                mujoco.mj_kinematics(model, data)
                hr = data.xmat[self.hole_id].reshape(3, 3)
                rel_r = hr.T @ br
                yaw = math.atan2(rel_r[1, 0], rel_r[0, 0])
                e = self.eqs[b]
                model.eq_data[e, 0:3] = 0
                model.eq_data[e, 3:6] = [0, 0, 0]
                model.eq_data[e, 6:10] = quat_z(yaw)
                model.eq_data[e, 10] = 1.0
                data.eq_active[e] = 1
                return b
        return None
