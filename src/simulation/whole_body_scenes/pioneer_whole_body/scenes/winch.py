"""Cranking a winch (a continuous circular motion): the robot winds a load up a mast by turning a crank handle round and
round, then locks it with a lever. The crank turns the drum; the drum and the load
are coupled rigidly (an equality constraint: the load rises 4 cm a turn), the load's weight pulls back on the crank,
the crank's friction holds it where it is let go.

The robot walks up to the bench. On it:

  winch   on the right: a crank wheel lying flat, its handle a knob on a free-spinning post at the rim (the hand
          keeps its grip; the knob turns under it); where round the wheel's far half the handle
          starts changes with every seed
  mast    behind it: the load hangs from its top; a latch at the top catches the load when it comes up
  lever   on the left: the lock; pushed down, it locks the load in the latch: the lamp comes up

Steps (``progress``): 1. wind the load up (n cm), 2. lock it.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import READY
from pioneer_whole_body.parts import KNOB, KNOB_H, KNOB_R, STEEL, quat_z
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

CRANK = (0.50, -0.14)                    # the crank wheel's axis
CRANK_Z = 0.05                           # its height above the bench
CRANK_R = 0.065                          # the handle's radius
POST_H = 0.012
LIFT = 0.04 / (2 * math.pi)              # m of load per radian of crank
TOP = 0.16                               # the latch: the load this high
MAST = (0.66, -0.14)
LOAD_Z0 = 0.03                           # the load's bottom above the bench at the start
LEVER = (0.60, 0.16)
LEVER_TRAVEL = 0.5                       # rad
STEPS = ("wind the load up", "lock it")
_U_STEP, _U_LOCK = 0, 1


def _height(model, data) -> float:
    return float(data.qpos[model.joint("load").qposadr[0]])


def step(model, data) -> None:
    lever = data.qpos[model.joint("lever").qposadr[0]]
    if lever < -0.5 * LEVER_TRAVEL and _height(model, data) > TOP - 0.006:
        data.userdata[_U_LOCK] = 1
    a = model.joint("lamp").qposadr[0]
    target = 0.0 if data.userdata[_U_LOCK] else -math.pi / 2
    data.qpos[a] += float(np.clip(target - data.qpos[a], -0.004, 0.004))
    data.qvel[model.joint("lamp").dofadr[0]] = 0.0
    k = int(data.userdata[_U_STEP])
    if k == 0 and _height(model, data) > TOP - 0.006:
        data.userdata[_U_STEP] = 1
    elif k == 1 and data.userdata[_U_LOCK]:
        data.userdata[_U_STEP] = 2


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 0:
        return 0, len(STEPS), f"wind the load up ({100 * _height(model, data):.0f} of {100 * TOP:.0f} cm)"
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def reset(model, data, rng) -> None:
    """The handle anywhere round the far half of the wheel (the load at the bottom all the same; on the near half
    it'd be too close to the body to take)."""
    a = rng.uniform(-0.5 * math.pi, 0.5 * math.pi)
    data.qpos[model.joint("crank").qposadr[0]] = a
    model.eq_data[model.equality("drum").id, 0] = -LIFT * a


@scene("winch", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  cranking a winch: the load wound up the mast, locked")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 2
    world = spec.worldbody
    grey, dark = [0.32, 0.34, 0.37, 1], [0.16, 0.18, 0.22, 1]
    yellow = [0.93, 0.72, 0.12, 1]
    cx, cy = CRANK
    base = world.add_body(name="winch", pos=[cx, cy, T])
    base.add_geom(type=CYL, size=[0.03, CRANK_Z / 2 - 0.004, 0], pos=[0, 0, CRANK_Z / 2 - 0.004], rgba=dark)
    wheel = base.add_body(name="crank", pos=[0, 0, CRANK_Z])
    wheel.add_joint(name="crank", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.02, frictionloss=0.04,
                    armature=0.002)
    wheel.add_geom(type=CYL, size=[CRANK_R + 0.012, 0.004, 0], pos=[0, 0, 0.004], rgba=yellow, mass=0.05)
    wheel.add_geom(type=BOX, size=[CRANK_R / 2, 0.006, 0.0005], pos=[CRANK_R / 2, 0, 0.0085], rgba=dark,
                   contype=0, conaffinity=0, mass=0.001)
    handle = wheel.add_body(name="handle", pos=[CRANK_R, 0, 0.008])
    handle.add_joint(name="handle", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.001)
    handle.add_geom(type=CYL, size=[KNOB_R, POST_H / 2, 0], pos=[0, 0, POST_H / 2], rgba=STEEL, mass=0.004)
    z = POST_H
    handle.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[0, 0, z + KNOB_H / 2], rgba=KNOB, mass=0.015,
                    friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):
        a = 2 * math.pi * i / 10
        handle.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                        pos=[KNOB_R * math.cos(a), KNOB_R * math.sin(a), z + KNOB_H / 2], quat=quat_z(a), rgba=KNOB,
                        contype=0, conaffinity=0, mass=0.0001)
    handle.add_site(name="handle_top", pos=[0, 0, z + KNOB_H + 0.0012])
    spec.add_exclude(bodyname1="winch", bodyname2="crank")
    spec.add_exclude(bodyname1="crank", bodyname2="handle")
    # the mast and the load on it (a slide, coupled to the crank)
    mx, my = MAST
    mast = world.add_body(name="mast", pos=[mx, my, T])
    mast.add_geom(type=BOX, size=[0.03, 0.03, 0.004], pos=[0, 0, 0.004], rgba=dark)
    mast.add_geom(type=BOX, size=[0.008, 0.008, 0.15], pos=[0.03, 0, 0.15], rgba=grey)
    mast.add_geom(type=BOX, size=[0.045, 0.01, 0.006], pos=[0.0, 0, 0.3], rgba=grey)                 # the jib
    mast.add_geom(type=BOX, size=[0.012, 0.016, 0.008], pos=[-0.005, 0, LOAD_Z0 + TOP + 0.075], rgba=yellow,
                  contype=0, conaffinity=0)                                                           # the latch
    load = mast.add_body(name="load", pos=[-0.02, 0, LOAD_Z0])
    load.add_joint(name="load", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[0, TOP + 0.04],
                   limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.5)
    load.add_geom(type=BOX, size=[0.025, 0.025, 0.02], pos=[0, 0, 0.02], rgba=[0.8, 0.3, 0.12, 1], mass=0.25,
                  contype=0, conaffinity=0)
    load.add_geom(type=CYL, size=[0.0015, 0.13, 0], pos=[0, 0, 0.17], rgba=[0.15, 0.15, 0.15, 1], contype=0,
                  conaffinity=0, mass=0.001)                                                           # the rope
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name="drum", name1="load", name2="crank",
                      data=[0, LIFT, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # the lock lever on the left (pushed down with a fist) and the lamp
    lx, ly = LEVER
    lb = world.add_body(name="lever_base", pos=[lx, ly, T])
    lb.add_geom(type=BOX, size=[0.03, 0.03, 0.025], pos=[0, 0, 0.025], rgba=dark)
    lever = lb.add_body(name="lever", pos=[0, 0, 0.05])
    lever.add_joint(name="lever", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-math.degrees(LEVER_TRAVEL), 0],   # (degrees: MjSpec's default)
                    limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=0.6, springref=0.0, damping=0.05)
    lever.add_geom(type=BOX, size=[0.05, 0.006, 0.004], pos=[-0.05, 0, 0.004], rgba=grey, mass=0.02,
                   quat=[math.cos(-0.25), 0, math.sin(-0.25), 0])
    lever.add_geom(type=CYL, size=[0.014, 0.008, 0], pos=[-0.095, 0, 0.027], rgba=[0.85, 0.15, 0.12, 1],
                   mass=0.01)
    lever.add_site(name="lever_top", pos=[-0.095, 0, 0.035])
    spec.add_exclude(bodyname1="lever_base", bodyname2="lever")
    post = world.add_body(name="lamp_post", pos=[0.66, 0.16, T])
    post.add_geom(type=CYL, size=[0.006, 0.05, 0], pos=[0, 0, 0.05], rgba=grey, contype=0, conaffinity=0)
    flag = post.add_body(name="lamp", pos=[0, 0, 0.1])
    flag.add_joint(name="lamp", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], ref=0.0)
    flag.add_geom(type=BOX, size=[0.003, 0.02, 0.03], pos=[0, 0, 0.03], rgba=[0.15, 0.85, 0.3, 1], contype=0,
                  conaffinity=0, mass=0.01)
    flag.gravcomp = 1.0


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S

    m, d = wb.m, wb.d
    d.qpos[m.joint("lamp").qposadr[0]] = -math.pi / 2
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    wb.walk(0.30)
    # take the handle
    S.pick_knob(wb, "r", "handle_top", "handle", yaw=0.5, lift=0.0)
    wb.look([MAST[0], MAST[1], T + 0.15], dur=0.8)
    turns = 0.0
    a0 = _angle(wb)
    for _ in range(30):                               # (at most 30 quarter turns)
        if _height(m, d) >= TOP:
            break
        # a quarter turn at a time (anticlockwise: the load goes up), the last one just as far as the latch
        _crank(wb, "r", min(0.5 * math.pi, (TOP + 0.002 - _height(m, d)) / LIFT))
        turns = (_angle(wb) - a0) / (2 * math.pi)
    # the handle on the near half (by the body) is awkward to let go of: on round to the far half (the load has room
    # above the latch)
    knob = S.site(wb, "handle_top")
    a = math.atan2(knob[1] - CRANK[1], knob[0] - CRANK[0])
    if math.cos(a) < 0.2:
        _crank(wb, "r", (-0.5 * math.pi + 0.5 - a) % (2 * math.pi))
        turns = (_angle(wb) - a0) / (2 * math.pi)
    notes["turns"] = round(turns, 2)
    notes["height"] = round(_height(m, d), 3)
    S.release(wb, "r", up=0.06)
    wb.arm_joints_to("r", READY, 1.0)
    wb.settle()
    if _height(m, d) < TOP - 0.006:
        raise S.TaskFailed(f"the load is at {_height(m, d):.3f} m")
    # lock it: the left fist on the lever
    top = S.site(wb, "lever_top")
    wb.look(top, dur=0.5)
    wb.fingers("l", "fist", dur=0.4)
    S.press(wb, "l", "l_thumb_tip", top, depth=0.06, hold=0.4)
    wb.fingers("l", "relaxed", dur=0.4)
    wb.arm_joints_to("l", READY, 1.0)
    wb.look([0.66, 0.16, T + 0.12], dur=0.6)
    wb.wait(2.0)
    if not d.userdata[_U_LOCK]:
        raise S.TaskFailed("not locked")


def _angle(wb) -> float:
    return float(wb.d.qpos[wb.m.joint("crank").qposadr[0]])


def _crank(wb, side: str, sweep: float, step: float = math.radians(12), dt: float = 0.12) -> None:
    """Carry the handle round the crank's axis by ``sweep`` (rad), the palm keeping its orientation (the knob turns
    under it): small straight moves along the circle, starting from where the handle is now."""
    from pioneer_whole_body import skills as S
    p, r = wb.hand_goal[side]
    c = np.array([CRANK[0], CRANK[1]])
    knob = S.site(wb, "handle_top")
    off = p - knob                                  # the palm from the knob's top
    a = math.atan2(knob[1] - c[1], knob[0] - c[0])
    n = max(1, math.ceil(abs(sweep) / step))
    for k in range(1, n + 1):
        b = a + sweep * k / n
        q = np.array([c[0] + CRANK_R * math.cos(b), c[1] + CRANK_R * math.sin(b), knob[2]]) + off
        wb.reach(side, q, r, dt)
        wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["winch"].plan = plan
