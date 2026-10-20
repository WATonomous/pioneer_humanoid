"""Socket wrench (long horizon, tool use): drive two hex-head bolts home with a T-handle socket wrench, put the
wrench back, test.

The robot walks up to the bench. On it:

  plate      in front: two hex-head bolts started in their threads (a hand can't turn a hex head: a tool can)
  wrench     on the right, standing in its holder: a socket at the bottom, a shaft, a knob handle on top
  button     on the left: its lamp comes up when both bolts are home and the wrench is back in its holder

Steps, each checked from the simulation and latched in order (``progress``):
  1. pick up the wrench      right hand, by its knob
  2. drive bolt 1 home       the socket set over the bolt's head (it takes it), the wrench turned by its knob:
                             grip, turn, let go, turn back, regrasp
  3. drive bolt 2 home       the wrench lifted off, over the next head, the same
  4. wrench back             in its holder
  5. test                    left thumb on the button
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY
from pioneer_whole_body.parts import (KNOB, KNOB_H, KNOB_R, ORANGE, STEEL, Thread, add_bolt_weld, add_thread_hole,
                                      add_weld, quat_z)
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

PLATE = (0.52, -0.12)
PLATE_H = 0.02
BOLT_Y = {1: -0.06, 2: -0.18}            # (plate frame: the plate's centre is at PLATE)
TRAVEL = 0.009
SHAFT = 0.02                             # the bolt's shaft (its tip starts at the thread's mouth)
HEAD_R, HEAD_H = 0.011, 0.009
SOCKET_R, SOCKET_H = 0.016, 0.022        # the wrench's socket (fits over a head)
WRENCH_SHAFT = 0.07
HOLDER = (0.48, -0.40)
HOLDER_R = 0.026                          # 6 mm of play round the socket
BUTTON_POS = (0.58, 0.18)               # on the left (the wrench stands between the right hand and anything right)
BUTTON_TRAVEL = 0.008
TAKE_R = 0.004                           # the socket takes a head within this of its axis

STEPS = ("pick up the wrench", "drive bolt 1 home", "drive bolt 2 home", "wrench back", "test")
THREADS = (Thread("thread1", ("bolt1",), TRAVEL), Thread("thread2", ("bolt2",), TRAVEL))
_U_STEP, _U_ON, _U_LIFT = 0, 1, 2
_BOUND: dict = {}


def threads(model):
    key = id(model)
    if key not in _BOUND:
        _BOUND[key] = [Thread(t.hole, t.bolts, t.travel, t.radius).bind(model) for t in THREADS]
    return _BOUND[key]


def on_head(model, data):
    """The bolt the wrench is on (its socket welded to the head), or None."""
    for k in (1, 2):
        if data.eq_active[model.equality(f"socket{k}").id]:
            return k
    return None


def _gripped(model, data) -> bool:
    """A hand closed on the wrench: three or more of its finger links touching it."""
    w = model.body("wrench").id
    n = 0
    for c in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if w in (b1, b2) and "_hand_" in model.body(b2 if b1 == w else b1).name:
            n += 1
    return n >= 3


def wrench_home(model, data) -> bool:
    w = data.site_xpos[model.site("wrench_tip").id]
    up = data.xmat[model.body("wrench").id][8] > math.cos(math.radians(10))
    return math.hypot(w[0] - HOLDER[0], w[1] - HOLDER[1]) < 0.012 and w[2] < T + 0.008 and up


def step(model, data) -> None:
    ths = threads(model)
    for t in ths:
        t.step(data)
    busy = {t.engaged(data) for t in ths} - {None}
    for t in ths:
        t.try_engage(model, data, busy)
    # the socket takes a head: centred over it, upright, slid down over it
    if on_head(model, data) is None:
        tip = data.site_xpos[model.site("wrench_tip").id]
        up = data.xmat[model.body("wrench").id][8] > math.cos(math.radians(8))
        for k, t in zip((1, 2), ths):
            if t.seated(data):
                continue
            h = data.site_xpos[model.site(f"bolt{k}_headbase").id]
            if up and math.hypot(tip[0] - h[0], tip[1] - h[1]) < TAKE_R and abs(tip[2] - h[2]) < 0.004:
                e = model.equality(f"socket{k}").id
                b = model.body(f"bolt{k}").id
                q, qn = np.zeros(4), np.zeros(4)
                mujoco.mju_negQuat(qn, data.xquat[b])
                mujoco.mju_mulQuat(q, qn, data.xquat[model.body("wrench").id])
                model.eq_data[e, 0:3] = 0
                model.eq_data[e, 3:6] = data.xmat[b].reshape(3, 3).T @ (data.xpos[model.body("wrench").id] -
                                                                       data.xpos[b])
                model.eq_data[e, 6:10] = q
                model.eq_data[e, 10] = 1.0
                data.eq_active[e] = 1
                break
    else:                                   # the bolt home and the wrench in a hand again: it lifts off the head
        k = on_head(model, data)
        if ths[k - 1].seated(data) and data.userdata[_U_LIFT] and _gripped(model, data):
            data.eq_active[model.equality(f"socket{k}").id] = 0
            data.userdata[_U_LIFT] = 0
    if data.qpos[model.joint("button").qposadr[0]] < -0.6 * BUTTON_TRAVEL and data.userdata[_U_STEP] >= 4:
        data.userdata[_U_ON] = 1
    a = model.joint("lamp").qposadr[0]
    target = 0.0 if data.userdata[_U_ON] else -math.pi / 2
    data.qpos[a] += float(np.clip(target - data.qpos[a], -0.004, 0.004))
    data.qvel[model.joint("lamp").dofadr[0]] = 0.0
    k = int(data.userdata[_U_STEP])
    if k < len(STEPS) and _step_done(model, data, k):
        data.userdata[_U_STEP] = k + 1


def _step_done(model, data, k) -> bool:
    ths = threads(model)
    if k == 0:
        return data.site_xpos[model.site("wrench_tip").id][2] > T + 0.04
    if k == 1:
        return ths[0].seated(data)
    if k == 2:
        return ths[1].seated(data)
    if k == 3:
        return wrench_home(model, data)
    return bool(data.userdata[_U_ON])


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _hex_bolt(world, name, pos):
    """A hex-head bolt standing on its tip: origin at the tip."""
    b = world.add_body(name=name, pos=list(pos))
    b.add_freejoint(name=name)
    b.add_geom(type=CYL, size=[0.0045, SHAFT / 2, 0], pos=[0, 0, SHAFT / 2], rgba=STEEL, mass=0.01)
    b.add_geom(type=CYL, size=[HEAD_R * 1.25, 0.001, 0], pos=[0, 0, SHAFT + 0.001], rgba=STEEL, mass=0.002)  # washer
    z = SHAFT + 0.002
    for i in range(3):                      # the hex: three crossed boxes
        b.add_geom(type=BOX, size=[HEAD_R, HEAD_R * math.tan(math.pi / 6), HEAD_H / 2], pos=[0, 0, z + HEAD_H / 2],
                   quat=quat_z(i * math.pi / 3), rgba=[0.55, 0.57, 0.6, 1], mass=0.003)
    b.add_site(name=f"{name}_tip", pos=[0, 0, 0])
    b.add_site(name=f"{name}_headbase", pos=[0, 0, z])
    b.add_site(name=f"{name}_top", pos=[0, 0, z + HEAD_H])
    return b


@scene("wrench", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress,
       title="Pioneer v2  |  socket wrench: drive two hex bolts home with a tool, put it back, test")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec, half_y=0.7)
    spec.nuserdata = 3
    world = spec.worldbody
    grey, dark = [0.32, 0.34, 0.37, 1], [0.16, 0.22, 0.36, 1]

    # the plate with its two threads, the bolts started in them
    px, py = PLATE
    plate = world.add_body(name="plate", pos=[px, py, T])
    plate.add_geom(type=BOX, size=[0.05, 0.12, PLATE_H / 2], pos=[0, 0, PLATE_H / 2], rgba=dark)
    plate.add_geom(type=BOX, size=[0.03, 0.09, 0.0006], pos=[-0.025, 0, PLATE_H + 0.0006], rgba=ORANGE, contype=0,
                   conaffinity=0)
    for k, y in BOLT_Y.items():
        add_thread_hole(spec, plate, f"thread{k}", [0, y - py, PLATE_H], TRAVEL)
        _hex_bolt(world, f"bolt{k}", [px, y, T + PLATE_H])
        spec.add_exclude(bodyname1="plate", bodyname2=f"bolt{k}")
    for t in THREADS:
        for b in t.bolts:
            add_bolt_weld(spec, t.hole, b)

    # the wrench: socket, shaft, knob handle
    hx, hy = HOLDER
    w = world.add_body(name="wrench", pos=[hx, hy, T + 0.0005])
    w.add_freejoint(name="wrench")
    n = 10
    half_t = (SOCKET_R + 0.003) * math.tan(math.pi / n) + 0.0005
    for i in range(n):                      # the socket: a ring (open at the bottom, over the head)
        a = 2 * math.pi * i / n
        w.add_geom(type=BOX, size=[0.0015, half_t, SOCKET_H / 2],
                   pos=[(SOCKET_R + 0.0015) * math.cos(a), (SOCKET_R + 0.0015) * math.sin(a), SOCKET_H / 2],
                   quat=quat_z(a), rgba=[0.2, 0.21, 0.24, 1], mass=0.004)
    w.add_geom(type=CYL, size=[SOCKET_R + 0.003, 0.003, 0], pos=[0, 0, SOCKET_H + 0.003], rgba=[0.2, 0.21, 0.24, 1],
               mass=0.01)
    z = SOCKET_H + 0.006
    w.add_geom(type=CYL, size=[0.006, WRENCH_SHAFT / 2, 0], pos=[0, 0, z + WRENCH_SHAFT / 2], rgba=STEEL, mass=0.03)
    z += WRENCH_SHAFT
    w.add_geom(type=CYL, size=[KNOB_R, 0.0015, 0], pos=[0, 0, z + 0.0015], rgba=STEEL, mass=0.003)
    w.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[0, 0, z + 0.003 + KNOB_H / 2], rgba=[0.75, 0.12, 0.1, 1],
               mass=0.02, friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):
        a = 2 * math.pi * i / 10
        w.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                   pos=[KNOB_R * math.cos(a), KNOB_R * math.sin(a), z + 0.003 + KNOB_H / 2], quat=quat_z(a),
                   rgba=[0.75, 0.12, 0.1, 1], contype=0, conaffinity=0, mass=0.0001)
    w.add_geom(type=CYL, size=[KNOB_R * 0.45, 0.0006, 0], pos=[0, 0, z + 0.003 + KNOB_H + 0.0006], rgba=ORANGE,
               contype=0, conaffinity=0, mass=0.0001)
    w.add_site(name="wrench_tip", pos=[0, 0, 0])
    w.add_site(name="wrench_top", pos=[0, 0, z + 0.003 + KNOB_H])
    for k in (1, 2):
        spec.add_exclude(bodyname1="wrench", bodyname2=f"bolt{k}")
        add_weld(spec, f"socket{k}", f"bolt{k}", "wrench", active=False)
    # its holder: a ring the socket stands in
    holder = world.add_body(name="holder", pos=[hx, hy, T])
    for i in range(12):
        a = 2 * math.pi * i / 12
        holder.add_geom(type=BOX, size=[0.002, (HOLDER_R + 0.004) * math.tan(math.pi / 12) + 0.0005, 0.006],
                        pos=[(HOLDER_R + 0.002) * math.cos(a), (HOLDER_R + 0.002) * math.sin(a), 0.006],
                        quat=quat_z(a), rgba=grey)

    # test button and lamp
    bx, by = BUTTON_POS
    housing = world.add_body(name="button_housing", pos=[bx, by, T])
    housing.add_geom(type=BOX, size=[0.03, 0.03, 0.02], pos=[0, 0, 0.02], rgba=grey)
    housing.add_geom(type=CYL, size=[0.024, 0.003, 0], pos=[0, 0, 0.043], rgba=[0.95, 0.8, 0.1, 1])
    cap = housing.add_body(name="button", pos=[0, 0, 0.046])
    cap.add_joint(name="button", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-BUTTON_TRAVEL, 0],
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=150.0, springref=0.0, damping=3.0)
    cap.add_geom(type=CYL, size=[0.02, 0.006, 0], pos=[0, 0, 0.006], rgba=[0.1, 0.65, 0.2, 1], mass=0.02)
    cap.add_site(name="button_top", pos=[0, 0, 0.012])
    spec.add_exclude(bodyname1="button_housing", bodyname2="button")
    post = world.add_body(name="lamp_post", pos=[px + 0.08, py + 0.14, T])
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
    ths = threads(m)
    d.qpos[m.joint("lamp").qposadr[0]] = -math.pi / 2
    wb.look([PLATE[0], PLATE[1], T], dur=0.8)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _hands(wb, HIGH, 0.8)
    wb.walk(0.30)

    # 1. the wrench out of its holder
    S.pick_knob(wb, "r", "wrench_top", "wrench", yaw=0.6, lift=0.08)

    # 2. / 3. each bolt: the socket over its head, down until it takes it, then turned home by the knob
    notes["strokes"] = []
    for k, t in zip((1, 2), ths):
        head = S.site(wb, f"bolt{k}_headbase")
        S.carry(wb, "r", "wrench_tip", head, clear=HEAD_H + 0.02, dur=1.4, over=0.04)
        for attempt in range(3):
            S.align_held(wb, "r", "wrench", "wrench_tip", head, yaw=None, rounds=3, dur=0.4)
            hover = wb.hand_goal["r"]
            if S.lower_until(wb, "r", HEAD_H + 0.035, lambda k=k: on_head(m, d) == k, dur=1.2):
                break
            wb.reach("r", *hover, 0.5)
            wb.settle()
        else:
            raise S.TaskFailed(f"the socket did not take bolt {k}")
        wb.wait(0.15)
        notes["strokes"].append(S.tighten(wb, "r", t, "wrench", yaw0=1.1, yaw1=0.25, max_strokes=16))
        d.userdata[_U_LIFT] = 1                     # (the socket lets go of the head once the hand has the knob)
        S.pick_knob(wb, "r", "wrench_top", "wrench", yaw=0.6, lift=0.06)      # lift it off the head

    # 4. back in its holder
    home = np.array([HOLDER[0], HOLDER[1], T])
    S.carry(wb, "r", "wrench_tip", home, clear=0.03, dur=1.4, over=0.04)
    S.align_held(wb, "r", "wrench", "wrench_tip", home, yaw=None, rounds=2)
    S.lower_until(wb, "r", 0.045, lambda: wrench_home(m, d), dur=1.0)
    S.release(wb, "r", up=0.08)
    wb.arm_joints_to("r", READY, 1.0)
    wb.settle()

    # 5. test: left thumb on the button
    wb.arm_joints_to("l", READY, 1.0)
    wb.settle()
    top = S.site(wb, "button_top")
    wb.look(top, dur=0.5)
    wb.fingers("l", "fist", dur=0.4)
    S.press(wb, "l", "l_thumb_tip", top)
    wb.fingers("l", "relaxed", dur=0.4)
    wb.arm_joints_to("l", READY, 1.0)
    wb.look([PLATE[0] + 0.08, PLATE[1] + 0.14, T + 0.12], dur=0.6)
    wb.wait(2.0)
    notes["final"] = dict(bolts=[round(t.depth(d), 4) for t in ths], home=bool(wrench_home(m, d)),
                          on=bool(d.userdata[_U_ON]))


def _hands(wb, pose, dur: float = 1.0) -> None:
    for side in ("l", "r"):
        wb.arm_joints_to(side, pose, dur)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["wrench"].plan = plan
