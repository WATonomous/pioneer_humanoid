"""Hand-over (long horizon, both hands): a bar that one hand can't take all the way is passed from the right hand to
the left, set in a fixture on the left and bolted down.

The robot walks up to the bench. On it:

  bar        on two supports on the right: an aluminium bar with a knob handle at each end and a bolt hole between
  fixture    on the left: a cradle the bar drops into (it latches), with a thread under the bar's hole
  bolt rack  by the fixture: the bar's knob bolt
  button     on the right: "done"; its lamp comes up if the bar is bolted down in the fixture

Steps, each checked from the simulation and latched in order (``progress``):
  1. pick up the bar        right hand, by its inner knob
  2. hand it over           to the middle; the left hand takes the other knob, the right lets go
  3. into the fixture       left hand: set down in the cradle (it latches)
  4. bolt it down           left hand: the bolt through the bar into the thread, screwed home
  5. done                   right thumb on the button
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY
from pioneer_whole_body.parts import (KNOB, KNOB_H, KNOB_R, ORANGE, STEEL, Thread, add_bolt_weld, add_knob_bolt,
                                      add_thread_hole, add_weld, quat_z)
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

BAR = (0.016, 0.13, 0.01)               # half sizes (x, y, z): 26 cm long
KNOB_Y = 0.10                           # the knob handles, either side of the middle
KNOB_POST = 0.012
BAR_START = (0.50, -0.40)               # on its supports, along y
SUPPORT_H = 0.03
FIX = (0.50, 0.40)                      # the fixture (cradle) on the left: the bar's middle goes here
FIX_H = 0.05                            # top of the cradle's bed (the thread is under it)
CRADLE_PLAY = 0.008
HOLE = 0.0095                           # half width of the bar's bolt hole (5 mm of play round the shaft)
TRAVEL = 0.009
THREAD_DEEP = 0.045                     # the thread starts this far down in the cradle (fingers clear of the bar)
SHAFT = 2 * BAR[2] + THREAD_DEEP + TRAVEL - 0.0005
RACK = (0.47, 0.62)                     # out of the arm's way: a side step left to fetch the bolt
STEP = 0.15
BUTTON_POS = (0.60, -0.30)
BUTTON_TRAVEL = 0.008
HANDOVER = (0.54, 0.10, T + 0.09)       # the bar's middle during the hand-over (the left hand's knob at +0.20)

STEPS = ("pick up the bar", "hand it over", "into the fixture", "bolt it down", "done")
THREADS = (Thread("fix_thread", ("bolt",), TRAVEL, radius=0.008),)       # the bar's hole guides the bolt
_U_STEP, _U_ON = 0, 1
_BOUND: dict = {}


def threads(model):
    key = id(model)
    if key not in _BOUND:
        _BOUND[key] = [Thread(t.hole, t.bolts, t.travel, t.radius).bind(model) for t in THREADS]
    return _BOUND[key]


def _in_cradle(model, data) -> bool:
    p = data.xpos[model.body("bar").id]
    r = data.xmat[model.body("bar").id].reshape(3, 3)
    flat = r[2, 2] > math.cos(math.radians(4))
    along = abs(r[1, 1]) > math.cos(math.radians(6))
    return abs(p[0] - FIX[0]) < CRADLE_PLAY + 0.002 and abs(p[1] - FIX[1]) < 0.006 and \
        p[2] < T + FIX_H + BAR[2] + 0.003 and flat and along


def _held(model, data, side) -> bool:
    """A hand's palm close over one of the bar's knobs."""
    palm = data.site_xpos[model.site(f"{side}_palm").id]
    return any(np.linalg.norm(palm - data.site_xpos[model.site(f"bar_knob{k}_top").id]) < 0.06 for k in (0, 1))


def step(model, data) -> None:
    ths = threads(model)
    for t in ths:
        t.step(data)
    busy = {t.engaged(data) for t in ths} - {None}
    for t in ths:
        t.try_engage(model, data, busy)
    latch = model.equality("cradle_latch").id
    if not data.eq_active[latch] and _in_cradle(model, data):
        po, ro = data.xpos[model.body("cradle").id], data.xmat[model.body("cradle").id].reshape(3, 3)
        q, qn = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qn, data.xquat[model.body("cradle").id])
        mujoco.mju_mulQuat(q, qn, data.xquat[model.body("bar").id])
        model.eq_data[latch, 0:3] = 0
        model.eq_data[latch, 3:6] = ro.T @ (data.xpos[model.body("bar").id] - po)
        model.eq_data[latch, 6:10] = q
        model.eq_data[latch, 10] = 1.0
        data.eq_active[latch] = 1
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
    bar = data.xpos[model.body("bar").id]
    if k == 0:
        return bar[2] > T + SUPPORT_H + 0.05
    if k == 1:
        return bar[2] > T + 0.05 and _held(model, data, "l") and not _held(model, data, "r")
    if k == 2:
        return bool(data.eq_active[model.equality("cradle_latch").id])
    if k == 3:
        return threads(model)[0].seated(data)
    return bool(data.userdata[_U_ON])


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _knob(body, pos, name):
    x, y, z = pos
    body.add_geom(type=CYL, size=[0.008, KNOB_POST / 2, 0], pos=[x, y, z + KNOB_POST / 2], rgba=STEEL, mass=0.003)
    z += KNOB_POST
    body.add_geom(type=CYL, size=[KNOB_R, 0.0015, 0], pos=[x, y, z + 0.0015], rgba=STEEL, mass=0.002)
    body.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[x, y, z + 0.003 + KNOB_H / 2], rgba=KNOB, mass=0.012,
                  friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):
        a = 2 * math.pi * i / 10
        body.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                      pos=[x + KNOB_R * math.cos(a), y + KNOB_R * math.sin(a), z + 0.003 + KNOB_H / 2],
                      quat=quat_z(a), rgba=KNOB, contype=0, conaffinity=0, mass=0.0001)
    body.add_geom(type=CYL, size=[KNOB_R * 0.45, 0.0006, 0], pos=[x, y, z + 0.003 + KNOB_H + 0.0006], rgba=ORANGE,
                  contype=0, conaffinity=0, mass=0.0001)
    body.add_site(name=f"{name}_top", pos=[x, y, z + 0.003 + KNOB_H])


@scene("handover", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress,
       title="Pioneer v2  |  hand-over: right hand to left, into the fixture, bolted down")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec, half_y=0.8)
    spec.nuserdata = 2
    world = spec.worldbody
    grey, dark = [0.32, 0.34, 0.37, 1], [0.16, 0.22, 0.36, 1]
    alu = [0.78, 0.8, 0.84, 1]

    # the bar on its supports
    bx, by = BAR_START
    for dy in (-0.05, 0.05):
        sup = world.add_body(pos=[bx, by + dy, T])
        sup.add_geom(type=BOX, size=[0.025, 0.012, SUPPORT_H / 2], pos=[0, 0, SUPPORT_H / 2], rgba=grey)
    bar = world.add_body(name="bar", pos=[bx, by, T + SUPPORT_H + BAR[2] + 0.0005])
    bar.add_freejoint(name="bar")
    hx, hy, hz = BAR
    for y0, y1 in ((-hy, -HOLE), (HOLE, hy)):
        bar.add_geom(type=BOX, size=[hx, (y1 - y0) / 2, hz], pos=[0, (y0 + y1) / 2, 0], rgba=alu, mass=0.03,
                     friction=[1.0, 0.02, 0.002])
    for s in (-1, 1):
        bar.add_geom(type=BOX, size=[(hx - HOLE) / 2, HOLE, hz], pos=[s * (HOLE + (hx - HOLE) / 2), 0, 0], rgba=alu,
                     mass=0.003)
    bar.add_geom(type=BOX, size=[hx * 0.6, 0.03, 0.0005], pos=[0, 0.05, hz + 0.0005], rgba=ORANGE, contype=0,
                 conaffinity=0, mass=0.0001)
    for k, s in enumerate((-1, 1)):          # knob 0 on the bar's -y end, knob 1 on its +y end
        _knob(bar, [0, s * KNOB_Y, hz], f"bar_knob{k}")
    bar.add_site(name="bar_base", pos=[0, 0, -hz])

    # the fixture: a bed with side walls (the bar drops in), the thread under the bar's hole
    fx, fy = FIX
    cradle = world.add_body(name="cradle", pos=[fx, fy, T])
    cradle.add_geom(type=BOX, size=[0.04, 0.2, FIX_H / 2], pos=[0, 0, FIX_H / 2], rgba=dark)
    for s in (-1, 1):
        cradle.add_geom(type=BOX, size=[0.005, 0.2, 0.008], pos=[s * (hx + CRADLE_PLAY + 0.005), 0, FIX_H + 0.008],
                        rgba=ORANGE)
    add_thread_hole(spec, cradle, "fix_thread", [0, 0, FIX_H - THREAD_DEEP + 0.0], TRAVEL)
    add_weld(spec, "cradle_latch", "cradle", "bar", active=False)
    spec.add_exclude(bodyname1="cradle", bodyname2="bolt")

    # the bolt rack
    rx, ry = RACK
    rack = world.add_body(name="rack", pos=[rx, ry, T])
    rack.add_geom(type=BOX, size=[0.03, 0.03, 0.0015], pos=[0, 0, -0.0012], rgba=grey, contype=0, conaffinity=0)
    for ddx, ddy, sx, sy in ((0.009, 0, 0.004, 0.013), (-0.009, 0, 0.004, 0.013), (0, 0.009, 0.005, 0.004),
                             (0, -0.009, 0.005, 0.004)):
        rack.add_geom(type=BOX, size=[sx, sy, 0.008], pos=[ddx, ddy, 0.008], rgba=dark)
    add_knob_bolt(world, "bolt", [rx, ry, T + 0.0005], yaw=0.3, shaft=SHAFT)
    for t in THREADS:
        for b in t.bolts:
            add_bolt_weld(spec, t.hole, b)

    # the done button and its lamp
    bx_, by_ = BUTTON_POS
    housing = world.add_body(name="button_housing", pos=[bx_, by_, T])
    housing.add_geom(type=BOX, size=[0.03, 0.03, 0.02], pos=[0, 0, 0.02], rgba=grey)
    housing.add_geom(type=CYL, size=[0.024, 0.003, 0], pos=[0, 0, 0.043], rgba=[0.95, 0.8, 0.1, 1])
    cap = housing.add_body(name="button", pos=[0, 0, 0.046])
    cap.add_joint(name="button", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-BUTTON_TRAVEL, 0],
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=150.0, springref=0.0, damping=3.0)
    cap.add_geom(type=CYL, size=[0.02, 0.006, 0], pos=[0, 0, 0.006], rgba=[0.1, 0.65, 0.2, 1], mass=0.02)
    cap.add_site(name="button_top", pos=[0, 0, 0.012])
    spec.add_exclude(bodyname1="button_housing", bodyname2="button")
    post = world.add_body(name="lamp_post", pos=[fx + 0.08, fy - 0.12, T])
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
    th = threads(m)[0]
    d.qpos[m.joint("lamp").qposadr[0]] = -math.pi / 2
    wb.look([BAR_START[0], BAR_START[1], T], dur=0.8)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _hands(wb, HIGH, 0.8)
    wb.walk(0.30)

    # 1. the right hand takes the bar by its outer knob (knob 0, its -y end) and lifts it off the supports
    S.pick_knob(wb, "r", "bar_knob0_top", "bar", yaw=0.5, lift=0.08)

    # 2. across to the middle (the free knob, knob 1, now left of the middle); the left hand takes it, the right
    #    lets go
    hand_mid = np.array(HANDOVER)
    S.carry(wb, "r", "bar_base", hand_mid - [0, 0, BAR[2] * 2], clear=0.0, dur=1.6, over=0.03)
    wb.wait(0.3)
    _grasp(wb, "l", "bar_knob1_top", yaw=-0.5)
    S.release(wb, "r", up=0.06)
    wb.arm_joints_to("r", HIGH, 0.8)
    wb.settle()
    if not (_held(m, d, "l") and d.xpos[m.body("bar").id][2] > T + 0.05):
        raise S.TaskFailed("the hand-over dropped the bar")

    # 3. the left hand sets it in the fixture (the bar along y), down until it latches
    seat = np.array([FIX[0], FIX[1], T + FIX_H])
    S.carry(wb, "l", "bar_base", seat, clear=0.03, dur=1.6, over=0.03)
    latched = lambda: bool(d.eq_active[m.equality("cradle_latch").id])     # noqa: E731
    for _ in range(3):
        S.align_held(wb, "l", "bar", "bar_base", seat, yaw=0.0, sym=math.pi, rounds=2)
        S.lower_until(wb, "l", 0.045, latched, dur=1.0)
        wb.wait(0.2)
        if latched():
            break
        p, r = wb.hand_goal["l"]
        wb.reach("l", p + [0, 0, 0.02], r, 0.5)
        wb.settle()
    else:
        raise S.TaskFailed("the bar is not in the fixture")
    S.release(wb, "l", up=0.08)
    wb.arm_joints_to("l", HIGH, 0.8)
    wb.settle()

    # 4. bolt it down: a side step left to the rack, the bolt, back, through the bar's hole into the thread
    wb.look([RACK[0], RACK[1], T + 0.1], dur=0.5)
    _hands(wb, HIGH, 0.6)
    wb.step("l", 0.0, STEP)
    wb.step("r", 0.0, STEP)
    S.pick_knob(wb, "l", "bolt_top", "bolt", yaw=-0.4)
    wb.look([FIX[0], FIX[1], T + 0.05], dur=0.5)
    wb.step("r", 0.0, -STEP, carry=True)
    wb.step("l", 0.0, -STEP, carry=True)
    S.insert_bolt(wb, "l", th, "bolt", over=0.07, clear=THREAD_DEEP + 0.03)
    notes["strokes"] = S.tighten(wb, "l", th, "bolt", yaw0=-0.3, yaw1=-1.05, max_strokes=16)
    S.release(wb, "l", up=0.05, swing=True, ease=False)
    wb.arm_joints_to("l", READY, 1.0)
    wb.settle()

    # 5. done: right thumb on the button
    wb.arm_joints_to("r", READY, 1.0)
    wb.settle()
    top = S.site(wb, "button_top")
    wb.look(top, dur=0.5)
    wb.fingers("r", "fist", dur=0.4)
    S.press(wb, "r", "r_thumb_tip", top)
    wb.fingers("r", "relaxed", dur=0.4)
    wb.arm_joints_to("r", READY, 1.0)
    wb.look([FIX[0] + 0.08, FIX[1] - 0.12, T + 0.12], dur=0.6)
    wb.wait(2.0)
    notes["final"] = dict(latched=bool(latched()), bolt=round(th.depth(d), 4), on=bool(d.userdata[_U_ON]))


def _grasp(wb, side: str, knob: str, yaw: float) -> None:
    """Take a knob of a part the other hand holds (from above, wrapped over the top)."""
    from pioneer_whole_body import skills as S
    top = S.site(wb, knob)
    p, r = S.knob_pose(top, yaw, None)
    wb.look(top, dur=0.4)
    wb.fingers(side, "lid_pre", dur=0.5)
    wb.reach(side, p + [0, 0, S.CLEAR], r, 1.2)
    wb.settle()
    top = S.site(wb, knob)                       # (the held part sways a little): aim again
    p, r = S.knob_pose(top, yaw, None)
    wb.reach(side, p, r, 0.7)
    wb.settle()
    wb.fingers(side, "lid", amount=S.SQUEEZE, dur=0.45)
    wb.settle()
    wb.wait(0.2)


def _hands(wb, pose, dur: float = 1.0) -> None:
    for side in ("l", "r"):
        wb.arm_joints_to(side, pose, dur)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["handover"].plan = plan
