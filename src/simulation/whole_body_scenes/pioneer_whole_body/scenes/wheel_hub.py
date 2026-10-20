"""Wheel hub (long horizon, whole body): mount a wheel on its axle, put the hub nut on, spin-test it.

The robot walks up to the bench. On it:

  axle stand   in front: a vertical axle on a base with a collar the wheel rests on; the axle is threaded
               near the collar (the hub nut's thread, its mouth just above a mounted wheel's hub)
  wheel        a side step to the left, on a stand: rim, spokes, hub, two handle knobs on the rim
  hub nut      in a rack on the right

Steps, each checked from the simulation and latched in order (``progress``):
  1. mount the wheel on the axle      taken by both handles, carried over, lowered over the axle onto the collar
  2. start the hub nut                down over the axle top, its thread caught
  3. tighten the hub nut              screwed home (it retains the wheel; the wheel still turns)
  4. spin the wheel                   pushed round by a fingertip on a handle: one full turn

The axle goes through the nut (its bore isn't modelled: their contacts are off); the wheel turns on the axle by
contact, as on a plain bearing.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY, palm_down
from pioneer_whole_body.parts import (DARK, KNOB, KNOB_H, KNOB_R, ORANGE, STEEL, Thread, add_bolt_weld,
                                      add_knob_bolt, add_thread_hole, quat_z)
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

AXLE_POS = (0.52, 0.0)
BASE_H = 0.02
AXLE_R = 0.009
COLLAR = (0.022, 0.006)                 # radius, height: the wheel rests on it
AXLE_TOP = 0.085                        # above the base's top
WHEEL_R = 0.11
WHEEL_H = 0.02
HUB_R_IN = 0.0125                       # 3.5 mm of play round the axle
HUB_R_OUT = 0.032
HANDLE_R = 0.088                        # the two handle knobs on the rim, opposite each other
TRAVEL = 0.009
STEP = 0.15
WHEEL_SIDE = 3 * STEP
NUT_RACK = (0.48, -0.20)
NUT_SHAFT = 0.0                         # the nut is all knob: its "tip" is its underside
MOUNTED_Z = T + BASE_H + COLLAR[1]      # a mounted wheel's underside
MOUTH_Z = MOUNTED_Z + WHEEL_H + TRAVEL + 0.001

STEPS = ("mount the wheel on the axle", "start the hub nut", "tighten the hub nut", "spin the wheel")
THREADS = (Thread("hub_thread", ("nut",), TRAVEL, radius=0.007),)    # the axle centres the nut
_U_STEP, _U_TURN, _U_LAST = 0, 1, 2
_BOUND: dict = {}


def threads(model):
    key = id(model)
    if key not in _BOUND:
        _BOUND[key] = [Thread(t.hole, t.bolts, t.travel, t.radius).bind(model) for t in THREADS]
    return _BOUND[key]


def wheel_mounted(model, data) -> bool:
    w = data.xpos[model.body("wheel").id]
    flat = data.xmat[model.body("wheel").id][8] > math.cos(math.radians(6))
    return math.hypot(w[0] - AXLE_POS[0], w[1] - AXLE_POS[1]) < HUB_R_IN - AXLE_R + 0.001 and \
        abs(w[2] - MOUNTED_Z) < 0.004 and flat


def wheel_yaw(model, data) -> float:
    r = data.xmat[model.body("wheel").id].reshape(3, 3)
    return math.atan2(r[1, 0], r[0, 0])


def step(model, data) -> None:
    for t in threads(model):
        t.step(data)
        t.try_engage(model, data, set())
    # the wheel's turning, counted while it's on the axle under its nut
    a = wheel_yaw(model, data)
    if wheel_mounted(model, data) and threads(model)[0].seated(data):
        d = (a - data.userdata[_U_LAST] + math.pi) % (2 * math.pi) - math.pi
        data.userdata[_U_TURN] += abs(d)
    data.userdata[_U_LAST] = a
    k = int(data.userdata[_U_STEP])
    if k < len(STEPS) and _step_done(model, data, k):
        data.userdata[_U_STEP] = k + 1


def _step_done(model, data, k) -> bool:
    th = threads(model)[0]
    if k == 0:
        return wheel_mounted(model, data)
    if k == 1:
        return th.engaged(data) is not None
    if k == 2:
        return th.seated(data)
    return data.userdata[_U_TURN] >= 2 * math.pi


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _ring(body, name, r_in, r_out, h, n, rgba, z0=0.0, mass=None):
    kw = {} if mass is None else {"mass": mass / n}
    w = r_out - r_in
    half_t = r_out * math.tan(math.pi / n) + 0.0005
    for i in range(n):
        a = 2 * math.pi * i / n
        body.add_geom(name=f"{name}_{i}", type=BOX, size=[w / 2, half_t, h / 2],
                      pos=[(r_in + w / 2) * math.cos(a), (r_in + w / 2) * math.sin(a), z0 + h / 2],
                      quat=quat_z(a), rgba=rgba, **kw)


@scene("wheel_hub", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress,
       title="Pioneer v2  |  wheel hub: fetch the wheel with both hands, mount it, hub nut on, spin test")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec, half_y=0.8)
    spec.nuserdata = 3
    world = spec.worldbody
    grey, dark = [0.32, 0.34, 0.37, 1], [0.12, 0.13, 0.15, 1]

    # axle stand
    ax, ay = AXLE_POS
    stand = world.add_body(name="axle_stand", pos=[ax, ay, T])
    stand.add_geom(type=BOX, size=[0.06, 0.06, BASE_H / 2], pos=[0, 0, BASE_H / 2], rgba=grey)
    stand.add_geom(name="collar", type=CYL, size=[COLLAR[0], COLLAR[1] / 2, 0], pos=[0, 0, BASE_H + COLLAR[1] / 2],
                   rgba=STEEL, friction=[0.12, 0.005, 0.0001])
    stand.add_geom(name="axle", type=CYL, size=[AXLE_R, AXLE_TOP / 2, 0], pos=[0, 0, BASE_H + AXLE_TOP / 2],
                   rgba=STEEL, friction=[0.2, 0.005, 0.0001])
    add_thread_hole(spec, stand, "hub_thread", [0, 0, MOUTH_Z - T], TRAVEL)

    # the wheel on its stand, a side step to the left
    wx, wy = AXLE_POS[0], WHEEL_SIDE
    wstand = world.add_body(name="wheel_stand", pos=[wx, wy, T])
    wstand.add_geom(type=CYL, size=[0.05, 0.015, 0], pos=[0, 0, 0.015], rgba=grey)
    wheel = world.add_body(name="wheel", pos=[wx, wy, T + 0.03])
    wheel.add_freejoint(name="wheel")
    rim = [0.85, 0.87, 0.9, 1]
    _ring(wheel, "hub", HUB_R_IN, HUB_R_OUT, WHEEL_H, 14, ORANGE, mass=0.06)
    _ring(wheel, "rim", WHEEL_R - 0.014, WHEEL_R, WHEEL_H, 28, rim, mass=0.12)
    _ring(wheel, "tyre", WHEEL_R, WHEEL_R + 0.008, WHEEL_H * 0.8, 28, dark, z0=WHEEL_H * 0.1, mass=0.04)
    for i in range(5):                                      # spokes
        a = 2 * math.pi * i / 5 + math.pi / 10
        mid = (HUB_R_OUT + WHEEL_R - 0.014) / 2
        wheel.add_geom(type=BOX, size=[(WHEEL_R - 0.014 - HUB_R_OUT) / 2 + 0.002, 0.007, WHEEL_H * 0.35],
                       pos=[mid * math.cos(a), mid * math.sin(a), WHEEL_H / 2], quat=quat_z(a), rgba=rim, mass=0.01)
    for k, a in enumerate((math.pi / 2, -math.pi / 2)):        # handle knobs (left, right in the wheel's frame)
        x, y = HANDLE_R * math.cos(a), HANDLE_R * math.sin(a)
        wheel.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[x, y, WHEEL_H + KNOB_H / 2], rgba=KNOB,
                       mass=0.015, friction=[2.0, 0.03, 0.003], condim=4)
        wheel.add_site(name=f"wheel_handle{k}", pos=[x, y, WHEEL_H + KNOB_H])
        wheel.add_site(name=f"wheel_handle{k}_mid", pos=[x, y, WHEEL_H + KNOB_H / 2])
    wheel.add_site(name="wheel_base", pos=[0, 0, 0])
    wheel.add_site(name="wheel_top", pos=[0, 0, WHEEL_H])

    # hub nut in its rack on the right (a knob with no shaft: its underside goes onto the thread)
    nx, ny = NUT_RACK
    rack = world.add_body(name="nut_rack", pos=[nx, ny, T])
    rack.add_geom(type=BOX, size=[0.03, 0.03, 0.025], pos=[0, 0, 0.025], rgba=grey)
    add_knob_bolt(world, "nut", [nx, ny, T + 0.05], shaft=NUT_SHAFT, color=ORANGE)
    add_bolt_weld(spec, "hub_thread", "nut")
    spec.add_exclude(bodyname1="axle_stand", bodyname2="nut")


def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S

    m, d = wb.m, wb.d
    th = threads(m)[0]
    wb.look([AXLE_POS[0], 0.0, T], dur=0.8)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _hands(wb, HIGH, 0.8)
    wb.walk(0.30)
    wb.look([AXLE_POS[0], WHEEL_SIDE, T + 0.04], dur=0.8)
    wb.wait(0.3)

    # 1. the wheel: side step left, both handles, back, down over the axle
    _side_step(wb, WHEEL_SIDE)
    S.pick_two(wb, {"l": ("wheel_handle0", -0.3), "r": ("wheel_handle1", 0.3)}, "wheel", lift=0.08, shape="lid",
               above=S.GRASP_ABOVE, squeeze=1.25)
    wb.look([AXLE_POS[0], 0.0, T + 0.08], dur=0.6)
    _side_step(wb, -WHEEL_SIDE, carry=True)
    hub = np.array([AXLE_POS[0], AXLE_POS[1], MOUNTED_Z])
    over = AXLE_TOP + BASE_H + 0.02 - (MOUNTED_Z - T)           # hub bottom clear of the axle top
    base = S.site(wb, "wheel_base")
    S.move_two(wb, [hub[0] - base[0], hub[1] - base[1], over + hub[2] - base[2]], dur=1.3)
    S.align_two(wb, "wheel", "wheel_base", hub, yaw=0.0, rounds=3, sym=2 * math.pi)
    for side in ("l", "r"):
        p, r = wb.hand_goal[side]
        wb.reach(side, p - [0, 0, over + 0.01], r, 1.6)
    wb.until(lambda: wheel_mounted(m, d), 2.0)
    wb.wait(0.3)
    if S.DEBUG:
        for side in ("l", "r"):
            pc, _ = wb.palm_pose(side, True)
            ps, _ = wb.palm_pose(side, False)
            print("before release", side, "goal", np.round(wb.hand_goal[side][0], 3), "cmd-sim", np.round(pc - ps, 3))
    S.let_go(wb, "wheel", up=0.08)
    _hands(wb, HIGH)
    if not wheel_mounted(m, d):
        raise S.TaskFailed("wheel not on the axle")

    # 2-3. the hub nut: right hand, down over the axle until it catches, then screw it home
    S.pick_knob(wb, "r", "nut_top", "nut", yaw=0.9, lift=0.06)
    mouth = d.xpos[th.hole_id].copy()
    S.carry(wb, "r", "nut_tip", mouth + [0, 0, AXLE_TOP + BASE_H - (MOUTH_Z - T)], clear=0.02, dur=1.4)
    S.align_held(wb, "r", "nut", "nut_tip", mouth, yaw=None, rounds=2)
    p, r = wb.hand_goal["r"]
    wb.reach("r", p - [0, 0, (S.site(wb, "nut_tip") - mouth)[2] + 0.01], r, 1.6)
    if not wb.until(lambda: th.engaged(d) is not None, 2.0):
        raise S.TaskFailed("hub nut did not catch")
    wb.wait(0.2)
    notes["strokes"] = S.tighten(wb, "r", th, "nut", yaw0=1.1, yaw1=0.25, max_strokes=16)
    S.release(wb, "r", up=0.06)
    _hands(wb, HIGH)

    # 4. spin test: the index finger pushes a handle round, twice
    wb.fingers("l", "point", dur=0.4)
    for push in range(5):
        if d.userdata[_U_TURN] >= 2 * math.pi:
            break
        _push_round(wb, "l", S)
        notes.setdefault("turn_after_push", []).append(round(float(d.userdata[_U_TURN]), 2))
    wb.fingers("l", "relaxed", dur=0.4)
    _hands(wb, READY)
    wb.look([AXLE_POS[0], AXLE_POS[1], T + 0.05], dur=0.5)
    wb.wait(1.5)
    notes["final"] = dict(mounted=bool(wheel_mounted(m, d)), nut=(th.engaged(d), round(th.depth(d), 4)),
                          turned=round(float(d.userdata[_U_TURN]), 2))


def _push_round(wb, side, S) -> None:
    """Fingertip on the near handle's outer side, then round the axle (counterclockwise seen from above)."""
    m, d = wb.m, wb.d
    c = np.array([AXLE_POS[0], AXLE_POS[1]])
    sites = [f"wheel_handle{k}_mid" for k in range(2)]
    # the handle nearest the robot's left front, pushed from behind (its clockwise side)
    pts = [S.site(wb, s) for s in sites]
    angs = [math.atan2(p[1] - c[1], p[0] - c[0]) for p in pts]
    k = int(np.argmin([abs(((a - math.radians(-150)) + math.pi) % (2 * math.pi) - math.pi) for a in angs]))
    a0, z = angs[k], pts[k][2]
    rad = HANDLE_R
    tip = f"{side}_index_tip"
    behind = a0 - (KNOB_R + 0.012) / rad
    p0 = np.array([c[0] + rad * math.cos(behind), c[1] + rad * math.sin(behind), z])
    wb.look([c[0], c[1], T + 0.05], dur=0.4)
    wb.reach_tip(side, tip, p0 + [0, 0, 0.06], 1.0)
    wb.settle()
    wb.reach_tip(side, tip, p0, 0.5)
    wb.settle()
    for i in range(1, 9):                                      # 140 degrees of arc, in short chords
        a = behind + math.radians(140) * i / 8
        wb.reach_tip(side, tip, [c[0] + rad * math.cos(a), c[1] + rad * math.sin(a), z], 0.16)
        wb.settle()
    p, _ = wb.hand_goal[side]
    wb.reach_tip(side, tip, p + [0, 0, 0.07], 0.4)
    wb.settle()


def _hands(wb, pose, dur: float = 1.0) -> None:
    for side in ("l", "r"):
        wb.arm_joints_to(side, pose, dur)
    wb.settle()


def _side_step(wb, dy: float, carry: bool = False) -> None:
    lead, trail = ("l", "r") if dy > 0 else ("r", "l")
    for _ in range(max(1, round(abs(dy) / STEP))):
        wb.step(lead, 0.0, math.copysign(STEP, dy), carry=carry)
        wb.step(trail, 0.0, math.copysign(STEP, dy), carry=carry)


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["wheel_hub"].plan = plan
