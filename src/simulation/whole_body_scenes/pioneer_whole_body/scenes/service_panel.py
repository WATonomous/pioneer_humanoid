"""Service panel (long horizon, whole body, both hands): fetch a cartridge from a drawer, fit it in a panel behind
a sliding hatch, bolt the hatch shut, close the drawer, power on.

The robot walks up to the bench. On it:

  parts cabinet   on the left: a drawer that pulls out to the right (a knob on a bracket at its front) with a
                  cartridge standing in it, out of reach under the cabinet's top until the drawer is out
  panel           in front: a cartridge bay under a hatch that slides to the right (a knob on its right end); the
                  hatch's bolt hole lines up with a thread in the panel when it's shut
  bolt rack       on the right: the hatch's knob bolt
  power button    on the right, behind the rack: the panel's lamp flag comes up if it's pressed with the cartridge
                  in and the hatch bolted

Steps, each checked from the simulation and latched in order (``progress``):
  1. open the drawer             left hand: grip its knob, pull it out
  2. take the cartridge out      left hand
  3. slide the hatch open        right hand
  4. fit the cartridge           left hand: into the bay, standing
  5. slide the hatch shut        right hand
  6. bolt the hatch              right hand: the bolt from the rack, through the hatch, screwed home
  7. close the drawer            left hand: pushed back in
  8. power on                    right thumb on the power button: the lamp comes up
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY
from pioneer_whole_body.parts import (KNOB, KNOB_H, KNOB_R, ORANGE, STEEL, Thread, add_bolt_weld, add_knob_bolt,
                                      add_thread_hole, quat_z)
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

CART_R, CART_H = 0.019, 0.062           # the cartridge (the cells' size: the grasp is solved for it)
SOCKET_R = 0.025
BAY_R, BAY_H = 0.027, 0.012                 # the bay's ring: 8 mm of play, low (a cartridge doesn't catch its rim)
# the panel: a box along y, open on top, the hatch over it
PANEL_X = 0.52
PANEL = (0.08, 0.22)                    # outer half sizes (x, y), centred at y = PANEL_Y
PANEL_Y = -0.02
PANEL_H = 0.075                         # top of the walls
LOW_WALL = 0.035
WALL = 0.006
BAY_Y = 0.15                            # the cartridge bay (left end)
HOLE_Y = -0.10                          # the hatch's bolt / the thread (right end)
HATCH = (0.08, 0.20, 0.004)             # half sizes; shut it spans y -0.20 .. 0.20
HATCH_OPEN = 0.16                       # it slides this far to the right
HATCH_KNOB_Y = -0.19                   # hatch frame
TRAVEL = 0.009
THREAD_DEEP = 0.045                     # the thread starts this far down in the panel (a long bolt: the fingers on
                                        # its knob stay clear of the hatch while it catches)
SHAFT = 2 * HATCH[2] + THREAD_DEEP + TRAVEL - 0.0005
# the parts cabinet and its drawer (slides along x, towards the robot)
CAB_X = 0.52
CAB_FRONT = 0.45                        # y of the drawer's front when shut (it slides out to the right)
KNOB_X = 0.05                           # the drawer knob's x off the drawer's centre (away from the robot: clear of the forearm)
CAB_TOP_Z = T + 0.084
DRAWER_OPEN = 0.155
DRAWER = (0.10, 0.06, 0.012)            # tray outer half sizes (x, y, z)
KNOB_POST = 0.03                        # the drawer knob stands on a post (the hand can't go lower)
CART_IN_DRAWER = 0.08                   # the cartridge's x behind the drawer front
# bolt rack and power button
RACK = (0.48, -0.44)
BUTTON_POS = (0.60, -0.32)
BUTTON_TRAVEL = 0.008
DETENT, DETENT_F = 0.02, 12.0          # the slides' open-end detents (m, N)
SOFT_CLOSE, SOFT_CLOSE_F = 0.03, 2.0    # the drawer's and hatch's soft-close (m, N)

STEPS = ("open the drawer", "take the cartridge out", "slide the hatch open", "fit the cartridge",
         "slide the hatch shut", "bolt the hatch", "close the drawer", "power on")
THREADS = (Thread("hatch_thread", ("bolt",), TRAVEL, radius=0.008),)   # the hatch's hole guides the bolt to it
_U_STEP, _U_ON = 0, 1
_BOUND: dict = {}


def threads(model):
    key = id(model)
    if key not in _BOUND:
        _BOUND[key] = [Thread(t.hole, t.bolts, t.travel, t.radius).bind(model) for t in THREADS]
    return _BOUND[key]


def drawer_out(model, data) -> float:
    return float(data.qpos[model.joint("drawer").qposadr[0]])


def hatch_open(model, data) -> float:
    return -float(data.qpos[model.joint("hatch").qposadr[0]])


def cartridge_in(model, data) -> bool:
    c = data.xpos[model.body("cartridge").id]
    up = data.xmat[model.body("cartridge").id][8] > math.cos(math.radians(12))
    return math.hypot(c[0] - PANEL_X, c[1] - (PANEL_Y + BAY_Y)) < BAY_R and c[2] < T + 0.008 and up


def cartridge_out(model, data) -> bool:
    """Out of the drawer: held up clear of it."""
    return data.xpos[model.body("cartridge").id][2] > T + 0.04


def step(model, data) -> None:
    ths = threads(model)
    for t in ths:
        t.step(data)
    busy = {t.engaged(data) for t in ths} - {None}
    for t in ths:
        t.try_engage(model, data, busy)
    # detents at the open ends: a spring holds the drawer / hatch fully open (letting go of the knob can't shove
    # them back), pushing them shut overcomes it
    for j, sign, travel in (("drawer", 1.0, DRAWER_OPEN), ("hatch", -1.0, HATCH_OPEN)):
        q = sign * data.qpos[model.joint(j).qposadr[0]]
        f = sign * DETENT_F if q > travel - DETENT else 0.0
        if q < SOFT_CLOSE:                             # and soft-close: pulled shut over the last 3 cm (a hatch
            f = -sign * SOFT_CLOSE_F                    # pushed on by the bolt stays put)
        data.qfrc_applied[model.joint(j).dofadr[0]] = f
    if data.qpos[model.joint("button").qposadr[0]] < -0.6 * BUTTON_TRAVEL:
        if cartridge_in(model, data) and ths[0].seated(data) and hatch_open(model, data) < 0.003:
            data.userdata[_U_ON] = 1
    a = model.joint("lamp").qposadr[0]
    target = 0.0 if data.userdata[_U_ON] else -math.pi / 2
    data.qpos[a] += float(np.clip(target - data.qpos[a], -0.004, 0.004))
    data.qvel[model.joint("lamp").dofadr[0]] = 0.0
    k = int(data.userdata[_U_STEP])
    if k < len(STEPS) and _step_done(model, data, k):
        data.userdata[_U_STEP] = k + 1


def _step_done(model, data, k) -> bool:
    if k == 0:
        return drawer_out(model, data) > DRAWER_OPEN - 0.015
    if k == 1:
        return cartridge_out(model, data)
    if k == 2:
        return hatch_open(model, data) > HATCH_OPEN - 0.015
    if k == 3:
        return cartridge_in(model, data)
    if k == 4:
        return hatch_open(model, data) < 0.003
    if k == 5:
        return threads(model)[0].seated(data)
    if k == 6:
        return drawer_out(model, data) < 0.005
    return bool(data.userdata[_U_ON])


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _ring(body, r, h, wall, n, rgba, z0=0.0):
    half_t = (r + wall) * math.tan(math.pi / n) + 0.0005
    for i in range(n):
        a = 2 * math.pi * i / n
        body.add_geom(type=BOX, size=[wall / 2, half_t, h / 2],
                      pos=[(r + wall / 2) * math.cos(a), (r + wall / 2) * math.sin(a), z0 + h / 2],
                      quat=quat_z(a), rgba=rgba)


def _knob(body, pos, name):
    """A fixed pull knob (the bolts' knob) standing on ``body`` at ``pos``."""
    x, y, z = pos
    body.add_geom(type=CYL, size=[KNOB_R, 0.0015, 0], pos=[x, y, z + 0.0015], rgba=STEEL, mass=0.003)
    body.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[x, y, z + 0.003 + KNOB_H / 2], rgba=KNOB, mass=0.02,
                  friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):
        a = 2 * math.pi * i / 10
        body.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                      pos=[x + KNOB_R * math.cos(a), y + KNOB_R * math.sin(a), z + 0.003 + KNOB_H / 2],
                      quat=quat_z(a), rgba=KNOB, contype=0, conaffinity=0, mass=0.0001)
    body.add_geom(type=CYL, size=[KNOB_R * 0.45, 0.0006, 0], pos=[x, y, z + 0.003 + KNOB_H + 0.0006], rgba=ORANGE,
                  contype=0, conaffinity=0, mass=0.0001)
    body.add_site(name=f"{name}_top", pos=[x, y, z + 0.003 + KNOB_H])


@scene("service_panel", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress,
       title="Pioneer  |  service panel: drawer, cartridge, sliding hatch, bolt, power on")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec, half_y=0.7)
    spec.nuserdata = 2
    world = spec.worldbody
    grey, dark = [0.32, 0.34, 0.37, 1], [0.16, 0.22, 0.36, 1]
    panel_rgba = [0.86, 0.87, 0.89, 1]

    # the panel: floor, walls, a block under the thread, the bay's socket
    panel = world.add_body(name="panel", pos=[PANEL_X, PANEL_Y, T])
    px, py = PANEL
    panel.add_geom(type=BOX, size=[px, py, 0.003], pos=[0, 0, 0.003], rgba=dark)
    for s in (-1, 1):       # the near wall and the left end are low: the forearm reaches in over them
        h = PANEL_H if s > 0 else LOW_WALL
        panel.add_geom(type=BOX, size=[WALL / 2, py, h / 2], pos=[s * (px - WALL / 2), 0, h / 2], rgba=panel_rgba)
        h = PANEL_H if s < 0 else LOW_WALL
        panel.add_geom(type=BOX, size=[px, WALL / 2, h / 2], pos=[0, s * (py - WALL / 2), h / 2], rgba=panel_rgba)
    panel.add_geom(type=BOX, size=[px - WALL, 0.03, PANEL_H / 2], pos=[0, HOLE_Y, PANEL_H / 2], rgba=grey)
    panel.add_geom(type=BOX, size=[px - WALL, 0.012, PANEL_H / 2], pos=[0, 0.0, PANEL_H / 2], rgba=grey)
    bay = panel.add_body(name="bay", pos=[0, BAY_Y, 0.006])
    _ring(bay, BAY_R, BAY_H, 0.004, 12, [0.93, 0.47, 0.16, 1])
    add_thread_hole(spec, panel, "hatch_thread", [0, HOLE_Y, PANEL_H - THREAD_DEEP], TRAVEL)
    # the hatch: a plate on a slide, a hole for the bolt, a pull knob
    hatch = panel.add_body(name="hatch", pos=[0, 0, PANEL_H + HATCH[2]])
    hatch.add_joint(name="hatch", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[-HATCH_OPEN, 0],
                    limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=6.0, frictionloss=0.4, armature=0.2)
    hx, hy, hz = HATCH
    hole = 0.0095                                     # 5 mm of play round the bolt
    for y0, y1 in ((-hy, HOLE_Y - hole), (HOLE_Y + hole, hy)):
        hatch.add_geom(type=BOX, size=[hx, (y1 - y0) / 2, hz], pos=[0, (y0 + y1) / 2, 0], rgba=[0.2, 0.45, 0.75, 1],
                       mass=0.08)
    for s in (-1, 1):
        hatch.add_geom(type=BOX, size=[(hx - hole) / 2, hole, hz], pos=[s * (hole + (hx - hole) / 2), HOLE_Y, 0],
                       rgba=[0.2, 0.45, 0.75, 1], mass=0.005)
    hatch.add_geom(type=BOX, size=[hx * 0.6, 0.04, 0.0006], pos=[0, BAY_Y - 0.02, hz + 0.0006], rgba=ORANGE, contype=0,
                   conaffinity=0, mass=0.0001)                         # label over the bay
    _knob(hatch, [0.04, HATCH_KNOB_Y, hz], "hatch_knob")       # on the far side: clear of the bolt's path
    spec.add_exclude(bodyname1="panel", bodyname2="hatch")
    spec.add_exclude(bodyname1="panel", bodyname2="bolt")
    # the lamp: a flag on a post at the panel's back right corner
    post = panel.add_body(name="lamp_post", pos=[px + 0.02, -py + 0.04, 0])
    post.add_geom(type=CYL, size=[0.006, 0.05, 0], pos=[0, 0, 0.05], rgba=grey, contype=0, conaffinity=0)
    flag = post.add_body(name="lamp", pos=[0, 0, 0.1])
    flag.add_joint(name="lamp", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], ref=0.0)
    flag.add_geom(type=BOX, size=[0.003, 0.02, 0.03], pos=[0, 0, 0.03], rgba=[0.15, 0.85, 0.3, 1], contype=0,
                  conaffinity=0, mass=0.01)
    flag.gravcomp = 1.0

    # the parts cabinet: a top over the drawer's back, side cheeks; the drawer slides out towards the robot
    dx, dy, dz = DRAWER
    cab = world.add_body(name="cabinet", pos=[CAB_X, CAB_FRONT, T], quat=quat_z(math.pi / 2))   # local x: into it
    depth = 2 * dx + 0.02
    cab.add_geom(type=BOX, size=[(depth - 0.03) / 2, dy + 0.014, 0.004],
                 pos=[0.03 + (depth - 0.03) / 2, 0, CAB_TOP_Z - T - 0.004], rgba=grey)   # over the cartridge
    for s in (-1, 1):
        cab.add_geom(type=BOX, size=[depth / 2, 0.006, (CAB_TOP_Z - T) / 2],
                     pos=[depth / 2, s * (dy + 0.008), (CAB_TOP_Z - T) / 2], rgba=grey)
    cab.add_geom(type=BOX, size=[0.006, dy + 0.014, (CAB_TOP_Z - T) / 2],
                 pos=[depth + 0.006, 0, (CAB_TOP_Z - T) / 2], rgba=grey)
    drawer = cab.add_body(name="drawer", pos=[0, 0, 0.001])                 # on its runners, clear of the bench
    drawer.add_joint(name="drawer", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, DRAWER_OPEN],
                     limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=6.0, frictionloss=0.4, armature=0.2)
    wood = [0.74, 0.62, 0.47, 1]
    drawer.add_geom(type=BOX, size=[dx, dy, 0.003], pos=[dx, 0, 0.004], rgba=wood, mass=0.1)
    drawer.add_geom(type=BOX, size=[0.005, dy, dz], pos=[0.005, 0, dz], rgba=[0.2, 0.45, 0.75, 1], mass=0.05)
    drawer.add_geom(type=BOX, size=[0.004, dy, dz * 0.7], pos=[2 * dx - 0.004, 0, dz * 0.7], rgba=wood, mass=0.03)
    for s in (-1, 1):
        drawer.add_geom(type=BOX, size=[dx, 0.004, dz * 0.7], pos=[dx, s * (dy - 0.004), dz * 0.7], rgba=wood,
                        mass=0.03)
    nest = drawer.add_body(name="drawer_socket", pos=[CART_IN_DRAWER, 0, 0.007])
    _ring(nest, SOCKET_R - 0.002, 0.012, 0.004, 12, grey)
    drawer.add_geom(type=BOX, size=[0.028, 0.012, 0.004], pos=[-0.022, -KNOB_X, 2 * dz - 0.004], rgba=STEEL, mass=0.01)
    drawer.add_geom(type=CYL, size=[0.008, KNOB_POST / 2, 0], pos=[-0.05, -KNOB_X, 2 * dz + KNOB_POST / 2], rgba=STEEL,
                    mass=0.005)
    _knob(drawer, [-0.05, -KNOB_X, 2 * dz + KNOB_POST], "drawer_knob")   # on a bracket out front, clear of the cabinet
    spec.add_exclude(bodyname1="cabinet", bodyname2="drawer")
    spec.add_exclude(bodyname1="world", bodyname2="drawer")

    # the cartridge, standing in the drawer
    c = world.add_body(name="cartridge", pos=[CAB_X, CAB_FRONT + CART_IN_DRAWER, T + 0.0085])
    c.add_freejoint(name="cartridge")
    c.add_geom(type=CYL, size=[CART_R, CART_H / 2 - 0.004, 0], pos=[0, 0, CART_H / 2], rgba=[0.55, 0.2, 0.62, 1],
               mass=0.045, friction=[1.6, 0.03, 0.003], condim=4)
    c.add_geom(type=CYL, size=[CART_R * 0.98, 0.002, 0], pos=[0, 0, CART_H - 0.002], rgba=ORANGE, mass=0.002,
               friction=[1.6, 0.03, 0.003], condim=4)
    c.add_geom(type=CYL, size=[CART_R * 0.98, 0.002, 0], pos=[0, 0, 0.002], rgba=STEEL, mass=0.002)
    c.add_site(name="cartridge_top", pos=[0, 0, CART_H])
    c.add_site(name="cartridge_base", pos=[0, 0, 0])

    # bolt rack
    rx, ry = RACK
    rack = world.add_body(name="rack", pos=[rx, ry, T])
    rack.add_geom(type=BOX, size=[0.03, 0.03, 0.0015], pos=[0, 0, -0.0012], rgba=grey, contype=0, conaffinity=0)
    for ddx, ddy, sx, sy in ((0.009, 0, 0.004, 0.013), (-0.009, 0, 0.004, 0.013), (0, 0.009, 0.005, 0.004),
                             (0, -0.009, 0.005, 0.004)):
        rack.add_geom(type=BOX, size=[sx, sy, 0.008], pos=[ddx, ddy, 0.008], rgba=dark)
    add_knob_bolt(world, "bolt", [rx, ry, T + 0.0005], yaw=-0.3, shaft=SHAFT)
    for t in THREADS:
        for b in t.bolts:
            add_bolt_weld(spec, t.hole, b)

    # power button
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


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S

    m, d = wb.m, wb.d
    th = threads(m)[0]
    d.qpos[m.joint("lamp").qposadr[0]] = -math.pi / 2
    wb.look([PANEL_X, PANEL_Y, T], dur=0.8)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _hands(wb, HIGH, 0.8)
    wb.walk(0.30)

    # 1. open the drawer: grip the knob, pull straight out
    _pull(wb, "l", "drawer_knob", [0, -DRAWER_OPEN - 0.025, 0], yaw=-0.7)
    if drawer_out(m, d) < DRAWER_OPEN - 0.015:
        raise S.TaskFailed(f"drawer only {drawer_out(m, d):.3f} out")
    wb.arm_joints_to("l", HIGH, 0.8)
    wb.settle()

    # 2. the cartridge out of the drawer (held up out of the way)
    S.pick_knob(wb, "l", "cartridge_top", "cartridge", yaw=-0.8, lift=0.10)

    # 3. the hatch open: right hand slides it to the right
    _pull(wb, "r", "hatch_knob", [0, -HATCH_OPEN, 0], yaw=0.7)
    wb.arm_joints_to("r", HIGH, 0.8)
    wb.settle()

    # 4. the cartridge into the bay
    bay = np.array([PANEL_X, PANEL_Y + BAY_Y, T + 0.006])
    S.carry(wb, "l", "cartridge_base", bay, clear=0.03, dur=1.4, over=0.06)
    down = lambda: d.xpos[m.body("cartridge").id][2] < T + 0.009          # noqa: E731
    for _ in range(3):
        S.align_held(wb, "l", "cartridge", "cartridge_base", bay, yaw=None, rounds=2)
        S.lower_until(wb, "l", 0.045, down, dur=1.0)
        wb.wait(0.2)
        if down():
            break
        p, r = wb.hand_goal["l"]                     # caught on the bay's rim: up a little, line up again
        wb.reach("l", p + [0, 0, 0.02], r, 0.5)
        wb.settle()
    S.release(wb, "l", up=0.10)
    wb.arm_joints_to("l", HIGH, 0.8)
    wb.settle()
    if not cartridge_in(m, d):
        raise S.TaskFailed("cartridge not in its bay")

    # 5. the hatch shut
    _pull(wb, "r", "hatch_knob", [0, HATCH_OPEN + 0.01, 0], yaw=0.7)
    wb.arm_joints_to("r", HIGH, 0.8)
    wb.settle()
    if hatch_open(m, d) > 0.003:
        raise S.TaskFailed("hatch not shut")

    # 6. bolt it: the bolt from the rack, through the hatch's hole, screwed home
    S.pick_knob(wb, "r", "bolt_top", "bolt", yaw=0.6)
    S.insert_bolt(wb, "r", th, "bolt", over=0.07, clear=THREAD_DEEP + 0.03)   # a long bolt: well over the hatch
    notes["strokes"] = S.tighten(wb, "r", th, "bolt", yaw0=1.1, yaw1=0.25, max_strokes=16)
    S.release(wb, "r", up=0.05, swing=True, ease=False)
    wb.arm_joints_to("r", HIGH, 0.8)
    wb.settle()

    # 7. the drawer shut: push it back in by its knob
    _push(wb, "l", "drawer_knob", DRAWER_OPEN + 0.025)
    wb.arm_joints_to("l", READY, 1.0)
    wb.settle()

    # 8. power on
    wb.arm_joints_to("r", READY, 1.0)
    wb.settle()
    top = S.site(wb, "button_top")
    wb.look(top, dur=0.5)
    wb.fingers("r", "fist", dur=0.4)
    S.press(wb, "r", "r_thumb_tip", top)
    wb.fingers("r", "relaxed", dur=0.4)
    wb.arm_joints_to("r", READY, 1.0)
    wb.look([PANEL_X, PANEL_Y - PANEL[1] + 0.04, T + 0.12], dur=0.6)
    wb.wait(2.0)
    notes["final"] = dict(drawer=round(drawer_out(m, d), 4), hatch=round(hatch_open(m, d), 4),
                          cartridge_in=bool(cartridge_in(m, d)), bolt=round(th.depth(d), 4),
                          on=bool(d.userdata[_U_ON]))


def _pull(wb, side: str, knob: str, shift, yaw: float, dur: float = 1.6) -> None:
    """Grip a fixed knob (a drawer's, a hatch's) and move it by ``shift`` (it slides), then let go."""
    from pioneer_whole_body import skills as S
    top = S.site(wb, knob + "_top")
    p, r = S.knob_pose(top, yaw, None)
    wb.look(top, dur=0.4)
    wb.fingers(side, "lid_pre", dur=0.5)
    wb.reach(side, p + [0, 0, S.CLEAR], r, 1.2)
    wb.settle()
    wb.reach(side, p, r, 0.7)
    wb.settle()
    wb.fingers(side, "lid", amount=S.SQUEEZE, dur=0.45)
    wb.settle()
    wb.wait(0.15)
    wb.look(top + np.asarray(shift), dur=0.3)
    wb.reach(side, p + np.asarray(shift), r, dur)
    wb.settle()
    wb.wait(0.2)
    S.release(wb, side, up=0.06)


def _push(wb, side: str, knob: str, dy: float, dur: float = 1.4) -> None:
    """Push a slide shut (+y) with the edge of a fist against the knob's side (the palm itself, not the fingers)."""
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import palm_down
    top = S.site(wb, knob + "_top")
    r = palm_down(-0.2)
    start = top + [0, -0.05, -0.028]
    wb.look(top, dur=0.4)
    wb.fingers(side, "fist", dur=0.4)
    wb.reach(side, start + [0, 0, 0.06], r, 1.2)
    wb.settle()
    wb.reach(side, start, r, 0.6)
    wb.settle()
    wb.look(top + [0, dy, 0], dur=0.3)
    wb.reach(side, start + [0, dy, 0], r, dur)
    wb.settle()
    wb.wait(0.2)
    wb.reach(side, start + [0, dy, 0.08], r, 0.6)            # straight up off the knob
    wb.settle()
    wb.fingers(side, "relaxed", dur=0.4)


def _hands(wb, pose, dur: float = 1.0) -> None:
    for side in ("l", "r"):
        wb.arm_joints_to(side, pose, dur)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["service_panel"].plan = plan
