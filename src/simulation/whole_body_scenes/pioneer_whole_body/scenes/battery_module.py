"""Battery module assembly (long horizon, whole body, two-handed): load four cells, close the lid, screw it down
with both hands at once, test it.

The robot walks up to the bench. On it:

  holder       in front: four cell sockets in a row and two pillars with threaded holes (y = +-0.21)
  cell rack    a side step to the right: four cells standing in it
  lid          a side step to the left, on a stand: two captive knob screws in it (they stay in the lid)
  tester       past the cell rack (another side step): its button, and an indicator that flips up when the module
               tests good

Steps, each checked from the simulation and latched in order (``progress``):
  1. load the first two cells      both hands fetch a cell each from the rack (side-steps there and back)
  2. load the last two cells
  3. put the lid on                 carried by its two screws, set on the pillars; the screws catch their threads
  4. screw the lid down             both screws home, turned by both hands at once
  5. test the module                the test button pressed with all four cells in and the lid screwed down
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY, palm_down
from pioneer_whole_body.parts import (DARK, ORANGE, STEEL, Thread, add_bolt_weld, add_knob_bolt, add_thread_hole,
                                      add_weld, quat_z)
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

HOLDER_X = 0.52
SOCKETS_Y = (0.135, 0.045, -0.045, -0.135)          # left to right
SOCKET_R = 0.025                                   # inner radius (cells are 19 mm: 6 mm of play)
SOCKET_H = 0.02
CELL_R, CELL_H = 0.019, 0.062
PILLAR_Y = 0.26
PILLAR_H = 0.068                                   # above the cells' tops (62 mm)
LID = (0.045, 0.30, 0.004)                         # half sizes
TRAVEL = 0.009
SHAFT = 2 * LID[2] + TRAVEL - 0.0005
STEP = 0.15                                        # side-step length
SIDE = 3 * STEP                                    # the rack and the lid stand are this far to the sides
RACK = [(0.44, -SIDE + 0.09), (0.44, -SIDE - 0.09), (0.56, -SIDE + 0.09), (0.56, -SIDE - 0.09)]
LID_SIDE = 4 * STEP                                # the lid stand is further: the lid is long
LID_STAND = (HOLDER_X, LID_SIDE)
TEST_SIDE = 4 * STEP                               # the tester: past the cell rack
BUTTON_POS = (0.46, -TEST_SIDE - 0.10)
BUTTON_TRAVEL = 0.008

STEPS = ("load the first two cells", "load the last two cells", "put the lid on", "screw the lid down",
         "test the module")
THREADS = (Thread("thread_l", ("screw_l", "screw_r"), TRAVEL), Thread("thread_r", ("screw_l", "screw_r"), TRAVEL))
_U_STEP, _U_TESTED = 0, 1
_BOUND: dict = {}


def threads(model):
    key = id(model)
    if key not in _BOUND:
        _BOUND[key] = [Thread(t.hole, t.bolts, t.travel).bind(model) for t in THREADS]
    return _BOUND[key]


def cells_in(model, data) -> int:
    n = 0
    for k in range(4):
        c = data.xpos[model.body(f"cell{k}").id]
        upright = data.xmat[model.body(f"cell{k}").id][8] > math.cos(math.radians(12))
        if any(math.hypot(c[0] - HOLDER_X, c[1] - y) < SOCKET_R and c[2] < T + 0.004 for y in SOCKETS_Y) and upright:
            n += 1
    return n


def lid_on(model, data) -> bool:
    p = data.xpos[model.body("lid").id]
    return abs(p[0] - HOLDER_X) < 0.006 and abs(p[1]) < 0.006 and abs(p[2] - (T + PILLAR_H + LID[2])) < 0.003


def step(model, data) -> None:
    ths = threads(model)
    for t in ths:
        t.step(data)
    busy = {t.engaged(data) for t in ths} - {None}
    for t in ths:
        b = t.try_engage(model, data, busy)
        if b:                                       # on its thread: no longer held captive in the lid
            data.eq_active[model.equality(f"captive_{b}").id] = 0
            busy.add(b)
    if data.qpos[model.joint("button").qposadr[0]] < -0.6 * BUTTON_TRAVEL:
        if cells_in(model, data) == 4 and all(t.seated(data) for t in ths):
            data.userdata[_U_TESTED] = 1
    # the indicator flag rises once the module tested good
    a = model.joint("indicator").qposadr[0]
    target = 0.0 if data.userdata[_U_TESTED] else -math.pi / 2
    data.qpos[a] += float(np.clip(target - data.qpos[a], -0.004, 0.004))
    data.qvel[model.joint("indicator").dofadr[0]] = 0.0
    k = int(data.userdata[_U_STEP])
    if k < len(STEPS) and _step_done(model, data, k):
        data.userdata[_U_STEP] = k + 1


def _step_done(model, data, k) -> bool:
    if k == 0:
        return cells_in(model, data) >= 2
    if k == 1:
        return cells_in(model, data) == 4
    if k == 2:
        return lid_on(model, data) and all(t.engaged(data) for t in threads(model))
    if k == 3:
        return all(t.seated(data) for t in threads(model))
    return bool(data.userdata[_U_TESTED])


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _ring(body, name, r, h, wall, n, rgba, z0=0.0):
    half_t = (r + wall) * math.tan(math.pi / n) + 0.0005
    for i in range(n):
        a = 2 * math.pi * i / n
        body.add_geom(name=f"{name}_{i}", type=BOX, size=[wall / 2, half_t, h / 2],
                      pos=[(r + wall / 2) * math.cos(a), (r + wall / 2) * math.sin(a), z0 + h / 2],
                      quat=quat_z(a), rgba=rgba)


@scene("battery_module", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress,
       title="Pioneer v2  |  battery module: load 4 cells, lid on, screw down with both hands, test")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec, half_y=0.98)
    spec.nuserdata = 2
    world = spec.worldbody
    grey, dark_blue = [0.32, 0.34, 0.37, 1], [0.16, 0.22, 0.36, 1]

    # holder: a base plate, four socket rings, two pillars with the threads on top
    holder = world.add_body(name="holder", pos=[HOLDER_X, 0, T])
    holder.add_geom(type=BOX, size=[0.05, 0.31, 0.002], pos=[0, 0, -0.0015], rgba=dark_blue, contype=0,
                    conaffinity=0)
    for k, y in enumerate(SOCKETS_Y):
        sock = holder.add_body(name=f"socket{k}", pos=[0, y, 0])
        _ring(sock, f"socket{k}", SOCKET_R, SOCKET_H, 0.004, 12, dark_blue)
    for name, y in (("thread_l", PILLAR_Y), ("thread_r", -PILLAR_Y)):
        holder.add_geom(type=BOX, size=[0.016, 0.016, PILLAR_H / 2], pos=[0, y, PILLAR_H / 2], rgba=grey)
        add_thread_hole(spec, holder, name, [0, y, PILLAR_H], TRAVEL)

    # the cells, standing in their rack (orange caps)
    rack = world.add_body(name="cell_rack", pos=[0.5, -SIDE, T])
    rack.add_geom(type=BOX, size=[0.1, 0.16, 0.002], pos=[0, 0, -0.0015], rgba=grey, contype=0, conaffinity=0)
    for k, (x, y) in enumerate(RACK):
        r = rack.add_body(pos=[x - 0.5, y + SIDE, 0])
        _ring(r, f"rack{k}", SOCKET_R - 0.002, 0.012, 0.004, 12, grey)
        c = world.add_body(name=f"cell{k}", pos=[x, y, T + 0.0005])
        c.add_freejoint(name=f"cell{k}")
        c.add_geom(type=CYL, size=[CELL_R, CELL_H / 2 - 0.004, 0], pos=[0, 0, CELL_H / 2], rgba=[0.14, 0.4, 0.75, 1],
                   mass=0.045, friction=[1.6, 0.03, 0.003], condim=4)
        c.add_geom(type=CYL, size=[CELL_R * 0.98, 0.002, 0], pos=[0, 0, CELL_H - 0.002], rgba=ORANGE, mass=0.002,
                   friction=[1.6, 0.03, 0.003], condim=4)
        c.add_geom(type=CYL, size=[CELL_R * 0.98, 0.002, 0], pos=[0, 0, 0.002], rgba=STEEL, mass=0.002)
        c.add_geom(type=CYL, size=[0.006, 0.0015, 0], pos=[0, 0, CELL_H + 0.0015], rgba=STEEL, mass=0.0005,
                   contype=0, conaffinity=0)
        c.add_site(name=f"cell{k}_top", pos=[0, 0, CELL_H])
        c.add_site(name=f"cell{k}_base", pos=[0, 0, 0])

    # the lid on its stand, with its two captive screws
    lx, ly = LID_STAND
    stand = world.add_body(name="lid_stand", pos=[lx, ly, T])
    for y in (-PILLAR_Y, PILLAR_Y):
        stand.add_geom(type=BOX, size=[0.016, 0.016, PILLAR_H / 2], pos=[0, y, PILLAR_H / 2], rgba=grey)
    lid = world.add_body(name="lid", pos=[lx, ly, T + PILLAR_H + LID[2]])
    lid.add_freejoint(name="lid")
    hx, hy, hz = LID
    hole = 0.0075
    lid_rgba = [0.88, 0.88, 0.9, 1]
    for y0, y1 in ((-hy, -PILLAR_Y - hole), (-PILLAR_Y + hole, PILLAR_Y - hole), (PILLAR_Y + hole, hy)):
        lid.add_geom(type=BOX, size=[hx, (y1 - y0) / 2, hz], pos=[0, (y0 + y1) / 2, 0], rgba=lid_rgba, mass=0.04)
    for yc in (-PILLAR_Y, PILLAR_Y):
        for sx in (-1, 1):
            lid.add_geom(type=BOX, size=[(hx - hole) / 2, hole, hz], pos=[sx * (hole + (hx - hole) / 2), yc, 0],
                         rgba=lid_rgba, mass=0.005)
    lid.add_geom(type=BOX, size=[hx * 0.8, 0.16, 0.0006], pos=[0, 0, hz + 0.0006], rgba=ORANGE, contype=0,
                 conaffinity=0, mass=0.0001)                                   # label
    lid.add_site(name="lid_base", pos=[0, 0, -hz])
    for side, y in (("l", PILLAR_Y), ("r", -PILLAR_Y)):
        add_knob_bolt(world, f"screw_{side}", [lx, ly + y, T + PILLAR_H], yaw=0.0, shaft=SHAFT)
        add_weld(spec, f"captive_screw_{side}", "lid", f"screw_{side}")
        spec.add_exclude(bodyname1="holder", bodyname2=f"screw_{side}")
        spec.add_exclude(bodyname1="lid_stand", bodyname2=f"screw_{side}")
    for t in THREADS:
        for b in t.bolts:
            add_bolt_weld(spec, t.hole, b)

    # test button and the indicator flag beside it
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
    post = world.add_body(name="indicator_post", pos=[bx + 0.06, by - 0.07, T])
    post.add_geom(type=CYL, size=[0.006, 0.05, 0], pos=[0, 0, 0.05], rgba=grey, contype=0, conaffinity=0)
    flag = post.add_body(name="indicator", pos=[0, 0, 0.1])
    flag.add_joint(name="indicator", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], ref=0.0)
    flag.add_geom(type=BOX, size=[0.003, 0.02, 0.03], pos=[0, 0, 0.03], rgba=[0.15, 0.85, 0.3, 1], contype=0,
                  conaffinity=0, mass=0.01)
    flag.gravcomp = 1.0


def reset_flag(model, data):
    data.qpos[model.joint("indicator").qposadr[0]] = -math.pi / 2


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S

    m, d = wb.m, wb.d
    th = {t.hole: t for t in threads(m)}
    reset_flag(m, d)
    wb.look([HOLDER_X, 0.0, T], dur=0.8)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _hands(wb, HIGH, 0.8)                       # hands up for walking along the bench (it's crowded)
    wb.walk(0.30)
    wb.look([HOLDER_X, -0.2, T + 0.05], dur=0.8)
    wb.wait(0.5)

    sockets = {k: np.array([HOLDER_X, y, T]) for k, y in enumerate(SOCKETS_Y)}
    for pair, (cl, cr, sl, sr) in enumerate(((0, 1, 0, 3), (2, 3, 1, 2))):
        # to the rack: a side step right, a cell in each hand, and back
        wb.look([0.5, -SIDE, T + 0.05], dur=0.6)
        _hands(wb, HIGH)
        _side_step(wb, -SIDE)
        S.pick_knob(wb, "l", f"cell{cl}_top", f"cell{cl}", yaw=-0.9, lift=0.12)
        S.pick_knob(wb, "r", f"cell{cr}_top", f"cell{cr}", yaw=0.9, lift=0.12)
        wb.look([HOLDER_X, 0.0, T + 0.05], dur=0.6)
        _side_step(wb, SIDE, carry=True)
        # into the sockets, one hand after the other
        for side, cell, sock in (("l", cl, sl), ("r", cr, sr)):
            S.carry(wb, side, f"cell{cell}_base", sockets[sock], clear=0.03, dur=1.2)
            S.align_held(wb, side, f"cell{cell}", f"cell{cell}_base", sockets[sock], yaw=None, rounds=2)
            S.lower_until(wb, side, 0.045, lambda c=cell: d.xpos[m.body(f"cell{c}").id][2] < T + 0.003, dur=1.0)
            S.release(wb, side, up=0.08)
            wb.arm_joints_to(side, HIGH, 0.8)
        wb.settle()
        if cells_in(m, d) < 2 * (pair + 1):
            raise S.TaskFailed(f"only {cells_in(m, d)} cells in")

    # the lid: side step left, take it by its two screws, back, set it on the pillars
    wb.look([LID_STAND[0], LID_STAND[1], T + 0.08], dur=0.6)
    _hands(wb, HIGH)
    _side_step(wb, LID_SIDE)
    S.pick_two(wb, {"l": ("screw_l_top", -0.7), "r": ("screw_r_top", 0.7)}, "lid", lift=0.05, shape="lid",
               above=S.GRASP_ABOVE, squeeze=1.25)
    wb.look([HOLDER_X, 0.0, T + 0.08], dur=0.6)
    _side_step(wb, -LID_SIDE, carry=True)
    seat = np.array([HOLDER_X, 0.0, T + PILLAR_H])
    base = S.site(wb, "lid_base")
    S.move_two(wb, [seat[0] - base[0], seat[1] - base[1], 0.025 + seat[2] - base[2]], dur=1.2)
    for attempt in range(3):
        S.align_two(wb, "lid", "lid_base", seat, yaw=0.0, rounds=3)
        for side in ("l", "r"):
            p, r = wb.hand_goal[side]
            wb.reach(side, p - [0, 0, 0.04], r, 1.0)
        wb.until(lambda: all(t.engaged(d) for t in th.values()), 1.3)
        wb.wait(0.2)
        if all(t.engaged(d) for t in th.values()):
            break
        S.move_two(wb, [0, 0, 0.03], dur=0.6)          # didn't catch: lift a little, line up again
    else:
        raise S.TaskFailed("lid screws did not catch")

    # both screws home at once
    notes["strokes"] = S.tighten_two(wb, [("l", th["thread_l"], "screw_l", -0.3, -1.05),
                                          ("r", th["thread_r"], "screw_r", 1.1, 0.25)], max_strokes=16)
    S.release(wb, ("l", "r"), up=0.06)
    _hands(wb, HIGH)

    # test: over to the tester, press its button
    wb.look(S.site(wb, "button_top"), dur=0.6)
    _side_step(wb, -TEST_SIDE)
    top = S.site(wb, "button_top")
    wb.look(top, dur=0.5)
    wb.fingers("r", "fist", dur=0.4)
    S.press(wb, "r", "r_thumb_tip", top)
    wb.fingers("r", "relaxed", dur=0.4)
    _hands(wb, READY)
    wb.look(S.site(wb, "button_top") + [0, 0.06, 0.12], dur=0.6)
    wb.wait(2.0)
    notes["final"] = dict(cells=cells_in(m, d), lid_on=bool(lid_on(m, d)),
                          threads=[(t.hole, t.engaged(d), round(t.depth(d), 4)) for t in th.values()])


def _hands(wb, pose, dur: float = 1.0) -> None:
    for side in ("l", "r"):
        wb.arm_joints_to(side, pose, dur)
    wb.settle()


def _side_step(wb, dy: float, carry: bool = False) -> None:
    """Side-step ``dy`` (m, + is left) in steps of STEP, leading with the foot on that side."""
    lead, trail = ("l", "r") if dy > 0 else ("r", "l")
    n = max(1, round(abs(dy) / STEP))
    for _ in range(n):
        wb.step(lead, 0.0, math.copysign(STEP, dy), carry=carry)
        wb.step(trail, 0.0, math.copysign(STEP, dy), carry=carry)


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["battery_module"].plan = plan
