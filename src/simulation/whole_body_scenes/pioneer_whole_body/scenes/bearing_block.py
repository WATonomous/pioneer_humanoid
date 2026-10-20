"""Bearing block assembly (long horizon, whole body): bolt a bearing block onto a base plate and ship it.

The robot starts a step back from the workbench, raises its arms and walks up to it. On the bench:

  base plate   in a jig in the middle; a pocket for the block's flange, two threaded holes (y = +-50 mm)
  bearing block  on a tray to the left: a 70 x 160 mm flange with two clearance holes, a round boss on top
  knob bolts   one in a rack on each side (left hand's, right hand's)
  outbound tray  far left, a side step away

Steps, each checked from the simulation and latched in order (``progress``):
  1. set the bearing block on the base plate     flange in the pocket (within 3 mm), upright
  2. insert the left bolt                         its thread caught (``parts.Thread``)
  3. tighten the left bolt                        screwed home
  4. insert the right bolt
  5. tighten the right bolt
  6. ship the assembly in the outbound tray      plate in the tray, block and both bolts still on it

The robot looks at what it works on (its gaze leads the hands). Bolts turn on a quick-acting thread (``parts.PITCH``), about two thirds of a turn from catch to seated.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import READY
from pioneer_whole_body.parts import (DARK, KNOB, ORANGE, STEEL, Thread, add_bolt_weld, add_knob_bolt,
                                      add_thread_hole, bolt_length, quat_z)
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

# base plate in its jig
JIG_POS = (0.52, 0.0)
JIG_H = 0.03
PLATE = (0.09, 0.13, 0.0075)          # half sizes
PLATE_TOP = T + JIG_H + 2 * PLATE[2]
HOLE_Y = 0.08
TRAVEL = 0.009                         # bolt goes this deep into the plate
# bearing block
FLANGE = (0.035, 0.105, 0.006)          # half sizes
HOLE_HALF = 0.0095                     # square clearance holes in the flange
BOSS_R, BOSS_H = 0.021, 0.032
POCKET_CLEAR = 0.003
BLOCK_START = (0.60, 0.33)
TRAY_H = 0.02
# bolts
SHAFT = 2 * FLANGE[2] + TRAVEL - 0.0005     # the washer clamps the flange just before the thread bottoms
RACK_H = 0.04
RACKS = {"l": (0.48, 0.20), "r": (0.48, -0.20)}
# outbound tray (a side step to the left)
OUT_POS = (0.52, 0.60)
OUT_HALF = (0.12, 0.16)

STEPS = ("set the bearing block on the base plate", "insert the left bolt", "tighten the left bolt",
         "insert the right bolt", "tighten the right bolt", "release the jig's clamp",
         "ship the assembly in the outbound tray")
# the jig's clamp release: a mushroom button beside the jig, pressed with the palm
BUTTON_POS = (0.60, -0.30)
BUTTON_R = 0.02
BUTTON_TRAVEL = 0.008
THREADS = (Thread("hole_l", ("bolt_l", "bolt_r"), TRAVEL), Thread("hole_r", ("bolt_l", "bolt_r"), TRAVEL))
_U_STEP = 0


# ----------------------------------------------------------------------------- hooks
_BOUND: dict[int, list[Thread]] = {}


def threads(model) -> list[Thread]:
    key = id(model)
    if key not in _BOUND:
        _BOUND[key] = [Thread(t.hole, t.bolts, t.travel).bind(model) for t in THREADS]
    return _BOUND[key]


def step(model, data) -> None:
    ths = threads(model)
    for t in ths:
        t.step(data)
    busy = {t.engaged(data) for t in ths} - {None}
    for t in ths:
        if t.try_engage(model, data, busy):
            busy = {t.engaged(data) for t in ths} - {None}
    if data.qpos[model.joint("button").qposadr[0]] < -0.6 * BUTTON_TRAVEL:
        data.eq_active[model.equality("jig_clamp").id] = 0       # pressed: the clamp opens (and stays open)
    k = int(data.userdata[_U_STEP])
    if k < len(STEPS) and _step_done(model, data, k):
        data.userdata[_U_STEP] = k + 1


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def block_seated(model, data, tol=POCKET_CLEAR + 0.001, ztol=0.003) -> bool:
    b = data.xpos[model.body("block").id]
    p = data.xpos[model.body("plate").id]
    upright = data.xmat[model.body("block").id][8] > math.cos(math.radians(5))
    return (abs(b[0] - p[0]) < tol and abs(b[1] - p[1]) < tol and abs(b[2] - (p[2] + PLATE[2])) < ztol
            and upright)


def _step_done(model, data, k) -> bool:
    th = {t.hole: t for t in threads(model)}
    if k == 0:
        return block_seated(model, data)
    if k == 1:
        return th["hole_l"].engaged(data) is not None
    if k == 2:
        return th["hole_l"].seated(data)
    if k == 3:
        return th["hole_r"].engaged(data) is not None
    if k == 4:
        return th["hole_r"].seated(data)
    if k == 5:
        return not data.eq_active[model.equality("jig_clamp").id]
    p = data.xpos[model.body("plate").id]
    in_tray = (abs(p[0] - OUT_POS[0]) < OUT_HALF[0] - PLATE[0] + 0.01 and
               abs(p[1] - OUT_POS[1]) < OUT_HALF[1] - PLATE[1] + 0.01 and p[2] < T + 0.03)
    return in_tray and block_seated(model, data, tol=POCKET_CLEAR + 0.002) and all(t.seated(data) for t in th.values())


# ----------------------------------------------------------------------------- scene
@scene("bearing_block", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress)
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 1
    world = spec.worldbody
    blue, grey, alu = [0.18, 0.32, 0.58, 1], [0.32, 0.34, 0.37, 1], [0.78, 0.80, 0.83, 1]

    # jig: a riser with corner stops; the plate sits in it, free
    jx, jy = JIG_POS
    jig = world.add_body(name="jig", pos=[jx, jy, T])
    jig.add_geom(type=BOX, size=[PLATE[0] + 0.012, PLATE[1] + 0.012, JIG_H / 2], pos=[0, 0, JIG_H / 2], rgba=grey)
    for sx in (-1, 1):
        for sy in (-1, 1):
            jig.add_geom(type=BOX, size=[0.012, 0.012, PLATE[2] - 0.0015],
                         pos=[sx * (PLATE[0] + 0.0125), sy * (PLATE[1] + 0.0125), JIG_H + PLATE[2] - 0.0015], rgba=blue)
    plate = world.add_body(name="plate", pos=[jx, jy, T + JIG_H + PLATE[2]])
    plate.add_freejoint(name="plate")
    plate.add_geom(name="plate", type=BOX, size=list(PLATE), rgba=alu, mass=0.25, friction=[0.9, 0.01, 0.001])
    # pocket rim for the flange: 3 mm walls with a 45 degree lead-in on top (a funnel ~4 mm wide), so a block put
    # down a few mm or degrees off slides in
    px, py = FLANGE[0] + POCKET_CLEAR, FLANGE[1] + POCKET_CLEAR
    c45, s45 = math.cos(math.pi / 8), math.sin(math.pi / 8)
    for sx in (-1, 1):
        plate.add_geom(type=BOX, size=[0.004, py + 0.008, 0.0015], pos=[sx * (px + 0.004), 0, PLATE[2] + 0.0015],
                       rgba=DARK, mass=0.002)
        plate.add_geom(type=BOX, size=[px + 0.008, 0.004, 0.0015], pos=[0, sx * (py + 0.004), PLATE[2] + 0.0015],
                       rgba=DARK, mass=0.002)
        # lead-ins: square bars turned 45 degrees about their length, inner corner flush with the pocket wall
        h = 0.003
        plate.add_geom(type=BOX, size=[h, py, h], pos=[sx * (px + h * math.sqrt(2)), 0, PLATE[2] + 0.003],
                       quat=[c45, 0, s45, 0], rgba=DARK, mass=0.001, friction=[0.2, 0.005, 0.0001])
        plate.add_geom(type=BOX, size=[px, h, h], pos=[0, sx * (py + h * math.sqrt(2)), PLATE[2] + 0.003],
                       quat=[c45, s45, 0, 0], rgba=DARK, mass=0.001, friction=[0.2, 0.005, 0.0001])
    for name, y in (("hole_l", HOLE_Y), ("hole_r", -HOLE_Y)):
        add_thread_hole(spec, plate, name, [0, y, PLATE[2]], TRAVEL)

    # bearing block on its tray: flange with two clearance holes, a round boss (also its handle)
    bx, by = BLOCK_START
    tray = world.add_body(name="block_tray", pos=[bx, by, T])
    tray.add_geom(type=BOX, size=[0.05, 0.105, TRAY_H / 2], pos=[0, 0, TRAY_H / 2], rgba=grey)
    block = world.add_body(name="block", pos=[bx, by, T + TRAY_H])
    block.add_freejoint(name="block")
    fx, fy, fz = FLANGE
    h = HOLE_HALF
    orange = ORANGE
    for y0, y1 in ((-fy, -HOLE_Y - h), (-HOLE_Y + h, HOLE_Y - h), (HOLE_Y + h, fy)):
        block.add_geom(type=BOX, size=[fx, (y1 - y0) / 2, fz], pos=[0, (y0 + y1) / 2, fz], rgba=orange, mass=0.03)
    for yc in (-HOLE_Y, HOLE_Y):
        for sx in (-1, 1):
            block.add_geom(type=BOX, size=[(fx - h) / 2, h, fz], pos=[sx * (h + (fx - h) / 2), yc, fz], rgba=orange,
                           mass=0.005)
    block.add_geom(name="boss", type=CYL, size=[BOSS_R, BOSS_H / 2, 0], pos=[0, 0, 2 * fz + BOSS_H / 2],
                   rgba=orange, mass=0.05, friction=[1.4, 0.03, 0.003], condim=4)
    block.add_geom(type=CYL, size=[0.011, 0.0008, 0], pos=[0, 0, 2 * fz + BOSS_H + 0.0008], rgba=STEEL,
                   contype=0, conaffinity=0, mass=0.0001)            # the bearing's bore (visual)
    block.add_site(name="block_top", pos=[0, 0, 2 * fz + BOSS_H])
    block.add_site(name="block_base", pos=[0, 0, 0])

    # bolt racks on risers
    for side, (rx, ry) in RACKS.items():
        rack = world.add_body(name=f"rack_{side}", pos=[rx, ry, T])
        rack.add_geom(type=BOX, size=[0.03, 0.03, RACK_H / 2], pos=[0, 0, RACK_H / 2], rgba=grey)
        for dx, dy, sx, sy in ((0.009, 0, 0.004, 0.013), (-0.009, 0, 0.004, 0.013), (0, 0.009, 0.005, 0.004),
                               (0, -0.009, 0.005, 0.004)):
            rack.add_geom(type=BOX, size=[sx, sy, 0.004], pos=[dx, dy, RACK_H + 0.004], rgba=blue)
        add_knob_bolt(world, f"bolt_{side}", [rx, ry, T + RACK_H + 0.0005], yaw=0.3 if side == "l" else -0.3,
                      shaft=SHAFT)

    # the clamp's release button: a grey housing, a red cap on a spring
    bx_, by_ = BUTTON_POS
    housing = world.add_body(name="button_housing", pos=[bx_, by_, T])
    housing.add_geom(type=BOX, size=[0.03, 0.03, 0.02], pos=[0, 0, 0.02], rgba=grey)
    housing.add_geom(type=CYL, size=[BUTTON_R + 0.004, 0.003, 0], pos=[0, 0, 0.043], rgba=[0.95, 0.8, 0.1, 1])
    cap = housing.add_body(name="button", pos=[0, 0, 0.046])
    cap.add_joint(name="button", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-BUTTON_TRAVEL, 0],
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=150.0, springref=0.0, damping=3.0)
    cap.add_geom(type=CYL, size=[BUTTON_R, 0.006, 0], pos=[0, 0, 0.006], rgba=[0.85, 0.1, 0.08, 1], mass=0.02)
    cap.add_site(name="button_top", pos=[0, 0, 0.012])
    spec.add_exclude(bodyname1="button_housing", bodyname2="button")

    # outbound tray: a shallow box
    ox, oy = OUT_POS
    out = world.add_body(name="out_tray", pos=[ox, oy, T])
    hx, hy = OUT_HALF
    green = [0.22, 0.48, 0.32, 1]
    out.add_geom(type=BOX, size=[hx, hy, 0.003], pos=[0, 0, 0.003], rgba=green)
    for sx in (-1, 1):
        out.add_geom(type=BOX, size=[0.004, hy, 0.012], pos=[sx * hx, 0, 0.012], rgba=green)
        out.add_geom(type=BOX, size=[hx, 0.004, 0.012], pos=[0, sx * hy, 0.012], rgba=green)

    for t in THREADS:
        for b in t.bolts:
            add_bolt_weld(spec, t.hole, b)
    for b in ("bolt_l", "bolt_r"):
        spec.add_exclude(bodyname1="plate", bodyname2=b)
    # the jig's clamp: holds the plate while it's worked on (a toggle clamp, released with its lever)
    clamp = spec.add_equality(name="jig_clamp", type=mujoco.mjtEq.mjEQ_WELD, name1="jig", name2="plate",
                              objtype=mujoco.mjtObj.mjOBJ_BODY, active=True)
    clamp.solref = [0.004, 1.0]


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    """The scripted demo (a skilled operator's plan). Raises skills.TaskFailed if any step fails."""
    from pioneer_whole_body import skills as S

    m, d = wb.m, wb.d
    th = {t.hole: t for t in threads(m)}
    wb.look([0.5, 0.0, T], dur=0.8)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    wb.walk(0.30)
    wb.look(S.site(wb, "block_top"), dur=0.6)
    wb.wait(0.4)

    # 1. the block: left hand by its boss, onto the plate's pocket (put down, let go; lift and try again if it
    # didn't drop in)
    for attempt in range(3):
        if attempt == 0:
            S.pick_knob(wb, "l", "block_top", "block", yaw=-0.45)
        else:
            S.pick_knob(wb, "l", "block_top", "block", yaw=-0.3 - 0.1 * attempt, lift=0.04)
        seat = d.xpos[m.body("plate").id] + [0, 0, PLATE[2]]
        S.carry(wb, "l", "block_base", seat, clear=0.02, dur=1.8 if attempt == 0 else 1.0)
        S.align_held(wb, "l", "block", "block_base", seat, yaw=S.body_yaw(wb, "plate"), sym=math.pi, rounds=4)
        S.lower_until(wb, "l", 0.035, lambda: block_seated(m, d, ztol=0.0012), dur=1.2)
        S.release(wb, "l")
        wb.wait(0.3)
        if block_seated(m, d):
            break
    else:
        raise S.TaskFailed("block not seated")

    # 2-3. left bolt (the pocket keeps the block from turning with it)
    S.pick_knob(wb, "l", "bolt_l_top", "bolt_l", yaw=-0.8)
    S.insert_bolt(wb, "l", th["hole_l"], "bolt_l")
    notes["strokes_l"] = S.tighten(wb, "l", th["hole_l"], "bolt_l", yaw0=-0.3, yaw1=-1.05, max_strokes=16)
    S.release(wb, "l", up=0.05)
    wb.arm_joints_to("l", READY, 1.0)

    # 4-5. right bolt
    S.pick_knob(wb, "r", "bolt_r_top", "bolt_r", yaw=1.0)
    S.insert_bolt(wb, "r", th["hole_r"], "bolt_r")
    notes["strokes_r"] = S.tighten(wb, "r", th["hole_r"], "bolt_r", yaw0=1.1, yaw1=0.25, max_strokes=16)
    S.release(wb, "r", up=0.05)
    wb.arm_joints_to("r", READY, 1.0)

    # 6. release the clamp: thumb on the button
    top = S.site(wb, "button_top")
    wb.look(top, dur=0.5)
    from pioneer_whole_body.controller import palm_down
    wb.fingers("r", "fist", dur=0.4)
    S.press(wb, "r", "r_thumb_tip", top)
    if d.eq_active[m.equality("jig_clamp").id]:
        raise S.TaskFailed("clamp not released")
    wb.fingers("r", "relaxed", dur=0.4)
    wb.arm_joints_to("r", READY, 1.0)

    # 7. ship: both hands take the bolt knobs and lift the assembly, side-step left with it, set it in the tray
    wb.look(S.site(wb, "block_top"), dur=0.5)
    S.pick_two(wb, {"l": ("bolt_l_top", -0.6), "r": ("bolt_r_top", 0.7)}, "plate", lift=0.06, shape="lid",
               above=S.GRASP_ABOVE, squeeze=1.25)
    wb.look([OUT_POS[0], OUT_POS[1], T], dur=0.6)
    for _ in range(3):
        wb.step("l", 0.0, 0.15, carry=True)
        wb.step("r", 0.0, 0.15, carry=True)
    dest = np.array([OUT_POS[0], OUT_POS[1], T + 0.006])          # the plate's underside on the tray floor
    for dur in (1.4, 0.5, 0.4):                                   # over the tray, look, correct
        base = d.xpos[m.body("plate").id] - [0, 0, PLATE[2]]
        S.move_two(wb, [dest[0] - base[0], dest[1] - base[1], 0.03 + dest[2] - base[2]], dur=dur)
        wb.wait(0.15)
    for side in ("l", "r"):
        p, r = wb.hand_goal[side]
        wb.reach(side, p - [0, 0, 0.05], r, 1.0)
    wb.until(lambda: _touching(m, d, "plate", "out_tray"), 1.3)
    wb.wait(0.2)
    from pioneer_whole_body.robot import hand_joints
    for side in ("l", "r"):                     # fingers off the knobs, thumbs left where they are, lift off
        keep = {j[2:]: wb.cmd[j] for j in hand_joints(side) if j[2:].startswith("th_")}
        wb.fingers(side, {**{f"{f}{j}": 0.0 for f in ("", "i_", "r_", "l_") for j in ("mcp", "pip")}, **keep},
                   dur=0.5)
    wb.settle()
    for side in ("l", "r"):
        p, r = wb.hand_goal[side]
        wb.reach(side, p + [0, 0, 0.08], r, 0.9)
    wb.settle()
    for side in ("l", "r"):
        wb.fingers(side, "relaxed", dur=0.5)
    for side in ("l", "r"):
        wb.arm_joints_to(side, READY, 1.0)
    wb.look([OUT_POS[0], OUT_POS[1], T + 0.04], dur=0.5)
    wb.wait(1.5)
    p = d.xpos[m.body("plate").id]
    notes["final"] = dict(plate=np.round(p, 4).tolist(), block_seated=bool(block_seated(m, d, tol=POCKET_CLEAR + 0.002)),
                          threads=[(t.hole, t.engaged(d), round(t.depth(d), 4)) for t in threads(m)])


def _touching(m, d, a: str, b: str) -> bool:
    ia, ib = m.body(a).id, m.body(b).id
    return any({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {ia, ib} for c in d.contact[:d.ncon])


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["bearing_block"].plan = plan
_REGISTRY["bearing_block"].title = "Pioneer v2  |  bearing block assembly: place, bolt down (2 knob bolts), ship"
