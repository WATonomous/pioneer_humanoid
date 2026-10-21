"""A drawer (an articulated object): the robot pulls a drawer open by its handle, looks into it, takes out the two
parts in it and puts each on the pad of its colour, then pushes the drawer shut.

The robot walks up to the bench. On it:

  cabinet  a low cabinet in the middle, its drawer shut (a slide with friction: it stays where it is left); the
           drawer's handle a knob on a bracket in front of its face
  parts    two knob-topped parts in the drawer, one red-topped and one blue-topped; which is on which side, and
           where in the drawer, changes with every seed (``reset``): the robot can't see them till it opens it
  pads     a red pad on the bench left of the cabinet, a blue one right of it

The plan: take the handle with the right hand, pull the drawer open (small straight moves along the slide) as far as
the arm reaches, then step back holding it to open it the rest of the way (enough to reach in clear of the cabinet's
roof), let go, step in again.
Look into the drawer: the coloured tops found in the head camera's image (``vision.Eye``), put on the plane of the
knob tops. Then each part to the pad of its colour, taken by the hand on its side (the one further left first, so an arm
never reaches over one still to be taken); a part whose pad is on the other side is carried there with a side step
along the bench, and the robot steps back. Last, a step back, take the handle again,
push the drawer shut with the arm and a step in.

Steps (``progress``): 1. open the drawer, 2. take the parts out (n of 2 on their pads), 3. shut it.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import READY
from pioneer_whole_body.parts import KNOB, KNOB_H, KNOB_R, STEEL, quat_z
from pioneer_whole_body.scenes import sort_bins as SB
from pioneer_whole_body.scenes import weigh_sort as WS
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

FRONT_X = 0.65                    # the drawer's face (shut)
DEPTH = 0.24                      # the drawer, front to back
HALF_W = 0.12                     # half its outside width
FLOOR_Z = 0.008                   # the drawer's floor above the bench
WALL_H = 0.035
FACE_H = 0.05                     # the face's top above the bench (low: the arms reach in over it)
ROOF_Z = 0.09                     # the cabinet's roof (underside) above the bench
OPEN = 0.19                       # how far it's pulled open: the arm pulls it ARM, a step back the rest
ARM = 0.085
BACK = OPEN - ARM + 0.003
HANDLE = (-0.05, 0.0)             # the handle knob from the face's top centre (x out in front, y)
PART_X = 0.075                    # the parts from the face (inside the drawer)
PART_Y = 0.05
PADS = {"red": (0.50, SB.HAND_Y), "blue": (0.50, -SB.HAND_Y)}
PAD = 0.05                        # half the pad's side
PAD_H = 0.004
STEPS = ("open the drawer", "take the parts out", "shut it")
_U_STEP = 0


def _slide(model, data) -> float:
    return float(data.qpos[model.joint("drawer").qposadr[0]])


def _on_pads(model, data) -> int:
    n = 0
    for c, (x, y) in PADS.items():
        p = data.xpos[model.body(f"part_{c}").id]
        up = data.xmat[model.body(f"part_{c}").id][8]
        if abs(p[0] - x) < PAD and abs(p[1] - y) < PAD and p[2] < T + PAD_H + 0.01 and up > 0.95:
            n += 1
    return n


def step(model, data) -> None:
    k = int(data.userdata[_U_STEP])
    if k == 0 and _slide(model, data) > OPEN - 0.015:
        data.userdata[_U_STEP] = 1
    elif k == 1 and _on_pads(model, data) == 2:
        data.userdata[_U_STEP] = 2
    elif k == 2 and _slide(model, data) < 0.005:
        data.userdata[_U_STEP] = 3


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 0:
        return 0, len(STEPS), f"open the drawer ({100 * _slide(model, data):.0f} of {100 * OPEN:.0f} cm)"
    if k == 1:
        return 1, len(STEPS), f"take the parts out ({_on_pads(model, data)} of 2 on their pads)"
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def reset(model, data, rng) -> None:
    """Which part is on which side of the drawer, and a little way off its spot (turned any way)."""
    sides = (PART_Y, -PART_Y) if rng.random() < 0.5 else (-PART_Y, PART_Y)
    for c, y in zip(("red", "blue"), sides):
        a = model.joint(f"part_{c}").qposadr[0]
        x = FRONT_X + PART_X + rng.uniform(-0.01, 0.01)
        data.qpos[a:a + 3] = [x, y + rng.uniform(-0.012, 0.012), T + FLOOR_Z + 0.0005]
        yaw = rng.uniform(0, 2 * math.pi)
        data.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


@scene("drawer", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  a drawer: pulled open, the parts taken out by sight, pushed shut")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 1
    world = spec.worldbody
    wood, dark, grey = [0.55, 0.42, 0.30, 1], [0.16, 0.18, 0.22, 1], [0.32, 0.34, 0.37, 1]
    body = [0.80, 0.80, 0.78, 1]
    # the cabinet: roof, sides, back (the drawer slides in under the roof)
    cab = world.add_body(name="cabinet", pos=[FRONT_X, 0, T])
    x0, x1 = 0.012, DEPTH + 0.02
    cab.add_geom(type=BOX, size=[(x1 - x0) / 2, HALF_W + 0.012, 0.005], pos=[(x0 + x1) / 2, 0, ROOF_Z + 0.005],
                 rgba=body)
    for s in (1, -1):
        cab.add_geom(type=BOX, size=[(x1 - x0) / 2, 0.006, ROOF_Z / 2], pos=[(x0 + x1) / 2, s * (HALF_W + 0.006),
                     ROOF_Z / 2], rgba=body)
    cab.add_geom(type=BOX, size=[0.006, HALF_W, ROOF_Z / 2], pos=[x1 - 0.006, 0, ROOF_Z / 2], rgba=body)
    # the drawer (its face at the body's origin, the inside behind it)
    dr = world.add_body(name="drawer", pos=[FRONT_X, 0, T])
    dr.add_joint(name="drawer", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, OPEN + 0.002],
                 limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=15.0, frictionloss=2.0, armature=0.05)
    dr.add_geom(type=BOX, size=[0.006, HALF_W + 0.012, FACE_H / 2], pos=[0.006, 0, FACE_H / 2 + 0.001], rgba=grey,
                mass=0.15)
    dr.add_geom(type=BOX, size=[DEPTH / 2, HALF_W - 0.002, 0.004], pos=[0.012 + DEPTH / 2, 0, FLOOR_Z - 0.004],
                rgba=wood, mass=0.1, friction=[0.9, 0.02, 0.002])
    for s in (1, -1):
        dr.add_geom(type=BOX, size=[DEPTH / 2, 0.004, WALL_H / 2], pos=[0.012 + DEPTH / 2, s * (HALF_W - 0.006),
                    FLOOR_Z + WALL_H / 2], rgba=wood, mass=0.03)
    dr.add_geom(type=BOX, size=[0.004, HALF_W - 0.002, WALL_H / 2], pos=[0.012 + DEPTH - 0.004, 0,
                FLOOR_Z + WALL_H / 2], rgba=wood, mass=0.03)
    # the handle: a bracket out in front of the face's top, a knob on it (on a boss as wide as the knob: no undercut
    # for the thumb to hook under)
    hx, hy = HANDLE
    dr.add_geom(type=BOX, size=[-hx / 2 + 0.01, 0.012, 0.004], pos=[hx / 2, hy, FACE_H - 0.004], rgba=dark,
                mass=0.01)
    z = FACE_H
    dr.add_geom(type=CYL, size=[KNOB_R, 0.006, 0], pos=[hx, hy, z + 0.006], rgba=STEEL, mass=0.004)
    z += 0.012
    dr.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[hx, hy, z + KNOB_H / 2], rgba=KNOB, mass=0.015,
                friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):
        a = 2 * math.pi * i / 10
        dr.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                    pos=[hx + KNOB_R * math.cos(a), hy + KNOB_R * math.sin(a), z + KNOB_H / 2], quat=quat_z(a),
                    rgba=KNOB, contype=0, conaffinity=0, mass=0.0001)
    dr.add_site(name="handle_top", pos=[hx, hy, z + KNOB_H + 0.0012])
    spec.add_exclude(bodyname1="cabinet", bodyname2="drawer")
    # the parts (in the drawer; ``reset`` puts them where they are)
    for c, y in (("red", PART_Y), ("blue", -PART_Y)):
        SB._part(world, f"part_{c}", (FRONT_X + PART_X, y, T + FLOOR_Z + 0.0005), c)
    # the pads
    for c, (x, y) in PADS.items():
        pad = world.add_body(name=f"pad_{c}", pos=[x, y, T])
        pad.add_geom(type=BOX, size=[PAD, PAD, PAD_H / 2], pos=[0, 0, PAD_H / 2], rgba=SB.COLOURS[c])


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    eye = Eye(m)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    wb.walk(0.30)
    # 1. pull it open: the arm pulls it as far as it reaches, then a step back with it
    _grip(wb)
    _move(wb, -ARM)
    wb.walk(-BACK, carry=True)
    _let_go(wb)
    wb.walk(BACK)
    notes["opened"] = round(_slide(m, d), 3)
    if _slide(m, d) < OPEN - 0.015:
        raise S.TaskFailed(f"the drawer's open {_slide(m, d):.3f} m")
    # 2. look in, take the parts out
    plane = T + FLOOR_Z + SB.TOP_Z
    inside = np.array([FRONT_X - _slide(m, d) + PART_X, 0.0, plane])
    wb.look(inside, dur=0.7)
    wb.wait(0.35)
    seen = {}
    for b in eye.find(d, plane):
        if b.colour in PADS and abs(b.world[0] - inside[0]) < 0.06 and abs(b.world[1]) < HALF_W:
            if b.colour not in seen or b.area > seen[b.colour].area:
                seen[b.colour] = b
    notes["seen"] = {c: [round(float(v), 4) for v in b.world[:2]] for c, b in seen.items()}
    if len(seen) < 2:
        raise S.TaskFailed(f"saw {sorted(seen)} in the drawer")
    for c in sorted(seen, key=lambda c: -seen[c].world[1]):          # the one further left first
        side = "l" if seen[c].world[1] > 0 else "r"                   # the hand on its side
        est = SB._refine(wb, eye, plane, c, seen[c].world)
        _take(wb, side, f"part_{c}", est)
        x, y = PADS[c]
        dy = y - (SB.HAND_Y if side == "l" else -SB.HAND_Y)           # its pad's on the other side: step over to it
        if dy:
            wb.look([x, y, T + 0.02], dur=0.5)
            SB._shift(wb, dy, carry=True)
        # (set down a little short: the opening hand nudges it on as it lets go; it ends with the hand up, clear of
        # the open drawer's handle on the way back)
        WS._place(wb, side, f"part_{c}", (x - 0.015, y), T + PAD_H + 0.002)
        if dy:
            wb.look([0.5, 0.0, T], dur=0.5)
            SB._shift(wb, -dy)
    for side in ("l", "r"):
        wb.arm_joints_to(side, READY, 0.8)
    wb.settle()
    notes["on_pads"] = _on_pads(m, d)
    if _on_pads(m, d) < 2:
        raise S.TaskFailed(f"{_on_pads(m, d)} of 2 parts on their pads")
    # 3. push it shut: a step back to take the handle (it's close to the body now), the arm pushes, a step in
    wb.walk(-BACK)
    _grip(wb)
    _move(wb, ARM)
    wb.walk(BACK + 0.004, carry=True)
    _let_go(wb)
    wb.look([0.55, 0.0, T + 0.05], dur=0.8)
    wb.wait(1.5)
    notes["shut"] = round(_slide(m, d), 4)
    if _slide(m, d) > 0.005:
        raise S.TaskFailed(f"the drawer's still {_slide(m, d):.3f} m open")


def _grip(wb) -> None:
    """Take the drawer's handle with the right hand: fingertips round the knob's rim (a grip it can pull on)."""
    from pioneer_whole_body import skills as S
    top = S.site(wb, "handle_top")
    p, r = S.knob_pose(top, 0.0, S.TWIST_ABOVE)
    wb.look(top, dur=0.5)
    wb.fingers("r", "knob_pre", dur=0.5)
    wb.reach("r", p + [0, 0, S.CLEAR], r, 1.0)
    wb.settle()
    wb.reach("r", p, r, 0.6)
    wb.settle()
    wb.fingers("r", "knob", amount=S.TWIST_SQUEEZE, dur=0.45)
    wb.settle()
    wb.wait(0.15)


def _move(wb, dx: float, step: float = 0.01, dt: float = 0.12) -> None:
    """Carry the held handle ``dx`` along x (minus: pulled open) in small straight moves."""
    p, r = wb.hand_goal["r"]
    n = max(1, math.ceil(abs(dx) / step))
    for k in range(1, n + 1):
        wb.reach("r", p + [dx * k / n, 0, 0], r, dt)
        wb.settle()


def _let_go(wb) -> None:
    """The fingertips straight out from the rim, then the hand straight up off it (an opening hand that swept its
    thumb across would shove the drawer)."""
    wb.wait(0.2)
    wb.fingers("r", "knob_pre", dur=0.5)
    wb.settle()
    q, r = wb.hand_goal["r"]
    wb.reach("r", q + [0, 0, 0.08], r, 0.7)
    wb.settle()
    wb.fingers("r", "relaxed", dur=0.4)
    wb.arm_joints_to("r", READY, 0.9)
    wb.settle()


def _take(wb, side, part, est) -> None:
    """Take the part by its knob (``sort_bins``' grasp); again from where it is now if it didn't come up."""
    from pioneer_whole_body import skills as S
    for _ in range(3):
        z0 = S.body_pos(wb, part)[2]
        SB._pick(wb, side, est, yaw=0.6 if side == "r" else -0.6)   # (fingers in toward the middle: straight
                                                                        # forward they'd reach the cabinet's roof)
        if S.body_pos(wb, part)[2] - z0 > 0.03:
            return
        S.release(wb, side, up=0.06)
        est = np.array([*S.body_pos(wb, part)[:2], est[2]])
    raise S.TaskFailed(f"{part} not taken")


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["drawer"].plan = plan
