"""Pouring (both hands): two cups of balls are emptied into a bowl. A cup is held by the knob on its lid and has one
side open above a low lip: the hand rolls about its fingers (the wrist and forearm turn the cup on its side) and the
balls run out of the open side, by gravity and contact alone.

The robot walks up to the bench. On it:

  cups   one on each side on a coaster, twelve red balls in the left one, twelve blue in the right one; the open side
         of each faces out (the left hand rolls its cup to the left, the right hand to the right)
  bowl   in the middle, a shallow tray

The plan, for each hand (left first): take the cup by its knob, lift it and carry it over the bowl (the cup's middle
on the far side of the bowl from its open side), roll the hand slowly until the cup lies on its side, hold it there
while the balls run out (and give it a shake), roll back, put the cup back on its coaster.

Steps (``progress``): 1. pour the red balls (n of 12 in the bowl), 2. pour the blue balls, 3. cups back.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY
from pioneer_whole_body.parts import KNOB, KNOB_H, KNOB_R, quat_z
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE

N = 12
BALL_R = 0.0065
CUP = (0.032, 0.032, 0.036)              # half sizes (outside)
WALL = 0.003
LIP = 0.008                              # the open side's lip
POST_H = 0.01
TOP_Z = 2 * CUP[2] + POST_H + KNOB_H + 0.0012     # the knob top above the cup's bottom
CUPS = {"l": (0.50, 0.24), "r": (0.50, -0.24)}   # the coasters
BOWL = (0.52, 0.0)
BOWL_IN = (0.075, 0.09)                  # half the inside
BOWL_H = 0.03
POCKET_H = 0.006                         # the coasters: thickness, radius
COASTER = 0.07
ROLL = 1.55                              # how far the hand rolls to pour (rad)
COLOURS = {"l": [0.85, 0.15, 0.12, 1], "r": [0.15, 0.35, 0.86, 1]}
STEPS = ("pour the red balls", "pour the blue balls", "put the cups back")
_U_STEP, _U_DONE = 0, 1


def _in_bowl(model, data, side) -> int:
    n = 0
    for i in range(N):
        p = data.xpos[model.body(f"ball_{side}{i}").id]
        if abs(p[0] - BOWL[0]) < BOWL_IN[0] and abs(p[1] - BOWL[1]) < BOWL_IN[1] and p[2] < T + BOWL_H:
            n += 1
    return n


def _home(model, data, side) -> bool:
    p = data.xpos[model.body(f"cup_{side}").id]
    up = data.xmat[model.body(f"cup_{side}").id][8] > 0.95
    return np.linalg.norm(p[:2] - CUPS[side]) < COASTER - 0.02 and p[2] < T + POCKET_H + 0.004 and up  # on its coaster


def step(model, data) -> None:
    k = int(data.userdata[_U_STEP])
    if k < 2 and _in_bowl(model, data, "lr"[k]) >= N - 1 and not _held_over(model, data, "lr"[k]):
        data.userdata[_U_STEP] = k + 1
    elif k == 2 and _home(model, data, "l") and _home(model, data, "r") and data.userdata[_U_DONE]:
        data.userdata[_U_STEP] = 3


def _held_over(model, data, side) -> bool:
    """The cup still away from its coaster (being poured): the step is done once it is back."""
    p = data.xpos[model.body(f"cup_{side}").id]
    return np.linalg.norm(p[:2] - CUPS[side]) > 0.03


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k < 2:
        return k, len(STEPS), f"{STEPS[k]} ({_in_bowl(model, data, 'lr'[k])} of {N} in the bowl)"
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _cup(spec, world, side):
    x, y = CUPS[side]
    s = -1 if side == "l" else 1                   # the open side faces out (+y for the left): the side it pours to
    b = world.add_body(name=f"cup_{side}", pos=[x, y, T + POCKET_H + 0.0005])
    b.add_freejoint(name=f"cup_{side}")
    hx, hy, hz = CUP
    grey = [0.75, 0.77, 0.8, 1]
    b.add_geom(type=BOX, size=[hx, hy, WALL / 2], pos=[0, 0, WALL / 2], rgba=grey, mass=0.035)   # bottom (weighted:
    b.add_geom(type=BOX, size=[hx, hy, WALL / 2], pos=[0, 0, 2 * hz - WALL / 2], rgba=grey, mass=0.005)  # the empty
    for k in (-1, 1):                                                                         # cup stands firm)
        b.add_geom(type=BOX, size=[WALL / 2, hy, hz], pos=[k * (hx - WALL / 2), 0, hz], rgba=grey, mass=0.004)
    b.add_geom(type=BOX, size=[hx, WALL / 2, hz], pos=[0, s * (hy - WALL / 2), hz], rgba=grey, mass=0.004)
    b.add_geom(type=BOX, size=[hx, WALL / 2, LIP / 2], pos=[0, -s * (hy - WALL / 2), LIP / 2], rgba=grey,
               mass=0.004)                                                                           # the lip
    b.add_geom(type=BOX, size=[hx * 0.8, 0.0008, hz * 0.6], pos=[0, s * (hy + 0.0008), hz], rgba=COLOURS[side],
               contype=0, conaffinity=0, mass=0.0001)                                                 # a label
    z = 2 * hz
    b.add_geom(type=CYL, size=[KNOB_R, POST_H / 2, 0], pos=[0, 0, z + POST_H / 2], rgba=grey, mass=0.004)
    # (a boss as wide as the knob under it: no undercut for the thumb to hook under when letting go)
    z += POST_H
    b.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[0, 0, z + KNOB_H / 2], rgba=KNOB, mass=0.015,
               friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):
        a = 2 * math.pi * i / 10
        b.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                   pos=[KNOB_R * math.cos(a), KNOB_R * math.sin(a), z + KNOB_H / 2], quat=quat_z(a), rgba=KNOB,
                   contype=0, conaffinity=0, mass=0.0001)
    b.add_site(name=f"cup_{side}_top", pos=[0, 0, TOP_Z])
    b.add_site(name=f"cup_{side}_bottom", pos=[0, 0, 0])
    # the balls, in layers on the bottom
    i = 0
    for layer in range(3):
        for gx in (-1, 1):
            for gy in (-1, 1):
                if i == N:
                    break
                p = [x + gx * 0.013, y + gy * 0.013, T + POCKET_H + WALL + BALL_R + 0.001 + layer * 2.05 * BALL_R]
                ball = world.add_body(name=f"ball_{side}{i}", pos=p)
                ball.add_freejoint(name=f"ball_{side}{i}")
                ball.add_geom(type=SPHERE, size=[BALL_R, 0, 0], rgba=COLOURS[side], mass=0.0015,
                              friction=[0.3, 0.005, 0.0005])
                i += 1


@scene("pour", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress,
       title="Pioneer  |  pouring: two cups of balls emptied into a bowl")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 2
    world = spec.worldbody
    dark = [0.16, 0.18, 0.22, 1]
    for side, (x, y) in CUPS.items():               # a coaster under each cup
        pk = world.add_body(name=f"coaster_{side}", pos=[x, y, T])
        pk.add_geom(type=CYL, size=[COASTER, POCKET_H / 2, 0], pos=[0, 0, POCKET_H / 2], rgba=dark)
        _cup(spec, world, side)
    bx, by = BOWL
    hx, hy = BOWL_IN
    bowl = world.add_body(name="bowl", pos=[bx, by, T])
    bowl.add_geom(type=BOX, size=[hx + 0.004, hy + 0.004, 0.0015], pos=[0, 0, 0.0015], rgba=[0.9, 0.9, 0.88, 1])
    for k in (-1, 1):
        bowl.add_geom(type=BOX, size=[0.004, hy + 0.008, BOWL_H / 2], pos=[k * (hx + 0.004), 0, BOWL_H / 2],
                      rgba=[0.9, 0.9, 0.88, 1])
        bowl.add_geom(type=BOX, size=[hx, 0.004, BOWL_H / 2], pos=[0, k * (hy + 0.004), BOWL_H / 2],
                      rgba=[0.9, 0.9, 0.88, 1])


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S

    m, d = wb.m, wb.d
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    for side in ("l", "r"):
        wb.arm_joints_to(side, HIGH, 0.8)
    wb.settle()
    wb.walk(0.30)
    notes["poured"] = {}
    for side in ("l", "r"):
        _pour(wb, side)
        notes["poured"][side] = _in_bowl(m, d, side)
        wb.arm_joints_to(side, HIGH, 0.9)
        wb.settle()
    d.userdata[_U_DONE] = 1
    for side in ("l", "r"):
        wb.arm_joints_to(side, READY, 1.0)
    wb.look([BOWL[0], BOWL[1], T], dur=0.6)
    wb.wait(1.5)
    notes["final"] = {s: _in_bowl(m, d, s) for s in ("l", "r")}
    if min(notes["final"].values()) < N - 1 or not (_home(m, d, "l") and _home(m, d, "r")):
        raise S.TaskFailed(f"bowl {notes['final']}, cups home {_home(m, d, 'l')}, {_home(m, d, 'r')}")


def _axis_rot(a, th):
    a = np.asarray(a, float) / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k


def _pour(wb, side: str) -> None:
    from pioneer_whole_body import skills as S
    sg = 1 if side == "r" else -1                  # the right hand rolls +, the left - (each pours toward its side)
    yaw = sg * 0.3
    cup = f"cup_{side}"
    S.pick_knob(wb, side, f"{cup}_top", cup, yaw=yaw, lift=0.04)
    # over the bowl: the cup's middle on the bowl's far side from its open side (the balls run toward the middle)
    over = np.array([BOWL[0], BOWL[1] + sg * 0.035, T + BOWL_H + 0.03])
    S.carry(wb, side, f"{cup}_bottom", over, clear=0.0, dur=1.6, over=0.04)
    p, r = wb.hand_goal[side]
    pivot = S.site(wb, f"{cup}_top") - [0, 0, CUP[2] + POST_H + KNOB_H]     # (about the cup's middle)
    f = r[:, 0]
    wb.look([BOWL[0], BOWL[1], T], dur=0.4)
    n = 12
    for k in range(1, n + 1):                      # roll it onto its side, slowly
        rot = _axis_rot(f, sg * ROLL * k / n)
        wb.reach(side, pivot + rot @ (p - pivot), rot @ r, 0.18)
        wb.settle()
    wb.wait(1.0)
    for k in range(5):                             # a shake
        for dy in (0.02, -0.02):
            q, rr = wb.hand_goal[side]
            wb.reach(side, q + [0, dy, 0], rr, 0.12)
            wb.settle()
    wb.wait(0.6)
    for k in range(n - 1, -1, -1):
        rot = _axis_rot(f, sg * ROLL * k / n)
        wb.reach(side, pivot + rot @ (p - pivot), rot @ r, 0.12)
        wb.settle()
    # back on its coaster
    x, y = CUPS[side]
    S.carry(wb, side, f"{cup}_bottom", np.array([x, y, T + POCKET_H]), clear=0.012, dur=1.6, over=0.05)
    S.lower_until(wb, side, 0.03, lambda: wb.d.xpos[wb.m.body(cup).id][2] < T + POCKET_H + 0.002, dur=0.8)
    wb.fingers(side, "lid_pre", dur=0.5)           # the fingers off the knob (not right open: they'd push the
                                                   # empty cup along), then straight up
    wb.settle()
    wb.wait(0.2)
    p, r = wb.hand_goal[side]
    wb.reach(side, p + [0, 0, 0.09], r, 0.8)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["pour"].plan = plan
