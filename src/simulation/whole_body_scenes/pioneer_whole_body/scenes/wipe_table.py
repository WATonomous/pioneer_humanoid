"""Wiping the bench, by sight (both hands, non-prehensile): crumbs spilled on the bench are swept into the gutter at
its back with a squeegee in each hand. Where the crumbs are comes from the robot's head camera (``vision.Eye``),
and it looks again after every pass: the crumbs move only by being pushed (contact physics), so some slip past the
blade's ends and need another pass.

The robot walks up to the bench. On it:

  crumbs     twelve small cubes spilled in front of the robot (where changes with every seed: ``reset``)
  squeegees  one on each side standing in its tray, a blade 14 cm long with a knob on top
  gutter     along the back of the work area: a tray with a back wall and sides, open to the front

The plan: look over the bench, find the crumbs; take a squeegee in each hand; then pass after pass, both hands
together, each in its own lanes (the left hand's left of the middle): set the blade down behind the nearest crumb
in the lane, push it to the gutter, lift it back; look again, and pass the lanes that still have crumbs in them,
until the bench is clean. Then the squeegees go back in their trays.

Steps (``progress``): 1. look over the bench, 2. take the squeegees, 3. wipe the bench (n of 12 in the gutter),
4. put the squeegees back.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY, palm_down
from pioneer_whole_body.parts import KNOB, KNOB_H, KNOB_R, STEEL
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

N = 12
CRUMB = 0.006                              # half size
CRUMB_RGBA = [0.25, 0.62, 0.18, 1]          # (peas): green, for the camera
SPILL = ((0.44, 0.545), (-0.23, 0.23))     # where they're spilled (x range, y range)
GUTTER = (0.575, 0.69, 0.27)               # front edge x, back wall x, half width y
BLADE = (0.016, 0.07, 0.015)               # half sizes: 14 cm long (along y)
POST_H = 0.012
GRIP_Z = 2 * BLADE[2] + POST_H + KNOB_H + 0.0012     # the knob top above the blade's bottom
HOME = {"l": (0.50, 0.36), "r": (0.50, -0.36)}       # where the squeegees lie
TRAY_IN = (0.045, 0.076)                   # the squeegees' trays: half the inside (x, y), wall height, floor
TRAY_WALL, TRAY_FLOOR = 0.03, 0.0
LANES = {"l": (0.065, 0.19), "r": (-0.065, -0.19)}   # each hand's lanes (the blade's middle, y)
END_X = 0.625                              # the blade's middle at the end of a pass: its front over the gutter
GRIP = 1.1                                 # how hard the hands close on the squeegees' knobs (``lid`` amount)
BEHIND = 0.05                              # the blade comes down this far behind the nearest crumb
HOVER = 0.005                              # the blade's bottom above the bench in a pass
ASIDE = {**READY, "shoulder_pitch": -0.45}          # hands up and out to the sides (out of the camera's view)
STEPS = ("look over the bench", "take the squeegees", "wipe the bench", "put the squeegees back")
_U_STEP, _U_SEEN, _U_TAKEN, _U_WIPED = 0, 1, 2, 3


def _in_gutter(model, data) -> int:
    n = 0
    for i in range(N):
        p = data.xpos[model.body(f"crumb{i}").id]
        if p[0] > GUTTER[0] - 0.002 and abs(p[1]) < GUTTER[2] and p[2] > T - 0.01:
            n += 1
    return n


def _home(model, data, side) -> bool:
    """The squeegee back in its tray (any way up)."""
    p = data.xpos[model.body(f"squeegee_{side}").id]
    x, y = HOME[side]
    return abs(p[0] - x) < TRAY_IN[0] and abs(p[1] - y) < TRAY_IN[1] and p[2] < T + TRAY_WALL + 0.02


def step(model, data) -> None:
    k = int(data.userdata[_U_STEP])
    if k == 0 and data.userdata[_U_SEEN]:
        data.userdata[_U_STEP] = 1
    elif k == 1 and data.userdata[_U_TAKEN]:
        data.userdata[_U_STEP] = 2
    elif k == 2 and _in_gutter(model, data) == N:
        data.userdata[_U_STEP] = 3
    elif k == 3 and _home(model, data, "l") and _home(model, data, "r") and data.userdata[_U_WIPED]:
        data.userdata[_U_STEP] = 4


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 2:
        return k, len(STEPS), f"wipe the bench ({_in_gutter(model, data)} of {N} crumbs in the gutter)"
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _spill(rng: np.random.Generator):
    (x0, x1), (y0, y1) = SPILL
    pts = []
    while len(pts) < N:
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if all(np.linalg.norm(p - q) > 0.025 for q in pts):
            pts.append(p)
    return pts


def reset(model, data, rng) -> None:
    for i, p in enumerate(_spill(rng)):
        a = model.joint(f"crumb{i}").qposadr[0]
        data.qpos[a:a + 3] = [p[0], p[1], T + CRUMB + 0.0005]
        yaw = rng.uniform(-math.pi, math.pi)
        data.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def _squeegee(world, side):
    x, y = HOME[side]
    b = world.add_body(name=f"squeegee_{side}", pos=[x, y, T + TRAY_FLOOR + 0.0005])
    b.add_freejoint(name=f"squeegee_{side}")
    hx, hy, hz = BLADE
    b.add_geom(type=BOX, size=[hx, hy, hz * 0.7], pos=[0, 0, hz * 1.3], rgba=[0.2, 0.22, 0.26, 1], mass=0.04)
    b.add_geom(type=BOX, size=[hx * 0.75, hy, hz * 0.3], pos=[0, 0, hz * 0.3], rgba=[0.9, 0.45, 0.12, 1], mass=0.01,
               friction=[0.1, 0.01, 0.001])                                           # the rubber blade
    z = 2 * hz
    b.add_geom(type=CYL, size=[0.008, POST_H / 2, 0], pos=[0, 0, z + POST_H / 2], rgba=STEEL, mass=0.004)
    z += POST_H
    # the handle: a square knob (the fingers on its flats: it can't turn or slip in the grip when the blade drags)
    k = KNOB_R * 0.88
    b.add_geom(type=BOX, size=[k, k, KNOB_H / 2], pos=[0, 0, z + KNOB_H / 2], rgba=KNOB, mass=0.015,
               friction=[2.0, 0.03, 0.003], condim=4)
    b.add_site(name=f"squeegee_{side}_top", pos=[0, 0, GRIP_Z])


@scene("wipe_table", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  wiping the bench: crumbs found by its head camera, swept into the gutter")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 4
    world = spec.worldbody
    grey = [0.32, 0.34, 0.37, 1]
    # the gutter: a thin floor (crumbs are pushed up onto it), a back wall and sides
    x0, x1, hy = GUTTER
    g = world.add_body(name="gutter", pos=[0, 0, T])
    g.add_geom(type=BOX, size=[(x1 - x0) / 2, hy, 0.0008], pos=[(x0 + x1) / 2, 0, 0.0008], rgba=[0.24, 0.26, 0.3, 1])
    g.add_geom(type=BOX, size=[0.005, hy + 0.01, 0.02], pos=[x1 + 0.005, 0, 0.02], rgba=grey)
    for s in (-1, 1):
        g.add_geom(type=BOX, size=[(x1 - x0) / 2 + 0.005, 0.005, 0.02], pos=[(x0 + x1) / 2, s * (hy + 0.005), 0.02],
                   rgba=grey)
    for side in ("l", "r"):
        _squeegee(world, side)
        x, y = HOME[side]                       # its tray: an open box it stands in
        hx, hy = TRAY_IN
        world.add_geom(type=BOX, size=[hx + 0.004, hy + 0.004, 0.0005], pos=[x, y, T + 0.0005], rgba=grey,
                       contype=0, conaffinity=0)              # (its floor: the bench)
        for k in (-1, 1):
            world.add_geom(type=BOX, size=[0.004, hy + 0.004, TRAY_WALL / 2], pos=[x + k * (hx + 0.004), y, T + TRAY_WALL / 2],
                           rgba=grey)
            world.add_geom(type=BOX, size=[hx, 0.004, TRAY_WALL / 2], pos=[x, y + k * (hy + 0.004), T + TRAY_WALL / 2],
                           rgba=grey)
    for i, p in enumerate(_spill(np.random.default_rng(0))):
        c = world.add_body(name=f"crumb{i}", pos=[p[0], p[1], T + CRUMB + 0.0005])
        c.add_freejoint(name=f"crumb{i}")
        c.add_geom(type=BOX, size=[CRUMB] * 3, rgba=CRUMB_RGBA, mass=0.003, friction=[0.5, 0.01, 0.001])


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    eye = Eye(m)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _hands(wb, HIGH, 0.8)
    wb.walk(0.30)

    crumbs = _scan(wb, eye, aside=False)
    d.userdata[_U_SEEN] = 1
    notes["seen"] = [len(crumbs)]
    for side in ("l", "r"):
        _take(wb, side)
    d.userdata[_U_TAKEN] = 1
    passes = 0
    for _ in range(6):
        lanes = {}                                       # (hand, 0 inner / 1 outer): (y, the crumbs' x)
        for side in ("l", "r"):
            for k, y in enumerate(LANES[side]):
                xs = [c[0] for c in crumbs if abs(c[1] - y) < BLADE[1] + 0.005]
                if xs:
                    lanes[side, k] = (y, xs)
        if not lanes:
            break
        # both hands at once, but never both in the inner lanes (the hands would meet in the middle)
        for pair in ((("l", 0), ("r", 1)), (("l", 1), ("r", 0))):
            job = {key[0]: lanes[key] for key in pair if key in lanes}
            if job:
                _pass(wb, {side: (min(xs) - BEHIND, y) for side, (y, xs) in job.items()})
                passes += 1
        crumbs = _scan(wb, eye)
        notes["seen"].append(len(crumbs))
    notes["passes"] = passes
    d.userdata[_U_WIPED] = 1
    for side in ("l", "r"):                              # the squeegees back in their holders
        _stow(wb, side)
        wb.arm_joints_to(side, HIGH, 1.0)                # (in the ready pose the forearm would lie over it)
        wb.settle()
    wb.look([0.69, 0.0, T], dur=0.8)
    wb.wait(1.5)
    notes["final"] = dict(in_gutter=_in_gutter(m, d))
    if _in_gutter(m, d) < N:
        raise S.TaskFailed(f"{_in_gutter(m, d)} of {N} crumbs in the gutter")


def _scan(wb, eye, aside: bool = True) -> list:
    """Two looks over the bench (``aside``: the hands out to the sides first, out of the camera's view); the crumbs
    (green blobs) not yet in the gutter, as (x, y)."""
    if aside:
        for side in ("l", "r"):
            wb.reach(side, wb.palm_pose(side)[0] + [0, 0, 0.08], None, 0.5)     # up off the bench, then aside
        wb.settle()
        for side in ("l", "r"):
            wb.arm_joints_to(side, ASIDE, 0.9)
        wb.settle()
    (x0, x1), (y0, y1) = SPILL
    plane = T + 2 * CRUMB
    found = []
    for y in (0.12, -0.12):
        wb.look([(x0 + x1) / 2, y, T], dur=0.7)
        wb.wait(0.35)
        img = eye.look(wb.d)
        for b in eye.find(wb.d, plane, min_area=8, image=img):
            if b.colour != "green":
                continue
            p = b.world[:2]
            if p[0] > GUTTER[0] - 0.002 or p[0] < 0.35 or abs(p[1]) > GUTTER[2] + 0.05:
                continue
            if any(np.linalg.norm(p - f) < 0.012 for f in found):
                continue
            found.append(p)
    return found


def _pass(wb, starts: dict) -> None:
    """One pass of each hand in ``starts`` (side: (x, y) of the blade's middle to start from), together: over the
    start, down, push to the gutter, up and back."""
    from pioneer_whole_body import skills as S

    goals = {}
    for s, (x, y) in starts.items():
        p, r = wb.hand_goal[s] or wb.palm_pose(s)
        r = palm_down(0.0)
        off = S.site(wb, f"squeegee_{s}_top") - p          # where the knob hangs from the palm
        start = np.array([x, y, T + HOVER + GRIP_Z]) - off
        goals[s] = (start, r, np.array([END_X, y, T + HOVER + GRIP_Z]) - off)
    look = np.mean([np.array([x, y, T]) for x, y in starts.values()], axis=0)
    wb.look(look, dur=0.4)
    for z_up, dur in ((0.06, 0.9), (0.0, 0.5)):
        for s, (start, r, _) in goals.items():
            wb.reach(s, start + [0, 0, z_up], r, dur)
        wb.settle()
    for s, (start, r, end) in list(goals.items()):     # the blade sits a little differently in each grip: down
        dz = wb.d.xpos[wb.m.body(f"squeegee_{s}").id][2] - (T + HOVER)    # until it just clears the bench
        goals[s] = (start - [0, 0, dz], r, end - [0, 0, dz])
        wb.reach(s, start - [0, 0, dz], r, 0.3)
    wb.settle()
    wb.look([0.62, look[1], T], dur=0.6)
    for s, (start, r, end) in goals.items():
        wb.reach(s, end, r, 1.6)
    wb.settle()
    for s, (start, r, end) in goals.items():
        wb.reach(s, end + [0, 0, 0.06], r, 0.4)
    wb.settle()


def _take(wb, side: str) -> None:
    from pioneer_whole_body import skills as S
    S.pick_knob(wb, side, f"squeegee_{side}_top", f"squeegee_{side}", yaw=0.0, lift=0.05)
    wb.fingers(side, "lid", amount=GRIP, dur=0.3)


def _stow(wb, side: str) -> None:
    """Back in its tray: over it, down until it's just above the floor, let go (it drops in), up off it."""
    from pioneer_whole_body import skills as S
    x, y = HOME[side]
    body = f"squeegee_{side}"
    wb.reach(side, (wb.hand_goal[side] or wb.palm_pose(side))[0], palm_down(0.0), 0.5)
    wb.settle()
    S.carry(wb, side, f"{body}_top", np.array([x, y, T + TRAY_WALL + GRIP_Z]), clear=0.01, dur=1.4)
    S.lower_until(wb, side, 0.05, lambda: wb.d.xpos[wb.m.body(body).id][2] < T + TRAY_FLOOR + 0.012, dur=0.8)
    wb.fingers(side, "open", dur=0.6)                  # the hand right open, then straight up off it (a half-open
    wb.settle()                                        # hand catches the knob's corners)
    wb.wait(0.3)
    p, r = wb.hand_goal[side]
    wb.reach(side, p + [0, 0, 0.08], r, 0.8)
    wb.settle()


def _hands(wb, pose, dur: float = 1.0) -> None:
    for side in ("l", "r"):
        wb.arm_joints_to(side, pose, dur)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["wipe_table"].plan = plan
