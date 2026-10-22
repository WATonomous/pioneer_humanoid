"""Sorting by colour, by sight (long horizon, whole body): parts in a kitting tray go into the bin of their colour.
What colour each part is and where it is come from the robot's head camera (``vision.Eye``), not from the simulator.

The robot walks up to the bench. On it:

  tray    a row of six pockets in front of the robot, a knob-topped part in each, the tops of the knobs red, green
          or blue (two of each; which is where changes with every seed, ``reset``, and the parts sit a little off
          centre, turned any way)
  bins    green in the middle behind the tray; red out at the bench's left end, blue at its right end

The plan: look over the tray (two looks, left and right), find the coloured tops in the images and put them on the
bench (each blob's ray met with the plane of the knob tops). Then, from the middle out, for each part: look straight
at it (the estimate from there), take it by its knob with the hand on its side of the body, and:
  green  put it in the green bin, in the hand's half (a side step over to the other half if that one is taken)
  red / blue  side-step along the bench carrying it until the bin is by the hand, put it in, step back
At the end it looks over the tray once more: anything still there (a part dropped back in) is done again.

Steps (``progress``): 1. look over the tray, 2. sort the parts (n of 6 in their bins).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY
from pioneer_whole_body.parts import KNOB, KNOB_H, KNOB_R, STEEL, quat_z
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

N = 6
COLOURS = {"red": [0.85, 0.12, 0.10, 1], "green": [0.12, 0.68, 0.25, 1], "blue": [0.12, 0.32, 0.85, 1]}
BINS = {"red": (0.60, 0.43), "green": (0.60, 0.0), "blue": (0.60, -0.43)}
BIN_IN = (0.048, 0.085)  # half the inside (x, y): two parts side by side, a hand's width apart
SLOTS = (-0.05, 0.05)     # where in a bin (y) its two parts go
BIN_WALL, BIN_H = 0.006, 0.035
CELLS = [(0.49, y) for y in (-0.25, -0.15, -0.05, 0.05, 0.15, 0.25)]   # the kitting tray's pockets (a row: nearer
# the body the hands can't come down over a part)
HAND_Y = 0.21                            # where (y from the body) a hand puts a part in a bin
STEP = 0.15
AREA = ((0.40, 0.53), (-0.29, 0.29))      # the tray: where the camera looks for parts
POCKET = 0.029                           # half the pocket's inside width (3 mm of play round the base)
POCKET_H = 0.008
TRAY_H = 0.006
BASE_R, BASE_H = 0.026, 0.01
POST_H = 0.012
TOP_Z = BASE_H + POST_H + KNOB_H + 0.0012          # the knob top above the part's bottom
STEPS = ("look over the tray", "sort the parts")
_U_STEP, _U_SEEN = 0, 1


def _sorted(model, data) -> int:
    n = 0
    for i in range(N):
        p = data.xpos[model.body(f"part{i}").id]
        colour = _colour_of(model, i)
        bx, by = BINS[colour]
        if abs(p[0] - bx) < BIN_IN[0] and abs(p[1] - by) < BIN_IN[1] and p[2] < T + BIN_H:  # (in the bin, not the tray)
            n += 1
    return n


def _colour_of(model, i) -> str:
    rgba = model.geom_rgba[model.geom(f"part{i}_cap").id]
    return min(COLOURS, key=lambda c: np.linalg.norm(np.asarray(COLOURS[c][:3]) - rgba[:3]))


def step(model, data) -> None:
    k = int(data.userdata[_U_STEP])
    if k == 0 and data.userdata[_U_SEEN]:
        data.userdata[_U_STEP] = 1
    elif k == 1 and _sorted(model, data) == N:
        data.userdata[_U_STEP] = 2


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 0:
        return 0, len(STEPS), STEPS[0]
    return k, len(STEPS), f"sort the parts by colour ({_sorted(model, data)} of {N} in their bins)"


def _layout(rng: np.random.Generator):
    """A part in each pocket (a little off centre), two of each colour, shuffled."""
    cells = rng.choice(len(CELLS), N, replace=False)
    pts = [np.asarray(CELLS[c]) + rng.uniform(-0.002, 0.002, 2) for c in cells]
    colours = [c for c in COLOURS for _ in range(N // len(COLOURS))]
    rng.shuffle(colours)
    return pts, colours


def reset(model, data, rng) -> None:
    pts, colours = _layout(rng)
    for i, (p, c) in enumerate(zip(pts, colours)):
        a = model.joint(f"part{i}").qposadr[0]
        data.qpos[a:a + 3] = [p[0], p[1], T + TRAY_H + 0.0005]
        yaw = rng.uniform(-math.pi, math.pi)
        data.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        model.geom_rgba[model.geom(f"part{i}_cap").id] = COLOURS[c]


def _part(world, name, pos, colour):
    b = world.add_body(name=name, pos=list(pos))
    b.add_freejoint(name=name)
    b.add_geom(type=CYL, size=[BASE_R, BASE_H / 2, 0], pos=[0, 0, BASE_H / 2], rgba=STEEL, mass=0.02,
               friction=[0.8, 0.02, 0.002])
    b.add_geom(type=CYL, size=[0.008, POST_H / 2, 0], pos=[0, 0, BASE_H + POST_H / 2], rgba=STEEL, mass=0.004)
    z = BASE_H + POST_H
    b.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[0, 0, z + KNOB_H / 2], rgba=KNOB, mass=0.015,
               friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):
        a = 2 * math.pi * i / 10
        b.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                   pos=[KNOB_R * math.cos(a), KNOB_R * math.sin(a), z + KNOB_H / 2], quat=quat_z(a), rgba=KNOB,
                   contype=0, conaffinity=0, mass=0.0001)
    b.add_geom(name=f"{name}_cap", type=CYL, size=[KNOB_R * 0.97, 0.0006, 0], pos=[0, 0, TOP_Z - 0.0006],
               rgba=COLOURS[colour], contype=0, conaffinity=0, mass=0.0001)
    b.add_site(name=f"{name}_top", pos=[0, 0, TOP_Z])
    b.add_site(name=f"{name}_bottom", pos=[0, 0, 0])


@scene("sort_bins", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer  |  sorting by colour: found with its head camera, into the bins")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 2
    world = spec.worldbody
    grey = [0.32, 0.34, 0.37, 1]
    for colour, (bx, by) in BINS.items():
        b = world.add_body(name=f"bin_{colour}", pos=[bx, by, T])
        hx, hy = BIN_IN
        b.add_geom(type=BOX, size=[hx + BIN_WALL, hy + BIN_WALL, 0.002], pos=[0, 0, 0.002], rgba=grey)
        for s in (-1, 1):
            b.add_geom(type=BOX, size=[BIN_WALL / 2, hy + BIN_WALL, BIN_H / 2],
                       pos=[s * (hx + BIN_WALL / 2), 0, BIN_H / 2], rgba=grey)
            b.add_geom(type=BOX, size=[hx, BIN_WALL / 2, BIN_H / 2], pos=[0, s * (hy + BIN_WALL / 2), BIN_H / 2],
                       rgba=grey)
        # the bin's colour: a tag on its far wall (the parts' area is what the camera searches)
        b.add_geom(type=BOX, size=[0.001, 0.035, 0.012], pos=[hx + BIN_WALL + 0.001, 0, BIN_H * 0.55],
                   rgba=COLOURS[colour], contype=0, conaffinity=0)
    # the kitting tray: a plate with four low blocks round each pocket
    tray = world.add_body(name="tray", pos=[0, 0, T])
    (x0, x1), (y0, y1) = AREA
    tray.add_geom(type=BOX, size=[(x1 - x0) / 2, (y1 - y0) / 2, TRAY_H / 2], pos=[(x0 + x1) / 2, (y0 + y1) / 2, TRAY_H / 2],
                  rgba=[0.2, 0.22, 0.25, 1])
    for cx, cy in CELLS:
        for dx, dy, sx, sy in ((POCKET + 0.003, 0, 0.003, 0.014), (-POCKET - 0.003, 0, 0.003, 0.014),
                               (0, POCKET + 0.003, 0.014, 0.003), (0, -POCKET - 0.003, 0.014, 0.003)):
            tray.add_geom(type=BOX, size=[sx, sy, POCKET_H / 2], pos=[cx + dx, cy + dy, TRAY_H + POCKET_H / 2],
                          rgba=[0.32, 0.34, 0.37, 1])
    pts, colours = _layout(np.random.default_rng(0))
    for i, p in enumerate(pts):
        _part(world, f"part{i}", [p[0], p[1], T + TRAY_H + 0.0005], colours[i])


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    eye = Eye(m)
    plane = T + TRAY_H + TOP_Z
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _hands(wb, HIGH, 0.8)
    wb.walk(0.30)

    found = _scan(wb, eye, plane, notes)
    d.userdata[_U_SEEN] = 1
    notes["seen"] = [(b.colour, [round(float(v), 4) for v in b.world[:2]]) for b in found]
    done = 0
    taken = {c: set() for c in COLOURS}
    for _ in range(3):                                # (a part still in the tray: again)
        todo = sorted(found, key=lambda b: abs(b.world[1]))
        for b in todo:
            _sort_one(wb, eye, plane, b, taken[b.colour])
            done += 1
        found = _scan(wb, eye, plane, notes)
        if not found:
            break
    _hands(wb, READY, 1.0)
    wb.look([0.63, 0.0, T + 0.05], dur=0.8)
    wb.wait(1.5)
    notes["final"] = dict(sorted=_sorted(m, d), handled=done)
    if _sorted(m, d) < N:
        raise S.TaskFailed(f"{_sorted(m, d)} of {N} parts in their bins")


def _scan(wb, eye, plane, notes) -> list:
    """Two looks over the parts' area (left half, right half), the blobs in it merged (one part seen twice: one)."""
    (x0, x1), (y0, y1) = AREA
    found = []
    for y in (0.13, -0.13):
        wb.look([(x0 + x1) / 2, y, T], dur=0.7)
        wb.wait(0.35)
        for b in eye.find(wb.d, plane):
            if not (x0 - 0.05 < b.world[0] < x1 + 0.05 and y0 - 0.06 < b.world[1] < y1 + 0.06):
                continue                                   # a bin tag, or something on the far side
            if any(np.linalg.norm(b.world[:2] - f.world[:2]) < 0.03 for f in found):
                continue
            found.append(b)
    notes.setdefault("scans", []).append(len(found))
    return found


def _sort_one(wb, eye, plane, blob, taken: set) -> None:
    """Take the part (the hand on its side of the body), put it in its bin. The green bin is in the middle, each
    hand has its own half of it (a step over to the other half if its own is taken); the red and blue bins are out at the bench's ends: the robot side-steps along the
    bench carrying the part until the bin is by the hand, puts it in and steps back. In those the slot nearer the
    middle of the body fills first (the hand's heel and the arm come in over the other one)."""
    est = _refine(wb, eye, plane, blob.colour, blob.world)
    bx, by = BINS[blob.colour]
    side = "l" if est[1] > 0 else "r"
    _pick(wb, side, est)
    if by == 0:                                     # its own half; if that's taken, a step over to the other
        own = SLOTS[0] if side == "r" else SLOTS[1]
        slot = own if own not in taken else -own
        dy = slot - own
    else:
        slot = _inner(side) if _inner(side) not in taken else -_inner(side)
        dy = by - (HAND_Y if side == "l" else -HAND_Y)      # stand so the bin is where the hand puts things
    taken.add(slot)
    if dy:
        wb.look([bx, by, T + 0.03], dur=0.5)
        _shift(wb, dy, carry=True)
    _put(wb, side, (bx, by + slot), T + 0.008, look=(bx, by, T + 0.03))
    if dy:
        wb.look([0.49, 0.0, T], dur=0.5)
        _shift(wb, -dy)


def _shift(wb, dy: float, carry: bool = False) -> None:
    """Side-step ``dy`` along the bench (leading foot, then the other, in steps of at most STEP)."""
    n = max(1, math.ceil(abs(dy) / STEP - 1e-9))
    lead, trail = ("l", "r") if dy > 0 else ("r", "l")
    for _ in range(n):
        wb.step(lead, 0.0, dy / n, carry=carry)
        wb.step(trail, 0.0, dy / n, carry=carry)


def _inner(side: str) -> float:
    """A hand's inner slot in a bin (toward the middle of the body)."""
    return SLOTS[1] if side == "r" else SLOTS[0]


def _refine(wb, eye, plane, colour, guess) -> np.ndarray:
    """Look straight at where a part should be; the estimate from there (the nearest blob of its colour)."""
    est = np.array([guess[0], guess[1], plane])
    wb.look(est, dur=0.5)
    wb.wait(0.25)
    near = [b for b in eye.find(wb.d, plane) if b.colour == colour and np.linalg.norm(b.world[:2] - est[:2]) < 0.04]
    if near:
        est[:2] = min(near, key=lambda b: np.linalg.norm(b.world[:2] - est[:2])).world[:2]
    return est


def _pick(wb, side, est, yaw: float | None = None) -> None:
    from pioneer_whole_body import skills as S
    p, r = S.knob_pose(est, (0.5 if side == "r" else -0.5) if yaw is None else yaw)
    wb.fingers(side, "lid_pre", dur=0.5)
    wb.reach(side, p + [0, 0, 0.15], r, 0.9)          # down from above it (not sweeping past its neighbours)
    wb.settle()
    wb.reach(side, p + [0, 0, S.CLEAR], r, 0.5)
    wb.settle()
    wb.reach(side, p, r, 0.6)
    wb.settle()
    wb.fingers(side, "lid", amount=S.SQUEEZE, dur=0.45)
    wb.settle()
    wb.wait(0.15)
    wb.reach(side, p + [0, 0, TOP_Z + 0.04], r, 0.9)
    wb.settle()


def _put(wb, side, xy, bottom_z, look) -> None:
    """Carry the held part over ``xy`` (an arc), its bottom down to ``bottom_z``, and let go. The lid grip draws the
    knob in to 20 mm out along the fingers from the palm: that is where the part hangs."""
    from pioneer_whole_body import skills as S
    p, r = wb.hand_goal[side]
    held = 0.02 * r[:2, 0]
    goal = np.array([xy[0] - held[0], xy[1] - held[1], bottom_z + TOP_Z + S.GRASP_ABOVE])
    wb.look(look, dur=0.5)
    clear = T + TRAY_H + 2 * TOP_Z + S.GRASP_ABOVE + 0.06      # the held part's bottom over the others' tops
    top = max(p[2], goal[2] + 0.05, clear)
    for target, dur in (([*p[:2], top], 0.4), ([*goal[:2], top], 1.0), (goal + [0, 0, 0.03], 0.5), (goal, 0.4)):
        wb.reach(side, target, r, dur)            # (each move finishes before the next: a new one replaces it)
        wb.settle()
    S.release(wb, side, up=0.07)
    wb.arm_joints_to(side, HIGH, 0.8)
    wb.settle()


def _hands(wb, pose, dur: float = 1.0) -> None:
    for side in ("l", "r"):
        wb.arm_joints_to(side, pose, dur)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["sort_bins"].plan = plan
