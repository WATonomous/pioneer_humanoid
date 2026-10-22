"""Picking off a moving conveyor (timing, by sight): parts ride a belt past the robot, each in a carrier; the robot
sees each one coming with its head camera, measures how fast it moves, and takes it on the move, the hand riding
along with it, then drops it in the bin of its colour.

The robot walks up to the bench. On it:

  conveyor  a belt across the bench in front of the robot, running from its left to its right (from far off on the
            left: a long one), at 2.5 cm/s once the robot is there; four carriers on it, a knob-topped part in each,
            red or blue (which is which changes with every seed, ``reset``); a carrier that reaches the end stops there
            (its part missed)
  bins      behind the belt, on stands: red on the left, blue on the right (two places in each)

The plan, part by part as they come: look up the belt for the next coloured top (the head camera; each blob's ray
met with the plane of the knob tops), look again a moment later: where it is and how fast it moves. The hand of its
colour's side (left: red, right: blue) picks a time to take it a couple of seconds ahead, where the part will be
then; goes above that spot early, comes down along the belt with it (the hand moving at the belt's speed), closes
round the knob, lifts it out of its carrier while still moving with it, and carries it to its bin. Nothing about the
parts comes from the simulator: positions, speed and colours are read off the images.

Steps (``progress``): 1. sort the parts off the belt (n of 4 in their bins).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import READY
from pioneer_whole_body.scenes import sort_bins as SB
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

BELT_X = 0.50                          # the belt's middle
BELT_HW = 0.05                         # its half width
BELT_H = 0.03                          # its top above the bench
Y_FAR, Y_END = 1.60, -0.55             # where the belt starts (off to the left) and where the carriers stop
SPEED = 0.025                          # m/s
START = 9.0                            # s: the belt starts (the robot is at the bench)
N = 4
Y0 = [0.60 + 0.30 * i for i in range(N)]   # where the carriers start (clear of the arms coming up)
CARRIER_H = 0.006
POCKET, POCKET_H = 0.029, 0.008
BINS = {"red": (0.625, 0.20), "blue": (0.625, -0.20)}
BIN_IN, BIN_WALL, BIN_H = (0.05, 0.075), 0.006, 0.035
SLOT = 0.035                           # each bin's two places, either side of its middle (y)
BIN_Z = 0.05                           # the bins stand on blocks (low and that far out is past the arms' reach)
ZONE = {"l": (0.22, -0.10), "r": (-0.02, -0.30)}   # where each hand takes a part (y, from upstream)
LEAD = 2.4                             # s: from deciding to the grasp
IDLE = {**READY, "shoulder_roll": 1.45, "shoulder_pitch": -0.3}     # hands out to the sides, up
STEPS = ("sort the parts off the belt",)
_U_STEP = 0


def _carrier_y(t: float, i: int) -> float:
    return max(Y_END, Y0[i] - SPEED * max(0.0, t - START))


def _colour(model, i) -> str:
    rgba = model.geom_rgba[model.geom(f"part{i}_cap").id]
    return "red" if rgba[0] > rgba[2] else "blue"


def _sorted(model, data) -> int:
    n = 0
    for i in range(N):
        p = data.xpos[model.body(f"part{i}").id]
        bx, by = BINS[_colour(model, i)]
        if abs(p[0] - bx) < BIN_IN[0] and abs(p[1] - by) < BIN_IN[1] and p[2] < T + BIN_Z + BIN_H:
            n += 1
    return n


def step(model, data) -> None:
    """The belt: each carrier driven along it (a kinematic drive, as a belt is: nothing the robot does slows it)."""
    for i in range(N):
        j = model.joint(f"carrier{i}")
        a, v = j.qposadr[0], j.dofadr[0]
        data.qpos[a] = _carrier_y(data.time, i) - Y0[i]
        data.qvel[v] = -SPEED if START < data.time and _carrier_y(data.time, i) > Y_END else 0.0
    if _sorted(model, data) == N:
        data.userdata[_U_STEP] = 1


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 0:
        return 0, 1, f"sort the parts off the belt ({_sorted(model, data)} of {N} in their bins)"
    return 1, 1, STEPS[0]


def reset(model, data, rng) -> None:
    """Two red and two blue, in any order."""
    colours = ["red", "red", "blue", "blue"]
    rng.shuffle(colours)
    for i, c in enumerate(colours):
        model.geom_rgba[model.geom(f"part{i}_cap").id] = SB.COLOURS[c]


@scene("conveyor", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer  |  picking off a moving conveyor: seen coming, taken on the move, sorted")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 1
    world = spec.worldbody
    grey, dark = [0.32, 0.34, 0.37, 1], [0.12, 0.13, 0.15, 1]
    # the belt (its frame runs on past the bench's end, on legs)
    y0, y1 = Y_END - 0.08, Y_FAR + 0.08
    belt = world.add_body(name="belt", pos=[BELT_X, (y0 + y1) / 2, T])
    hy = (y1 - y0) / 2
    belt.add_geom(type=BOX, size=[BELT_HW, hy, BELT_H / 2], pos=[0, 0, BELT_H / 2], rgba=dark,
                  friction=[0.3, 0.01, 0.001])
    for s in (1, -1):
        belt.add_geom(type=BOX, size=[0.006, hy, BELT_H / 2 + 0.004], pos=[s * (BELT_HW + 0.006), 0, BELT_H / 2 + 0.004],
                      rgba=grey)
    for yy in np.arange(-hy + 0.05, hy, 0.12):
        belt.add_geom(type=BOX, size=[BELT_HW, 0.004, 0.0005], pos=[0, yy, BELT_H + 0.0005], rgba=[0.2, 0.21, 0.23, 1],
                      contype=0, conaffinity=0)                                    # (slats)
    if y1 > 0.65:                                                                    # legs off the bench
        for s in (1, -1):
            belt.add_geom(type=BOX, size=[0.015, 0.015, T / 2], pos=[s * (BELT_HW + 0.02), hy - 0.05, -T / 2],
                          rgba=grey, contype=0, conaffinity=0)
    # the carriers, a part in each
    for i in range(N):
        c = world.add_body(name=f"carrier{i}", pos=[BELT_X, Y0[i], T + BELT_H])
        c.add_joint(name=f"carrier{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], armature=1.0)
        c.add_geom(type=BOX, size=[0.04, 0.04, CARRIER_H / 2], pos=[0, 0, CARRIER_H / 2], rgba=[0.55, 0.57, 0.6, 1],
                   mass=0.2)
        for dx, dy, sx, sy in ((POCKET + 0.003, 0, 0.003, 0.014), (-POCKET - 0.003, 0, 0.003, 0.014),
                               (0, POCKET + 0.003, 0.014, 0.003), (0, -POCKET - 0.003, 0.014, 0.003)):
            c.add_geom(type=BOX, size=[sx, sy, POCKET_H / 2], pos=[dx, dy, CARRIER_H + POCKET_H / 2], rgba=grey,
                       mass=0.01)
        spec.add_exclude(bodyname1="belt", bodyname2=f"carrier{i}")
        SB._part(world, f"part{i}", [BELT_X, Y0[i], T + BELT_H + CARRIER_H + 0.0005], "red" if i % 2 else "blue")
    # the bins
    for colour, (bx, by) in BINS.items():
        stand = world.add_body(name=f"bin_{colour}_stand", pos=[bx, by, T])
        stand.add_geom(type=BOX, size=[BIN_IN[0], BIN_IN[1], BIN_Z / 2], pos=[0, 0, BIN_Z / 2], rgba=dark)
        b = world.add_body(name=f"bin_{colour}", pos=[bx, by, T + BIN_Z])
        hx, hy = BIN_IN
        b.add_geom(type=BOX, size=[hx + BIN_WALL, hy + BIN_WALL, 0.002], pos=[0, 0, 0.002], rgba=grey)
        for s in (-1, 1):
            b.add_geom(type=BOX, size=[BIN_WALL / 2, hy + BIN_WALL, BIN_H / 2],
                       pos=[s * (hx + BIN_WALL / 2), 0, BIN_H / 2], rgba=grey)
            b.add_geom(type=BOX, size=[hx, BIN_WALL / 2, BIN_H / 2], pos=[0, s * (hy + BIN_WALL / 2), BIN_H / 2],
                       rgba=grey)
        b.add_geom(type=BOX, size=[0.001, 0.035, 0.012], pos=[hx + BIN_WALL + 0.001, 0, BIN_H * 0.55],
                   rgba=SB.COLOURS[colour], contype=0, conaffinity=0)


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    eye = Eye(m)
    plane = T + BELT_H + CARRIER_H + SB.TOP_Z
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _idle(wb, ("l", "r"))
    wb.walk(0.30)
    notes["taken"] = []
    v = None                                      # the belt's speed, as measured
    while len(notes["taken"]) < N and wb.t < START + (Y0[-1] - Y_END) / SPEED:
        seen = _look(wb, eye, plane)
        if not seen:
            wb.wait(0.5)
            continue
        first = min(seen, key=lambda b: b.world[1])           # the next one down the belt
        side = "l" if first.colour == "red" else "r"
        hi, lo = ZONE[side]
        if v is None or first.world[1] - hi > v * (LEAD + 2.0):
            # not near the hand yet: watch it come (this also measures the belt's speed)
            t0, y0 = wb.t, first.world[1]
            wb.wait(0.8)
            again = [b for b in _look(wb, eye, plane) if b.colour == first.colour and 0 <= y0 - b.world[1] < 0.06]
            if again:
                u = (y0 - max(b.world[1] for b in again)) / (wb.t - t0)
                if u > 0.005:
                    v = u if v is None else 0.5 * (v + u)
            continue
        # near: where it is now, when and where to take it
        y = first.world[1]
        t_g = wb.t + max(LEAD, (y - hi) / v)
        y_g = y - v * (t_g - wb.t)
        if y_g < lo:
            raise S.TaskFailed(f"a {first.colour} part went past (y {y_g:.2f})")
        _take(wb, side, y_g, t_g, v, plane)
        bx, by = BINS[first.colour]
        n = sum(1 for c, *_ in notes["taken"] if c == first.colour)
        slot = by - math.copysign(SLOT, by) * (1 if n == 0 else -1)    # the slot nearer the middle first (the arm
        # (aimed 15 mm short: taken on the move, a part hangs further out along the fingers than ``_put`` allows)
        SB._put(wb, side, (bx - 0.015, slot), T + BIN_Z + 0.004, look=(bx, by, T + BIN_Z + 0.03))  # comes in over the other)
        _idle(wb, (side,))
        notes["taken"].append([first.colour, round(y_g, 3), round(v, 4)])
    wb.look([0.6, 0.0, T + 0.05], dur=0.6)
    wb.wait(1.0)
    notes["sorted"] = _sorted(m, d)
    if _sorted(m, d) < N:
        raise S.TaskFailed(f"{_sorted(m, d)} of {N} parts in their bins")


def _idle(wb, sides) -> None:
    """Hands out to the sides, up: clear of the parts going by and of the camera's view of the belt."""
    for side in sides:
        wb.arm_joints_to(side, IDLE, 0.8)
    wb.settle()


def _look(wb, eye, plane) -> list:
    """The coloured tops on the belt, upstream of the end of the hands' zones (the head camera)."""
    wb.look([BELT_X, 0.15, T], dur=0.3)
    wb.settle()
    return [b for b in eye.find(wb.d, plane)
            if b.colour in BINS and abs(b.world[0] - BELT_X) < BELT_HW + 0.02 and b.world[1] > ZONE["r"][1]]


def _take(wb, side, y_g, t_g, v, plane) -> None:
    """Take the part that will be at ``y_g`` at ``t_g``: above it early, down along the belt with it, close, lift
    out of the carrier while still going with it."""
    from pioneer_whole_body import skills as S
    yaw = 0.5 if side == "r" else -0.5
    top = np.array([BELT_X, y_g, plane])
    p, r = S.knob_pose(top, yaw)
    dn, cl, up = 0.6, 0.45, 0.6                      # down, close, lift (s)
    wb.fingers(side, "lid_pre", dur=0.4)
    t_above = t_g - dn - 0.2
    wb.look(top, dur=0.4)
    wb.reach(side, p + [0, v * (t_g - t_above), S.CLEAR], r, max(0.6, t_above - wb.t - 0.3))
    wb.settle()
    wb.wait(t_above - wb.t)
    wb.reach(side, p + [0, v * (t_g - wb.t), S.CLEAR], r, 0.2)
    wb.settle()
    wb.reach(side, p, r, t_g - wb.t)                 # (down while the part comes on under it)
    wb.settle()
    wb.reach(side, p - [0, v * cl, 0], r, cl)
    wb.fingers(side, "lid", amount=S.SQUEEZE, dur=cl)
    wb.settle()
    q, _ = wb.hand_goal[side]
    wb.reach(side, q + [0, -v * up, SB.TOP_Z + 0.04], r, up)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["conveyor"].plan = plan

