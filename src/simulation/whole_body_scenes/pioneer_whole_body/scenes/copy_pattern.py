"""Copy a pattern (a goal shown, not told; by sight): a reference fixture at the back of the bench holds three
coloured parts in a row; the robot reads their order with its head camera and builds the same row in the empty
fixture in front of it, from a tray that holds those three colours and a fourth it doesn't need.

The robot walks up to the bench. On it:

  reference  at the back: three parts in a row, fixed, their tops three of red, green, blue, yellow
  fixture    in front of it: three empty pockets in a row, the same spacing
  tray       two pockets left of the fixture, two right: a part of each colour (where each is changes with every seed,
             as do the reference's colours and order, ``reset``)

The plan: look at the reference and read its row (the coloured blobs, each blob's ray met with the plane of its
tops, ordered along the row); look over the tray and find each colour there. Then for each pocket of the fixture, the
part of its reference's colour: taken with the hand on its side of the body, carried over and set in the pocket (if
it sits up on the pocket's rim, felt, it's taken again and set again). The fourth part stays in the tray. Then a last look at the fixture, to check its row against the reference's.

Steps (``progress``): 1. read the pattern, 2. build it (n of 3 in place).
"""
from __future__ import annotations

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.scenes import conveyor as CV
from pioneer_whole_body.scenes import sort_bins as SB
from pioneer_whole_body.scenes import weigh_sort as WS
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

PALETTE = {**SB.COLOURS, "yellow": [0.95, 0.8, 0.08, 1]}
COLOURS = tuple(PALETTE)
ROW_Y = (0.125, 0.0, -0.125)           # the rows' places (left to right as the robot sees them; this far apart:
                                       # a hand's thumb opening over one doesn't reach the next)
REF_X, FIX_X = 0.68, 0.50
TRAY = [(0.495, 0.21), (0.495, 0.31), (0.495, -0.21), (0.495, -0.31)]
STAND_H = 0.03                         # the reference stands on a block
STEPS = ("read the pattern", "build it")
_U_STEP, _U_SEEN = 0, 1


def _colour(model, name) -> str:
    rgba = model.geom_rgba[model.geom(f"{name}_cap").id]
    return min(PALETTE, key=lambda c: np.linalg.norm(np.asarray(PALETTE[c][:3]) - rgba[:3]))


def _built(model, data) -> int:
    """How many of the fixture's pockets hold the part of their reference's colour."""
    n = 0
    for k, y in enumerate(ROW_Y):
        want = _colour(model, f"ref{k}")
        for i in range(len(TRAY)):
            p = data.xpos[model.body(f"part{i}").id]
            upright = data.xmat[model.body(f"part{i}").id][8] > 0.95
            if _colour(model, f"part{i}") == want and abs(p[0] - FIX_X) < SB.POCKET and abs(p[1] - y) < SB.POCKET \
                    and p[2] < T + SB.TRAY_H + 0.006 and upright:
                n += 1
    return n


def step(model, data) -> None:
    if data.userdata[_U_SEEN] and data.userdata[_U_STEP] == 0:
        data.userdata[_U_STEP] = 1
    if data.userdata[_U_STEP] == 1 and _built(model, data) == len(ROW_Y):
        data.userdata[_U_STEP] = 2


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 1:
        return 1, 2, f"build it ({_built(model, data)} of 3 in place)"
    return k, 2, STEPS[min(k, 1)]


def reset(model, data, rng) -> None:
    """The reference: three of the four colours, in any order; the tray: the four, shuffled."""
    ref = list(rng.permutation(COLOURS))[:3]
    for k, c in enumerate(ref):
        model.geom_rgba[model.geom(f"ref{k}_cap").id] = PALETTE[c]
    for i, c in enumerate(rng.permutation(COLOURS)):
        model.geom_rgba[model.geom(f"part{i}_cap").id] = PALETTE[c]
        a = model.joint(f"part{i}").qposadr[0]
        x, y = TRAY[i]
        data.qpos[a:a + 2] = [x + rng.uniform(-0.002, 0.002), y + rng.uniform(-0.002, 0.002)]


def _pocket(body, x, y, z, rgba) -> None:
    for dx, dy, sx, sy in ((SB.POCKET + 0.003, 0, 0.003, 0.014), (-SB.POCKET - 0.003, 0, 0.003, 0.014),
                           (0, SB.POCKET + 0.003, 0.014, 0.003), (0, -SB.POCKET - 0.003, 0.014, 0.003)):
        body.add_geom(type=BOX, size=[sx, sy, SB.POCKET_H / 2], pos=[x + dx, y + dy, z + SB.POCKET_H / 2], rgba=rgba)


@scene("copy_pattern", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  copy a pattern: the reference row read by sight, built the same")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 2
    world = spec.worldbody
    plate, rim, dark = [0.2, 0.22, 0.25, 1], [0.32, 0.34, 0.37, 1], [0.12, 0.13, 0.15, 1]
    # the reference: a stand, three parts fixed on it
    ref = world.add_body(name="reference", pos=[REF_X, 0, T])
    ref.add_geom(type=BOX, size=[0.04, 0.185, STAND_H / 2], pos=[0, 0, STAND_H / 2], rgba=dark)
    for k, y in enumerate(ROW_Y):
        SB._part(ref, f"ref{k}", [0, y, STAND_H], "red")
    for k in range(len(ROW_Y)):                     # (fixed: no free joints)
        spec.delete(spec.joint(f"ref{k}"))
    # the fixture and the tray: plates with pockets
    fix = world.add_body(name="fixture", pos=[0, 0, T])
    fix.add_geom(type=BOX, size=[0.04, 0.17, SB.TRAY_H / 2], pos=[FIX_X, 0, SB.TRAY_H / 2], rgba=plate)
    for y in ROW_Y:
        _pocket(fix, FIX_X, y, SB.TRAY_H, [0.85, 0.85, 0.82, 1])
    tray = world.add_body(name="tray", pos=[0, 0, T])
    for s in (1, -1):
        tray.add_geom(type=BOX, size=[0.04, 0.09, SB.TRAY_H / 2], pos=[TRAY[0][0], s * 0.26, SB.TRAY_H / 2],
                      rgba=plate)
    for x, y in TRAY:
        _pocket(tray, x, y, SB.TRAY_H, rim)
    for i, (x, y) in enumerate(TRAY):
        SB._part(world, f"part{i}", [x, y, T + SB.TRAY_H + 0.0005], "red")


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    eye = Eye(m)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    for side in ("l", "r"):                       # (hands out to the sides: at READY the fingers hang over the tray)
        wb.arm_joints_to(side, CV.IDLE, 0.8)
    wb.settle()
    wb.walk(0.30)
    # 1. read the reference's row, find the tray's parts
    ref = _row(wb, eye, [REF_X, 0.0, T + STAND_H], T + STAND_H + SB.TOP_Z, lambda b: abs(b.world[0] - REF_X) < 0.05)
    notes["pattern"] = ref
    if len(ref) != len(ROW_Y):
        raise S.TaskFailed(f"read {ref} off the reference")
    d.userdata[_U_SEEN] = 1
    plane = T + SB.TRAY_H + SB.TOP_Z
    tray = {}
    for y in (0.26, -0.26):
        wb.look([0.495, y, T], dur=0.5)
        wb.settle()
        wb.wait(0.2)
        for b in eye.find(d, plane):
            if b.colour in PALETTE and abs(b.world[0] - 0.495) < 0.05 and 0.15 < abs(b.world[1]) < 0.37:
                tray.setdefault(b.colour, b.world)
    notes["tray"] = {c: [round(float(v), 3) for v in p[:2]] for c, p in tray.items()}
    missing = [c for c in ref if c not in tray]
    if missing:
        raise S.TaskFailed(f"didn't find {missing} in the tray")
    # 2. build it: from the outside in, so a hand never reaches over a part it set
    order = sorted(range(len(ROW_Y)), key=lambda k: -abs(ROW_Y[k]))
    for k in order:
        c = ref[k]
        est = SB._refine(wb, eye, plane, c, tray[c])
        side = "l" if est[1] > 0 else "r"
        part = _part_named(m, c)
        for _ in range(3):                          # (set down on a pocket's rim: take it again, set it again)
            WS._take(wb, side, part, est)
            WS._place(wb, side, part, (FIX_X, ROW_Y[k]), T + SB.TRAY_H + 0.002, up=0.16)   # (well up before
                                                                                       # the arm swings away: it'd clip a neighbour)
            wb.wait(0.3)
            if S.body_pos(wb, part)[2] < T + SB.TRAY_H + 0.003:
                break
            est = np.array([*S.body_pos(wb, part)[:2], S.site(wb, f"{part}_top")[2]])
            side = "l" if est[1] > 0 else "r"
        else:
            raise S.TaskFailed(f"the {c} part won't sit in its pocket")
    for side in ("l", "r"):
        wb.arm_joints_to(side, CV.IDLE, 0.8)
    wb.settle()
    # check: the fixture's row as the camera sees it, against the reference's
    built = _row(wb, eye, [FIX_X, 0.0, T], plane, lambda b: abs(b.world[0] - FIX_X) < 0.04)
    notes["built"] = built
    if built != ref:
        raise S.TaskFailed(f"built {built}, the pattern is {ref}")


def _row(wb, eye, at, plane, keep) -> list:
    """The colours of a row of parts, left to right as the robot sees them (one blob per place)."""
    wb.look(at, dur=0.6)
    wb.settle()
    wb.wait(0.3)
    found = {}
    for b in eye.find(wb.d, plane):
        if b.colour in PALETTE and keep(b):
            k = min(range(len(ROW_Y)), key=lambda k: abs(ROW_Y[k] - b.world[1]))
            if abs(ROW_Y[k] - b.world[1]) < 0.04 and (k not in found or b.area > found[k].area):
                found[k] = b
    return [found[k].colour for k in sorted(found)]


def _part_named(model, colour: str) -> str:
    """The tray part of that colour (for the hand's feel of it in the grip: ``weigh_sort``'s take and place)."""
    for i in range(len(TRAY)):
        if _colour(model, f"part{i}") == colour:
            return f"part{i}"
    raise KeyError(colour)


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["copy_pattern"].plan = plan
