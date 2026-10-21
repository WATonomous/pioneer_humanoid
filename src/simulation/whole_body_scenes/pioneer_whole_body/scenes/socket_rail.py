"""Socket rail (a workshop scene, by sight): four loose 3/4" drive sockets of different sizes stand on a rubber bench mat,
in any order; behind them, an empty socket rail with a stud for each. The robot measures each socket with its head
camera (the width of its red size band, put back on the bench) and puts them on the rail smallest to largest, left to
right as it sees them: each taken straight up by its mouth ("lid" grasp) with the hand on its side and set down over
its stud; one that has to cross the middle is set down in the middle of the mat by one hand and taken on by the other.

The props are CAD models (``assets/workshop/make_meshes.py``, build123d) in the workshop room (``workshop.py``).
The robot starts at the bench: no walking.

Steps (``progress``): 1. measure the sockets, 2. put them on the rail (n of 4 in order).
"""
from __future__ import annotations

import importlib.util
import math

import mujoco
import numpy as np
from scipy import ndimage

from pioneer_whole_body import scene
from pioneer_whole_body.controller import READY, SHAPES
from pioneer_whole_body.studio import BENCH_TOP_Z
from pioneer_whole_body.workshop import ASSETS, MESH, add_workshop, mesh, paint, texture

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

_spec = importlib.util.spec_from_file_location("make_meshes", ASSETS / "make_meshes.py")
_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm)
SIZES = _mm.RAIL_SOCKETS                 # (size, outside diameter), smallest first
H = _mm.SOCKET_H
RAIL = _mm.RAIL

MAT = dict(x=(0.37, 0.53), y=0.30, h=0.004)
MAT_Z = T + MAT["h"]
SPOTS = (0.24, 0.13, -0.13, -0.24)       # where the sockets stand on the mat (y), at x = LOOSE_X
LOOSE_X = 0.455
MIDDLE = (0.425, 0.0)                    # where one is set down to change hands
RAIL_X = 0.585
STUD_Y = tuple((1.5 - k) * RAIL["pitch"] for k in range(len(SIZES)))   # left (+y) to right
SEAT_Z = T + RAIL["height"] + 0.002      # a socket's bottom on its stud's collar
YAW = {"l": -0.6, "r": 0.6}              # the fingers on a socket: turned in (each arm's easy pose)
GRIP_BACK = (0.015, 0.8)                 # the palm behind the socket's middle along the fingers: 15 mm for the biggest
                                         # (its far side is in reach), 0.8 mm more per mm narrower (as measured)
TUCKED = {**SHAPES["fist"], "th_c": 0.0, "th_f": 0.0, "th_mcp": 0.0, "th_ip": 0.0}   # fingers curled, thumb flat
RAISED = {**READY, "shoulder_pitch": -0.45, "shoulder_roll": 0.8}    # a hand up and out to its side
CARRY = 0.08                             # a carried socket's bottom this far over the bench
REACH = {"l": lambda y: y > -0.02, "r": lambda y: y < 0.02}     # (each hand on its own side, and the middle)
STEPS = ("measure the sockets", "put them on the rail")
_U_STEP, _U_SEEN = 0, 1


def _name(k: int) -> str:
    return f"rs_{SIZES[k][0]}"


def _on_rail(model, data) -> int:
    """How many studs have the socket of their place (the k-th smallest on the k-th from the left) sitting on them."""
    n = 0
    for k, y in enumerate(STUD_Y):
        p = data.xpos[model.body(_name(k)).id]
        up = data.xmat[model.body(_name(k)).id][8] > 0.97
        if abs(p[0] - RAIL_X) < 0.006 and abs(p[1] - y) < 0.006 and p[2] < SEAT_Z + 0.006 and up:
            n += 1
    return n


def step(model, data) -> None:
    if data.userdata[_U_SEEN] and data.userdata[_U_STEP] == 0:
        data.userdata[_U_STEP] = 1
    if data.userdata[_U_STEP] == 1 and _on_rail(model, data) == len(SIZES):
        data.userdata[_U_STEP] = 2


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 1:
        return 1, 2, f"put them on the rail ({_on_rail(model, data)} of 4)"
    return k, 2, STEPS[min(k, 1)]


def reset(model, data, rng) -> None:
    """The sockets in any order on the mat, each a little off its spot and turned any way."""
    for k, spot in enumerate(rng.permutation(len(SPOTS))):
        a = model.joint(_name(k)).qposadr[0]
        th = rng.uniform(0, 2 * math.pi)
        data.qpos[a:a + 3] = [LOOSE_X + rng.uniform(-0.004, 0.004), SPOTS[spot] + rng.uniform(-0.004, 0.004), MAT_Z]
        data.qpos[a + 3:a + 7] = [math.cos(th / 2), 0, 0, math.sin(th / 2)]


@scene("socket_rail", robot_pos=(0.0, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  socket rail: four sockets measured by sight and put on the rail in size order",
       camera=((0.48, 0.0, 0.95), 1.7, 60, -24))
def build(spec: mujoco.MjSpec) -> None:
    add_workshop(spec)
    spec.nuserdata = 2
    world = spec.worldbody
    vis = dict(contype=0, conaffinity=0, group=1)
    chrome = "chrome"
    red = paint(spec, "band_red", [0.80, 0.05, 0.04, 1], specular=0.3, shininess=0.4)
    rail_black = paint(spec, "rail_black", [0.05, 0.05, 0.06, 1], specular=0.6, shininess=0.7)
    cap_red = paint(spec, "cap_red", [0.75, 0.06, 0.05, 1], specular=0.3, shininess=0.4)
    # the rubber bench mat
    (x0, x1), hy = MAT["x"], MAT["y"]
    mat = texture(spec, "rubber_mat", repeat=(3, 5), specular=0.05, shininess=0.1)
    world.add_geom(type=BOX, size=[(x1 - x0) / 2, hy, MAT["h"] / 2], pos=[(x0 + x1) / 2, 0, T + MAT["h"] / 2],
                   material=mat, friction=[1.0, 0.02, 0.002])
    # the rail
    rail = world.add_body(name="rail", pos=[RAIL_X, 0, T])
    for m, mt in (("socket_rail", rail_black), ("rail_studs", chrome), ("rail_caps", cap_red)):
        mesh(spec, m)
        rail.add_geom(type=MESH, meshname=m, material=mt, **vis)
    rail.add_geom(type=BOX, size=[RAIL["width"] / 2, RAIL["length"] / 2, RAIL["height"] / 2],
                  pos=[0, 0, RAIL["height"] / 2], rgba=[0, 0, 0, 0], group=3)
    for y in STUD_Y:
        rail.add_geom(type=CYL, size=[RAIL["stud_r"], RAIL["stud_h"] / 2],
                      pos=[0, y, RAIL["height"] + RAIL["stud_h"] / 2], rgba=[0, 0, 0, 0], group=3)
        rail.add_geom(type=CYL, size=[RAIL["stud_r"] + 0.003, 0.001], pos=[0, y, RAIL["height"] + 0.001],
                      rgba=[0, 0, 0, 0], group=3)
    # the sockets (placed by ``reset``): a solid top, the drive end four walls round the square hole (a stud fits in)
    for k, (af, od) in enumerate(SIZES):
        name = _name(k)
        b = world.add_body(name=name, pos=[LOOSE_X, SPOTS[k], MAT_Z])
        b.add_freejoint(name=name)
        for m, mt in ((name, chrome), (f"{name}_band", red)):
            mesh(spec, m)
            b.add_geom(type=MESH, meshname=m, material=mt, **vis)
        r, hole = od / 2, 0.0096
        col = dict(rgba=[0, 0, 0, 0], group=3, friction=[1.2, 0.02, 0.002], condim=4)
        b.add_geom(type=CYL, size=[r, (H - 0.014) / 2], pos=[0, 0, 0.014 + (H - 0.014) / 2], mass=0.18, **col)
        t = r - hole
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            b.add_geom(type=BOX, size=[t / 2 if dx else r * 0.72, t / 2 if dy else r * 0.72, 0.007],
                       pos=[dx * (hole + t / 2), dy * (hole + t / 2), 0.007], mass=0.02, **col)
        b.add_site(name=f"{name}_top", pos=[0, 0, H])


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    wb.start_ready()
    wb.fingers("l", "relaxed", dur=0.6)
    wb.fingers("r", "relaxed", dur=0.6)
    wb.look([0.52, 0.0, T + 0.02], dur=1.0)              # the mat and the rail: all in view from here
    wb.wait(0.6)
    # 1. measure: each red band's width, put back on the bench
    eye = Eye(m, width=1280, height=960)
    found = _measure(wb, eye)
    notes["measured_mm"] = {f"{y:+.2f}": round(w * 1000, 1) for y, w in found}
    if len(found) != len(SIZES):
        raise S.TaskFailed(f"saw {len(found)} sockets")
    d.userdata[_U_SEEN] = 1
    order = sorted(found, key=lambda f: f[1])            # smallest first: the rail's left (+y) stud first
    where = {k: _spot_body(m, d, y) for k, (y, _) in enumerate(order)}    # (which body: only for the hand's feel)
    # 2. onto the rail: the outer studs first (a hand then never reaches over a socket it has set)
    for k in sorted(range(len(SIZES)), key=lambda k: -abs(STUD_Y[k])):
        (y_src, width), body = order[k], where[k]
        y_dst = STUD_Y[k]
        hand = [s for s in ("l", "r") if REACH[s](y_src) and REACH[s](y_dst)]
        if hand:
            side = hand[0] if len(hand) == 1 else ("l" if y_dst > 0 else "r")
        else:                                            # across the middle: one hand sets it down there, the other
            side = "l" if y_src > 0 else "r"             # takes it on
            _take(wb, side, body, width)
            _put(wb, side, body, MIDDLE, MAT_Z)
            side = "r" if side == "l" else "l"
        _take(wb, side, body, width)
        _put(wb, side, body, (RAIL_X, y_dst), SEAT_Z)
    for side in ("l", "r"):
        wb.arm_joints_to(side, _ready(), 1.0)
    wb.settle()
    notes["on_rail"] = _on_rail(m, d)
    if _on_rail(m, d) != len(SIZES):
        raise S.TaskFailed(f"{_on_rail(m, d)} of 4 in place")


def _ready():
    from pioneer_whole_body.controller import READY, SHAPES
    return READY


def _measure(wb, eye) -> list:
    """The sockets as the head camera sees them: each red band's pixels put back on the plane of the band's middle;
    its width across the line of sight is the socket's diameter. Returns [(y, diameter)], left to right."""
    from pioneer_whole_body.vision import HUES, MIN_SAT, MIN_VAL, hsv
    img = eye.look(wb.d)
    h, s, v = hsv(img)
    mask = (s > MIN_SAT) & (v > MIN_VAL) & np.any([(h >= a) & (h < b) for a, b in HUES["red"]], axis=0)
    lab, n = ndimage.label(mask)
    z = MAT_Z + sum(_mm.BAND_Z) / 2
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 80:
            continue
        pts = np.array([eye.on_plane(wb.d, u + 0.5, w + 0.5, z) for u, w in zip(xs[::3], ys[::3])])
        if not (abs(np.median(pts[:, 0]) - LOOSE_X) < 0.05):                      # (the mat's row only)
            continue
        c = np.median(pts, axis=0)                       # its outline is as wide as it is across the line of sight
        ray = (c - wb.d.cam_xpos[eye.cam])[:2]
        across = np.array([-ray[1], ray[0]]) / np.linalg.norm(ray)
        w = pts[:, :2] @ across
        out.append((float(c[1]), float(w.max() - w.min())))
    return sorted(out, key=lambda f: -f[0])


def _spot_body(m, d, y) -> str:
    return min((_name(k) for k in range(len(SIZES))), key=lambda b: abs(d.xpos[m.body(b).id][1] - y))


def _take(wb, side: str, body: str, width: float, over: float = 0.09, above: float = 0.004) -> None:
    """Straight up by its mouth: the "lid" grasp from above (again, from where it is now, if it didn't come up)."""
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import palm_down
    for _ in range(3):
        r = palm_down(YAW[side])
        back = GRIP_BACK[0] + max(0.0, SIZES[-1][1] - width) * GRIP_BACK[1]
        p = S.site(wb, f"{body}_top") - r[:, 0] * back + [0, 0, above]
        wb.fingers(side, TUCKED, dur=0.4)                # (fingers curled, thumb flat: nothing hangs down on the way)
        q, _ = wb.palm_pose(side)
        for target, dur in (([q[0], q[1], T + 0.15], 0.5), ([p[0], q[1], T + 0.15], 0.5),   # (up, out, across: over
                            ([p[0], p[1], T + 0.15], 0.7), (p + [0, 0, over], 0.4)):         # the others; down)
            wb.reach(side, target, r, dur)
            wb.settle()
        wb.fingers(side, "lid_pre", dur=0.4)
        wb.settle()
        wb.reach(side, p, r, 0.6)
        wb.settle()
        wb.fingers(side, "lid", amount=S.SQUEEZE, dur=0.45)
        wb.settle()
        wb.wait(0.15)
        z0 = S.body_pos(wb, body)[2]
        wb.reach(side, p + [0, 0, 0.06], r, 0.6)
        wb.settle()
        if S.body_pos(wb, body)[2] - z0 > 0.04:
            return
        S.release(wb, side, up=0.05)
    raise S.TaskFailed(f"{body} not taken")


def _put(wb, side: str, body: str, xy, bottom_z: float, carry: float = None) -> None:
    """Over ``xy`` (clear of the studs), down till its bottom is at ``bottom_z`` (where it hangs from the palm is
    felt), let go slowly, straight up, back to READY."""
    from pioneer_whole_body import skills as S
    p, r = wb.hand_goal[side]
    off = S.body_pos(wb, body) - wb.palm_pose(side, False)[0]
    goal = np.array([xy[0], xy[1], bottom_z]) - off
    top = max(p[2], goal[2] + (T + (CARRY if carry is None else carry) - bottom_z))      # (carried with its bottom over the others' tops)
    for target, dur in (([*goal[:2], top], 1.0), (goal + [0, 0, 0.025], 0.5)):
        wb.reach(side, target, r, dur)
        wb.settle()
    q = goal + [0, 0, 0.025]
    for _ in range(3):                                   # hovering over it: where the socket is (felt), nudged on
        wb.wait(0.2)
        err = np.asarray(xy) - S.body_pos(wb, body)[:2]
        if np.linalg.norm(err) < 0.0008:
            break
        q = q + [*err, 0]
        shift = q[:2] - (goal[:2])
        if np.linalg.norm(shift) > 0.015:                # (a socket hanging crooked: don't chase it far)
            q[:2] = goal[:2] + shift / np.linalg.norm(shift) * 0.015
        wb.reach(side, q, r, 0.3)
        wb.settle()
    wb.reach(side, q + [0, 0, bottom_z - 0.002 - S.body_pos(wb, body)[2]], r, 0.7)
    wb.settle()
    wb.fingers(side, "lid_pre", dur=0.8)
    wb.settle()
    wb.wait(0.2)
    q, r = wb.hand_goal[side]
    for target, dur in ((q + [0, 0, 0.08], 0.6), (q + [-0.08, 0, 0.10], 0.6)):   # (up and back: READY's path would
        wb.reach(side, target, r, dur)                                       # sweep low through the mat)
        wb.settle()
    wb.fingers(side, "relaxed", dur=0.3)
    wb.arm_joints_to(side, RAISED, 0.8)                  # (home by way of up and out to the side: READY's joint path
    wb.settle()                                          # from here sweeps low across the mat)
    wb.arm_joints_to(side, _ready(), 0.7)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["socket_rail"].plan = plan
