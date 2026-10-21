"""Sorting by weight, read off a dial (perception of a gauge): four parts that look the same, two light and two heavy
(which are which changes with every seed: ``reset``), are weighed one at a time on a spring scale and put in the
light or the heavy bin. The weight is read from the scale's dial with the head camera: the angle of its red needle
in the image (about its hub, projected into the image), against the reading with the scale empty.

The robot walks up to the bench. On it:

  tray     four pockets in a row, a part in each (the knob-topped parts of ``sort_bins``, all grey)
  scale    in the middle: a platform on a spring (a slide joint), its dial behind it, the needle coupled to the
           platform by a joint equality (the physics weighs the part)
  bins     light on the left, heavy on the right

The plan: read the empty scale; then for each part: the hand on its side takes it and sets it on the platform, the
hands go up out of the view, the robot reads the dial, and the hand on the bin's side takes it from the platform and
puts it in its bin.

Steps (``progress``): 1. weigh and sort the parts (n of 4 in the right bins).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY
from pioneer_whole_body.scenes import sort_bins as SB
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

N = 4
CELLS = [(0.505, y) for y in (0.25, 0.14, -0.14, -0.25)]
SCALE = (0.505, 0.0)
PLATFORM_Z = 0.016                       # the platform's top above the bench (empty)
SPRING = 1000.0                          # N/m
DIAL = (0.70, 0.0, 0.13)                 # the dial's hub (x, y, height above the bench); it faces the robot
DIAL_R = 0.055
NEEDLE = 600.0                            # rad of needle per m of platform travel
MASSES = (0.02, 0.10)                    # light, heavy (the part's base)
HEAVY = math.radians(30)                 # a reading this far past the empty one: heavy
BINS = {"light": (0.60, 0.21), "heavy": (0.60, -0.21)}
BIN_IN = (0.048, 0.085)
STEPS = ("weigh and sort the parts",)
_U_STEP = 0


def _heavy(model, i) -> bool:
    return model.body_mass[model.body(f"part{i}").id] > 0.1


def _sorted(model, data) -> int:
    n = 0
    for i in range(N):
        p = data.xpos[model.body(f"part{i}").id]
        bx, by = BINS["heavy" if _heavy(model, i) else "light"]
        if abs(p[0] - bx) < BIN_IN[0] and abs(p[1] - by) < BIN_IN[1] and p[2] < T + 0.03:
            n += 1
    return n


def step(model, data) -> None:
    if int(data.userdata[_U_STEP]) == 0 and _sorted(model, data) == N:
        data.userdata[_U_STEP] = 1


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 0:
        return 0, 1, f"weigh and sort the parts ({_sorted(model, data)} of {N} in the right bins)"
    return k, 1, STEPS[0]


def reset(model, data, rng) -> None:
    heavy = [True, True, False, False]
    rng.shuffle(heavy)
    _masses(model, heavy)
    mujoco.mj_setConst(model, data)                # (the constants that depend on the masses)


def _masses(model, heavy) -> None:
    for i, h in enumerate(heavy):
        b = model.body(f"part{i}").id
        m1 = MASSES[1] + 0.022 if h else MASSES[0] + 0.022
        model.body_inertia[b] *= m1 / model.body_mass[b]      # (the inertia scaled with it)
        model.body_mass[b] = m1


@scene("weigh_sort", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  sorting by weight: each part weighed, the dial read with its head camera")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 1
    world = spec.worldbody
    grey, dark = [0.32, 0.34, 0.37, 1], [0.16, 0.18, 0.22, 1]
    # the tray
    tray = world.add_body(name="tray", pos=[0, 0, T])
    tray.add_geom(type=BOX, size=[0.05, 0.31, SB.TRAY_H / 2], pos=[0.505, 0, SB.TRAY_H / 2], rgba=[0.2, 0.22, 0.25, 1])
    for cx, cy in CELLS:
        for dx, dy, sx, sy in ((SB.POCKET + 0.003, 0, 0.003, 0.014), (-SB.POCKET - 0.003, 0, 0.003, 0.014),
                               (0, SB.POCKET + 0.003, 0.014, 0.003), (0, -SB.POCKET - 0.003, 0.014, 0.003)):
            tray.add_geom(type=BOX, size=[sx, sy, SB.POCKET_H / 2], pos=[cx + dx, cy + dy, SB.TRAY_H + SB.POCKET_H / 2],
                          rgba=grey)
    for i, (x, y) in enumerate(CELLS):
        SB._part(world, f"part{i}", [x, y, T + SB.TRAY_H + 0.0005], "red")
        spec.geom(f"part{i}_cap").rgba = [0.6, 0.62, 0.66, 1]
    # the scale: a housing (cut away round the platform), the platform on its spring, the dial and its needle
    sx, sy = SCALE
    base = world.add_body(name="scale", pos=[sx, sy, T])
    base.add_geom(type=BOX, size=[0.06, 0.06, 0.004], pos=[0, 0, 0.004], rgba=dark)
    plat = base.add_body(name="platform", pos=[0, 0, PLATFORM_Z])
    plat.add_joint(name="platform", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.012, 0.002],
                   limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=SPRING, springref=0.0, damping=25.0)
    plat.add_geom(type=CYL, size=[0.045, 0.003, 0], pos=[0, 0, -0.003], rgba=[0.85, 0.86, 0.88, 1], mass=0.05)
    spec.add_exclude(bodyname1="scale", bodyname2="platform")
    dx, dy, dz = DIAL
    post = world.add_body(name="dial", pos=[dx, dy, T])
    post.add_geom(type=BOX, size=[0.01, 0.01, dz / 2], pos=[0.012, 0, dz / 2], rgba=dark)
    post.add_geom(type=CYL, size=[DIAL_R, 0.003, 0], pos=[0, 0, dz], quat=[0.7071, 0, 0.7071, 0],
                  rgba=[0.95, 0.95, 0.93, 1], contype=0, conaffinity=0)                         # the face
    for k in range(9):                                                                          # the scale marks
        a = math.radians(-20 + 15 * k)
        post.add_geom(type=BOX, size=[0.0008, 0.0012, 0.006], pos=[-0.004, -0.045 * math.sin(a), dz + 0.045 * math.cos(a)],
                      quat=[math.cos(a / 2), math.sin(a / 2), 0, 0], rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0)
    needle = post.add_body(name="needle", pos=[-0.005, 0, dz])
    needle.add_joint(name="needle", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=0.0005, armature=0.0005)
    needle.add_geom(type=BOX, size=[0.0012, 0.0025, 0.021], pos=[0, 0, 0.021], rgba=[0.88, 0.1, 0.08, 1],
                    contype=0, conaffinity=0, mass=0.01)
    needle.add_geom(type=CYL, size=[0.006, 0.002, 0], pos=[-0.001, 0, 0], quat=[0.7071, 0, 0.7071, 0],
                    rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0, mass=0.005)
    needle.gravcomp = 1.0
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name="gauge", name1="needle", name2="platform",
                      data=[0, -NEEDLE, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for name, (bx, by) in BINS.items():
        b = world.add_body(name=f"bin_{name}", pos=[bx, by, T])
        hx, hy = BIN_IN
        b.add_geom(type=BOX, size=[hx + 0.006, hy + 0.006, 0.002], pos=[0, 0, 0.002], rgba=grey)
        for s in (-1, 1):
            b.add_geom(type=BOX, size=[0.003, hy + 0.006, 0.0175], pos=[s * (hx + 0.003), 0, 0.0175], rgba=grey)
            b.add_geom(type=BOX, size=[hx, 0.003, 0.0175], pos=[0, s * (hy + 0.003), 0.0175], rgba=grey)
        mark = [0.15, 0.7, 0.3, 1] if name == "light" else [0.85, 0.2, 0.15, 1]
        b.add_geom(type=BOX, size=[0.001, 0.03, 0.01], pos=[hx + 0.007, 0, 0.02], rgba=mark, contype=0, conaffinity=0)


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    eye = Eye(m, 640, 480)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _hands(wb, HIGH, 0.8)
    wb.walk(0.30)
    zero = _read(wb, eye)
    notes["zero"] = round(math.degrees(zero), 1)
    notes["readings"] = []
    plane = T + SB.TRAY_H + SB.TOP_Z
    on_scale = T + PLATFORM_Z + SB.TOP_Z
    filled = {"l": 0, "r": 0}
    for i, (x, y) in enumerate(CELLS):
        part = f"part{i}"
        side = "l" if y > 0 else "r"
        _take(wb, side, part, np.array([x, y, plane]))
        _place(wb, side, part, SCALE, T + PLATFORM_Z + 0.002)
        wb.wait(1.5)                                       # (the scale settles)
        a = _read(wb, eye) - zero
        heavy = a > HEAVY
        notes["readings"].append(round(math.degrees(a), 1))
        bx, by = BINS["heavy" if heavy else "light"]
        side = "r" if heavy else "l"
        _take(wb, side, part, np.array([*S.body_pos(wb, part)[:2], on_scale]))
        slot = (0.04 if side == "r" else -0.04) * (1 if filled[side] == 0 else -1)   # the inner slot first
        filled[side] += 1
        _place(wb, side, part, (bx, by + slot), T + 0.008)
    _hands(wb, READY, 1.0)
    wb.look([0.6, 0, T + 0.05], dur=0.6)
    wb.wait(1.5)
    notes["final"] = _sorted(m, d)
    if _sorted(m, d) < N:
        raise S.TaskFailed(f"{_sorted(m, d)} of {N} in the right bins")


def _project(eye, data, p) -> np.ndarray:
    """A world point's pixel in the head camera's image."""
    c = eye.cam
    r = data.cam_xmat[c].reshape(3, 3)
    q = r.T @ (np.asarray(p) - data.cam_xpos[c])
    return np.array([eye.w / 2 + eye.f * q[0] / -q[2], eye.h / 2 - eye.f * q[1] / -q[2]])


def _read(wb, eye) -> float:
    """The needle's angle in the image (rad, from the hub, 0 straight up, + clockwise as the robot sees it)."""
    from pioneer_whole_body.vision import hsv
    hub = np.array([DIAL[0], DIAL[1], T + DIAL[2]])
    wb.look(hub, dur=0.6)
    wb.wait(0.3)
    img = eye.look(wb.d)
    h, s, v = hsv(img)
    red = (s > 0.5) & (v > 0.25) & ((h > 340) | (h < 15))
    c = _project(eye, wb.d, hub)
    ys, xs = np.nonzero(red)
    near = (xs - c[0]) ** 2 + (ys - c[1]) ** 2 < (0.6 * DIAL_R * eye.f / np.linalg.norm(hub - wb.d.cam_xpos[eye.cam])) ** 2 * 4
    xs, ys = xs[near], ys[near]
    if len(xs) < 5:
        return 0.0
    dx, dy = xs.mean() - c[0], ys.mean() - c[1]
    return math.atan2(dx, -dy)


def _take(wb, side, part, est) -> None:
    """Take the part by its knob (``sort_bins``' grasp); again from where it is now if it didn't come up."""
    from pioneer_whole_body import skills as S
    for _ in range(3):
        z0 = S.body_pos(wb, part)[2]
        SB._pick(wb, side, est, yaw=0.5 if side == "r" else -0.5)
        if S.body_pos(wb, part)[2] - z0 > 0.03:
            return
        S.release(wb, side, up=0.06)
        est = np.array([*S.body_pos(wb, part)[:2], est[2]])
    raise S.TaskFailed(f"{part} not taken")


def _place(wb, side, part, xy, bottom_z) -> None:
    """Carry the part over ``xy`` and set it down there (its bottom at ``bottom_z``): where it hangs from the palm is
    felt (the simulator's pose, as everywhere outside the vision scenes), so a part that hangs tipped or off centre
    in the grip still goes down where it should."""
    from pioneer_whole_body import skills as S
    p, r = wb.hand_goal[side]
    off = S.body_pos(wb, part) - wb.palm_pose(side, False)[0]           # the part's bottom from the palm
    goal = np.array([xy[0], xy[1], bottom_z]) - off
    wb.look((xy[0], xy[1], T + 0.02), dur=0.5)
    top = max(p[2], goal[2] + 0.05, T + SB.TRAY_H + 2 * SB.TOP_Z + S.GRASP_ABOVE + 0.06)
    for target, dur in (([*p[:2], top], 0.4), ([*goal[:2], top], 1.0), (goal + [0, 0, 0.03], 0.5)):
        wb.reach(side, target, r, dur)
        wb.settle()
    off = S.body_pos(wb, part) - wb.palm_pose(side, False)[0]           # (again, close to it)
    wb.reach(side, np.array([xy[0], xy[1], bottom_z - 0.002]) - off, r, 0.5)
    wb.settle()
    wb.fingers(side, "lid_pre", dur=0.9)              # let go slowly (a quick release flicks the part over), then
    wb.settle()                                       # straight up off it
    wb.wait(0.3)
    p, r = wb.hand_goal[side]
    wb.reach(side, p + [0, 0, 0.08], r, 0.8)
    wb.settle()
    wb.arm_joints_to(side, HIGH, 0.8)
    wb.settle()


def _hands(wb, pose, dur: float = 1.0) -> None:
    for side in ("l", "r"):
        wb.arm_joints_to(side, pose, dur)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["weigh_sort"].plan = plan
