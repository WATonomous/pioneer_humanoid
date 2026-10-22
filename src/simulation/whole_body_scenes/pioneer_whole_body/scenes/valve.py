"""Setting two valves by their gauges (closed loop, by sight): each valve's knob turns a gauge's needle; a yellow
mark on each gauge's face is where its needle should point. The robot reads the needle and the mark with its head
camera, turns the knob, reads again, and so on until they line up.

The robot walks up to the bench. On it:

  valves   two knobs on the bench, left and right (stiff: they stay where they're left); each turns its gauge's
           needle, half as far (a joint equality: the physics turns the needle)
  gauges   behind them, facing the robot: a white face, a red needle, a yellow mark at the rim; where each needle
           starts and where its mark is change with every seed (``reset``)
  lamp     on the right: comes up when both needles are on their marks

The plan, for each valve (the left hand the left one, the right hand the right one): look at the gauge, find the
needle's red pixels and the mark's yellow ones round the hub in the image, their angles about the hub; the error is
the angle between them. Twist the knob to turn the needle by that much (a stroke: grip, turn the hand, let go, turn
back; at most 50 degrees of knob a stroke), look again. How far the needle moves for a turn of the hand is guessed
first (half), then learnt from each stroke (the grip slips a little; the gauge's ratio isn't given): a feedback loop
on the image, nothing read from the simulator. Done when the needle reads within 3 degrees of its mark (the
lamp wants both within 6 in fact).

Steps (``progress``): 1. set the left valve, 2. set the right valve.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import READY, rot_z
from pioneer_whole_body.parts import KNOB, KNOB_H, KNOB_R, STEEL, quat_z
from pioneer_whole_body.scenes.weigh_sort import _project
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

SIDES = ("l", "r")
VALVE = {"l": (0.50, 0.14), "r": (0.50, -0.14)}
BOSS_H = 0.03
GAUGE_X, GAUGE_Z = 0.69, 0.14               # the gauges' hubs (x, height above the bench); y as their valve's
DIAL_R = 0.055
RATIO = 0.5                                 # needle per knob
MARK_R = 0.049                              # the mark's distance from the hub (just past the needle's tip)
TOL = math.radians(3)                       # on the mark, as read
OK = math.radians(6)                        # on the mark, in fact (the lamp)
LAMP = (0.62, -0.30)
STEPS = ("set the left valve", "set the right valve")
_U_STEP, _U_OK = 0, 1


def _err(model, data, side) -> float:
    """How far the needle is off its mark (rad; the simulator's truth, for ``step`` / ``progress``)."""
    needle = data.qpos[model.joint(f"needle_{side}").qposadr[0]]
    return float(_wrap(_mark(model, side) - needle))


def _mark(model, side) -> float:
    """The mark's angle on its face (0 up, + clockwise as the robot sees it), from where ``reset`` put it."""
    g = model.geom(f"mark_{side}")
    y, z = model.geom_pos[g.id][1], model.geom_pos[g.id][2]
    return math.atan2(-y, z)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def step(model, data) -> None:
    ok = [abs(_err(model, data, s)) < OK for s in SIDES]
    k = int(data.userdata[_U_STEP])
    if k < 2 and ok[k]:
        data.userdata[_U_STEP] = k + 1
    data.userdata[_U_OK] = all(ok) and k >= 1
    a = model.joint("lamp").qposadr[0]
    target = 0.0 if data.userdata[_U_OK] else -math.pi / 2
    data.qpos[a] += float(np.clip(target - data.qpos[a], -0.004, 0.004))
    data.qvel[model.joint("lamp").dofadr[0]] = 0.0


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k < 2:
        s = SIDES[k]
        return k, len(STEPS), f"{STEPS[k]} ({math.degrees(abs(_err(model, data, s))):.0f} deg off its mark)"
    return k, len(STEPS), STEPS[-1]


def reset(model, data, rng) -> None:
    """Where each needle starts (the knob's angle) and where its mark is (40 to 140 degrees away from it)."""
    for s in SIDES:
        knob = rng.uniform(-1.4, 1.4)
        data.qpos[model.joint(f"valve_{s}").qposadr[0]] = knob
        data.qpos[model.joint(f"needle_{s}").qposadr[0]] = RATIO * knob
        while True:
            m = rng.uniform(-math.radians(110), math.radians(110))
            if math.radians(40) < abs(m - RATIO * knob) < math.radians(140):
                break
        _set_mark(model, s, m)
    data.qpos[model.joint("lamp").qposadr[0]] = -math.pi / 2


def _set_mark(model, side, a: float) -> None:
    g = model.geom(f"mark_{side}").id
    model.geom_pos[g] = [-0.004, -MARK_R * math.sin(a), MARK_R * math.cos(a)]
    model.geom_quat[g] = [math.cos(a / 2), math.sin(a / 2), 0, 0]


@scene("valve", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer  |  setting two valves by their gauges: turned until the needles meet their marks")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 2
    world = spec.worldbody
    dark, grey = [0.16, 0.18, 0.22, 1], [0.32, 0.34, 0.37, 1]
    spec.add_material(name="needle_lit", rgba=[0.9, 0.1, 0.08, 1], emission=0.5)      # (the gauges are lit
    spec.add_material(name="mark_lit", rgba=[0.98, 0.8, 0.05, 1], emission=0.5)       # instruments)
    for s in SIDES:
        vx, vy = VALVE[s]
        base = world.add_body(name=f"valve_{s}_base", pos=[vx, vy, T])
        base.add_geom(type=CYL, size=[0.035, 0.004, 0], pos=[0, 0, 0.004], rgba=dark)
        base.add_geom(type=CYL, size=[KNOB_R, BOSS_H / 2, 0], pos=[0, 0, BOSS_H / 2], rgba=STEEL)
        knob = base.add_body(name=f"valve_{s}", pos=[0, 0, BOSS_H])
        knob.add_joint(name=f"valve_{s}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.02,
                       frictionloss=0.04, armature=0.002)
        knob.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[0, 0, KNOB_H / 2], rgba=KNOB, mass=0.015,
                      friction=[2.0, 0.03, 0.003], condim=4)
        for i in range(10):
            a = 2 * math.pi * i / 10
            knob.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                          pos=[KNOB_R * math.cos(a), KNOB_R * math.sin(a), KNOB_H / 2], quat=quat_z(a), rgba=KNOB,
                          contype=0, conaffinity=0, mass=0.0001)
        knob.add_geom(type=BOX, size=[KNOB_R * 0.8, 0.002, 0.0006], pos=[KNOB_R * 0.2, 0, KNOB_H + 0.0006],
                      rgba=[0.9, 0.9, 0.9, 1], contype=0, conaffinity=0, mass=0.0001)      # (a line on its top)
        knob.add_site(name=f"valve_{s}_top", pos=[0, 0, KNOB_H + 0.0012])
        spec.add_exclude(bodyname1=f"valve_{s}_base", bodyname2=f"valve_{s}")
        # its gauge
        post = world.add_body(name=f"gauge_{s}", pos=[GAUGE_X, vy, T])
        post.add_geom(type=BOX, size=[0.01, 0.01, GAUGE_Z / 2], pos=[0.012, 0, GAUGE_Z / 2], rgba=dark)
        post.add_geom(type=CYL, size=[DIAL_R + 0.006, 0.004, 0], pos=[0.002, 0, GAUGE_Z], quat=[0.7071, 0, 0.7071, 0],
                      rgba=dark, contype=0, conaffinity=0)                                        # the bezel
        face = post.add_body(name=f"face_{s}", pos=[0, 0, GAUGE_Z])
        face.add_geom(type=CYL, size=[DIAL_R, 0.003, 0], quat=[0.7071, 0, 0.7071, 0], rgba=[0.95, 0.95, 0.93, 1],
                      contype=0, conaffinity=0)
        face.add_geom(name=f"mark_{s}", type=BOX, size=[0.001, 0.005, 0.005], pos=[-0.004, 0, MARK_R],
                      material="mark_lit", contype=0, conaffinity=0)
        needle = face.add_body(name=f"needle_{s}", pos=[-0.006, 0, 0])
        needle.add_joint(name=f"needle_{s}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=0.0005,
                         armature=0.0005)
        needle.add_geom(type=BOX, size=[0.0012, 0.003, 0.021], pos=[0, 0, 0.021], material="needle_lit",
                        contype=0, conaffinity=0, mass=0.01)
        needle.add_geom(type=CYL, size=[0.006, 0.002, 0], pos=[-0.001, 0, 0], quat=[0.7071, 0, 0.7071, 0],
                        rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0, mass=0.005)
        needle.gravcomp = 1.0
        spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name=f"gauge_{s}", name1=f"needle_{s}", name2=f"valve_{s}",
                          data=[0, RATIO, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    post = world.add_body(name="lamp_post", pos=[LAMP[0], LAMP[1], T])
    post.add_geom(type=CYL, size=[0.006, 0.05, 0], pos=[0, 0, 0.05], rgba=grey, contype=0, conaffinity=0)
    flag = post.add_body(name="lamp", pos=[0, 0, 0.1])
    flag.add_joint(name="lamp", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], ref=0.0)
    flag.add_geom(type=BOX, size=[0.003, 0.02, 0.03], pos=[0, 0, 0.03], rgba=[0.15, 0.85, 0.3, 1], contype=0,
                  conaffinity=0, mass=0.01)
    flag.gravcomp = 1.0


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    eye = Eye(m, 1280, 960)                          # (a gauge is small in the image: the full resolution)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    wb.walk(0.30)
    notes["strokes"], notes["errors"] = {}, {}
    for side in SIDES:
        notes["strokes"][side], notes["errors"][side] = _set(wb, eye, side)
        wb.arm_joints_to(side, READY, 0.9)
        wb.settle()
    wb.look([LAMP[0], LAMP[1], T + 0.12], dur=0.6)
    wb.wait(2.0)
    if not d.userdata[_U_OK]:
        off = {s: round(math.degrees(_err(m, d, s)), 1) for s in SIDES}
        raise S.TaskFailed(f"needles off their marks by {off} deg")


def _set(wb, eye, side: str, max_strokes: int = 14):
    """Read, turn, read again, until the needle's on its mark. Returns the strokes and the errors read."""
    from pioneer_whole_body import skills as S
    gain = RATIO                                     # needle (in the image) per knob: a first guess, then learnt
    errors = []
    needle, mark = _read(wb, eye, side)
    for k in range(max_strokes):
        e = _wrap(mark - needle)
        errors.append(round(math.degrees(e), 1))
        if abs(e) < TOL:
            return k, errors
        turn = float(np.clip(e / gain, -math.radians(50), math.radians(50)))
        _twist(wb, side, turn)
        new, mark = _read(wb, eye, side)
        moved = _wrap(new - needle)
        if abs(turn) > math.radians(15) and moved * turn > 0:
            gain = 0.5 * gain + 0.5 * moved / turn
        needle = new
    raise S.TaskFailed(f"{side} valve: {math.degrees(_wrap(mark - needle)):.0f} deg off after {max_strokes} strokes")


def _twist(wb, side: str, turn: float) -> None:
    """One stroke: grip the knob (fingertips round its rim), turn the hand ``turn`` (rad, + anticlockwise from above)
    about the knob's axis, let go (fingertips straight out, hand straight up). The hand starts from the end of its
    comfortable range that leaves room to turn that way."""
    from pioneer_whole_body import skills as S
    near, far = (0.15, 1.0) if side == "r" else (-0.15, -1.0)          # the hand's yaw range, palm down
    lo, hi = min(near, far), max(near, far)
    yaw0 = lo if turn > 0 else hi
    top = S.site(wb, f"valve_{side}_top")
    p, r = S.knob_pose(top, yaw0, S.TWIST_ABOVE)
    wb.fingers(side, "knob_pre", dur=0.3)
    wb.reach(side, p + [0, 0, 0.05], r, 0.6)
    wb.settle()
    wb.reach(side, p, r, 0.4)
    wb.settle()
    wb.fingers(side, "knob", amount=S.TWIST_SQUEEZE, dur=0.3)
    wb.settle()
    wb.reach(side, top + rot_z(turn) @ (p - top), rot_z(turn) @ r, 0.4 + 0.6 * abs(turn))
    wb.settle()
    wb.fingers(side, "knob_pre", dur=0.25)
    wb.settle()
    q, rr = wb.hand_goal[side]
    wb.reach(side, q + [0, 0, 0.06], rr, 0.3)
    wb.settle()


def _read(wb, eye, side: str):
    """The needle's and the mark's angles on the gauge's face (rad, 0 up, + clockwise as the robot sees it), from the
    head camera's image: each one's pixels (red, yellow) round the hub's pixel, put back on the face's plane (their
    rays met with it), the angle of their mean about the hub."""
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import hsv
    hub = np.array([GAUGE_X - 0.006, VALVE[side][1], T + GAUGE_Z])
    wb.look(hub, dur=0.5)
    wb.settle()
    wb.wait(0.3)
    img = eye.look(wb.d)
    h, s, v = hsv(img)
    c = _project(eye, wb.d, hub)
    rad = DIAL_R * eye.f / np.linalg.norm(hub - wb.d.cam_xpos[eye.cam])
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    r = np.hypot(xs - c[0], ys - c[1])
    out = []
    # the needle's outer part and the mark just past its tip: about the same distance from the hub, so the image's
    # slant skews both alike (and near the hub a pixel is many degrees)
    # (each on its own plane: the needle's face stands 2 mm proud of the mark's)
    for mask, r0, r1, x in ((((h > 340) | (h < 15)) & (s > 0.5) & (v > 0.25), 0.3, 0.85, GAUGE_X - 0.0072),
                            ((h > 42) & (h < 66) & (s > 0.5) & (v > 0.4), 0.5, 1.1, GAUGE_X - 0.005)):
        sel = mask & (r > r0 * rad) & (r < r1 * rad)
        if sel.sum() < 4:
            raise S.TaskFailed(f"{side} gauge: needle or mark not seen")
        # each pixel's ray met with the face's plane: the angle on the face itself, not the slanted image's
        cam_r, o = wb.d.cam_xmat[eye.cam].reshape(3, 3), wb.d.cam_xpos[eye.cam]
        u, v_ = xs[sel], ys[sel]
        dirs = cam_r @ np.stack([(u - eye.w / 2) / eye.f, -(v_ - eye.h / 2) / eye.f, -np.ones(len(u))])
        pts = o[:, None] + dirs * ((x - o[0]) / dirs[0])
        out.append(math.atan2(-(pts[1].mean() - hub[1]), pts[2].mean() - hub[2]))
    return out[0], out[1]


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["valve"].plan = plan
