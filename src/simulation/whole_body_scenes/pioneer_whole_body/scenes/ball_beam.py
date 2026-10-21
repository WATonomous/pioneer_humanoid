"""Ball and beam (continuous feedback control, both hands, by sight): the robot lifts a channel by the handles at its
ends and tilts it, one hand up and the other down, to roll the ball in it to a mark and hold it there; then sets it
down with the ball on the mark.

The robot walks up to the bench. On it:

  beam   a 44 cm channel lying in two cradles, a knob-topped handle at each end, a yellow mark on its walls somewhere
         along it and a red ball in it, somewhere else (both random per seed, ``reset``); its ends are closed
  lamp   on the right: comes up when the beam is back on its rests with the ball on the mark

The plan: take both handles (the firm wrap grip) and lift the beam 6 cm. Then, ten times a second: find the ball's
red and the mark's yellow in the head camera's image (each blob's ray met with the plane of the channel's floor),
the error between them along the beam and the ball's speed (from the last images); tilt the beam by
``KP * error - KD * speed`` (at most 4.5 degrees): the left hand goes down and the right up by the beam's half span
times its sine, or the other way. The ball rolls downhill (gravity and contact: nothing but the tilt moves it). When
the ball has stayed within 1 cm of the mark, nearly still, for 1.5 s: level, lower onto the rests, let go.

Steps (``progress``): 1. lift the beam, 2. bring the ball to the mark, 3. set it down.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH
from pioneer_whole_body.parts import KNOB, KNOB_H, KNOB_R, STEEL, quat_z
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

BEAM_X = 0.50
HALF_L = 0.22                       # half the channel's length
HANDLE_Y = 0.255                    # the handles (y, either side)
REST_Z = 0.04                       # the rests' tops above the bench
FLOOR_T, WALL_H, IN_W = 0.004, 0.016, 0.016     # the channel: floor thickness, wall height, half its inside width
BALL_R = 0.012
POST_H = 0.035                      # the handles' knobs stand on posts as wide as they are
LIFT = 0.06
KP, KD = 0.6, 0.5                   # tilt (rad) per m of error, per m/s of the ball's speed
TILT = math.radians(4.5)
NEAR, STILL, HOLD = 0.01, 0.012, 1.5
STEPS = ("lift the beam", "bring the ball to the mark", "set it down")
LAMP = (0.62, -0.36)
_U_STEP, _U_OK = 0, 1


def _ball_on_beam(model, data) -> float:
    """The ball's place along the beam (y in the beam's frame)."""
    b = data.xpos[model.body("ball").id]
    beam = model.body("beam").id
    return float((data.xmat[beam].reshape(3, 3).T @ (b - data.xpos[beam]))[1])


def _mark(model) -> float:
    return float(model.geom_pos[model.geom("mark_a").id][1])


def _resting(model, data) -> bool:
    return data.xpos[model.body("beam").id][2] < T + REST_Z + 0.004


def step(model, data) -> None:
    k = int(data.userdata[_U_STEP])
    on_mark = abs(_ball_on_beam(model, data) - _mark(model)) < 0.015
    if k == 0 and not _resting(model, data):
        data.userdata[_U_STEP] = 1
    elif k == 1 and on_mark and abs(data.qvel[model.joint("ball").dofadr[0] + 1]) < 0.02:
        data.userdata[_U_STEP] = 2
    elif k == 2 and on_mark and _resting(model, data):
        data.userdata[_U_STEP] = 3
    data.userdata[_U_OK] = k >= 2 and on_mark and _resting(model, data)
    a = model.joint("lamp").qposadr[0]
    target = 0.0 if data.userdata[_U_OK] else -math.pi / 2
    data.qpos[a] += float(np.clip(target - data.qpos[a], -0.004, 0.004))
    data.qvel[model.joint("lamp").dofadr[0]] = 0.0


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 1:
        e = 100 * abs(_ball_on_beam(model, data) - _mark(model))
        return 1, len(STEPS), f"bring the ball to the mark ({e:.0f} cm off)"
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def reset(model, data, rng) -> None:
    """Where the mark is and where the ball starts (at least 12 cm from it)."""
    mark = rng.uniform(-0.13, 0.13)
    while True:
        ball = rng.uniform(-HALF_L + 0.03, HALF_L - 0.03)
        if abs(ball - mark) > 0.12:
            break
    for g in ("mark_a", "mark_b"):
        model.geom_pos[model.geom(g).id][1] = mark
    a = model.joint("ball").qposadr[0]
    data.qpos[a:a + 3] = [BEAM_X, ball, T + REST_Z + FLOOR_T + BALL_R + 0.0005]
    data.qpos[model.joint("lamp").qposadr[0]] = -math.pi / 2


@scene("ball_beam", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  ball and beam: tilted with both hands to roll the ball to its mark, by sight")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 2
    world = spec.worldbody
    grey, dark = [0.55, 0.57, 0.6, 1], [0.16, 0.18, 0.22, 1]
    for s in (1, -1):
        rest = world.add_body(name=f"rest_{'l' if s > 0 else 'r'}", pos=[BEAM_X, s * 0.12, T])
        rest.add_geom(type=BOX, size=[0.03, 0.02, REST_Z / 2], pos=[0, 0, REST_Z / 2], rgba=dark)
        for sx in (1, -1):                          # a cradle: lips either side hold the beam in place
            rest.add_geom(type=BOX, size=[0.004, 0.02, 0.006], pos=[sx * (IN_W + 0.016), 0, REST_Z + 0.006],
                          rgba=dark)
    beam = world.add_body(name="beam", pos=[BEAM_X, 0, T + REST_Z])
    beam.add_freejoint(name="beam")
    beam.add_geom(type=BOX, size=[IN_W + 0.004, HALF_L, FLOOR_T / 2], pos=[0, 0, FLOOR_T / 2], rgba=grey, mass=0.06,
                  friction=[0.9, 0.01, 0.001])
    for s in (1, -1):
        beam.add_geom(type=BOX, size=[0.002, HALF_L, WALL_H / 2], pos=[s * (IN_W + 0.002), 0, FLOOR_T + WALL_H / 2],
                      rgba=grey, mass=0.01)
        beam.add_geom(type=BOX, size=[IN_W + 0.004, 0.002, WALL_H / 2], pos=[0, s * (HALF_L - 0.002),
                      FLOOR_T + WALL_H / 2], rgba=grey, mass=0.003)                              # (the closed ends)
        # an arm out to the handle, its post and knob
        y = s * HANDLE_Y
        beam.add_geom(type=BOX, size=[0.012, (HANDLE_Y - HALF_L) / 2 + 0.01, 0.003],
                      pos=[0, s * (HALF_L + HANDLE_Y) / 2, 0.003], rgba=dark, mass=0.005)
        beam.add_geom(type=CYL, size=[KNOB_R, POST_H / 2, 0], pos=[0, y, POST_H / 2], rgba=STEEL, mass=0.004)
        beam.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[0, y, POST_H + KNOB_H / 2], rgba=KNOB, mass=0.015,
                      friction=[2.0, 0.03, 0.003], condim=4)
        for i in range(10):
            a = 2 * math.pi * i / 10
            beam.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                          pos=[KNOB_R * math.cos(a), y + KNOB_R * math.sin(a), POST_H + KNOB_H / 2], quat=quat_z(a),
                          rgba=KNOB, contype=0, conaffinity=0, mass=0.0001)
        beam.add_site(name=f"handle_{'l' if s > 0 else 'r'}_top", pos=[0, y, POST_H + KNOB_H + 0.0012])
    for g, s in (("mark_a", 1), ("mark_b", -1)):                    # the mark: yellow on the walls' tops
        beam.add_geom(name=g, type=BOX, size=[0.003, 0.008, 0.0008], pos=[s * (IN_W + 0.002), 0.0,
                      FLOOR_T + WALL_H + 0.0008], rgba=[0.98, 0.8, 0.05, 1], contype=0, conaffinity=0, mass=0.0001)
    ball = world.add_body(name="ball", pos=[BEAM_X, 0.15, T + REST_Z + FLOOR_T + BALL_R + 0.0005])
    ball.add_freejoint(name="ball")
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], rgba=[0.88, 0.12, 0.1, 1], mass=0.03,
                  friction=[0.9, 0.005, 0.0001], condim=3)
    post = world.add_body(name="lamp_post", pos=[LAMP[0], LAMP[1], T])
    post.add_geom(type=CYL, size=[0.006, 0.05, 0], pos=[0, 0, 0.05], rgba=[0.32, 0.34, 0.37, 1], contype=0,
                  conaffinity=0)
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
    eye = Eye(m, 320, 240)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    for s in ("l", "r"):                            # (hands up: at READY the fingers would catch the handles)
        wb.arm_joints_to(s, HIGH, 0.8)
    wb.settle()
    wb.walk(0.30)
    S.pick_two(wb, {"l": ("handle_l_top", -0.6), "r": ("handle_r_top", 0.6)}, "beam", lift=LIFT, shape="lid",
               above=S.GRASP_ABOVE, squeeze=S.SQUEEZE)
    level = {s: wb.hand_goal[s] for s in ("l", "r")}
    plane = T + REST_Z + LIFT + FLOOR_T + BALL_R             # the ball's middle, and about the mark's height
    wb.look([BEAM_X, 0.0, plane], dur=0.6)
    wb.settle()
    held, prev, v, t_end = 0.0, None, 0.0, wb.t + 30.0
    notes["trace"] = []
    while wb.t < t_end:
        seen = _see(wb, eye, plane)
        if seen is None:
            wb.wait(0.1)
            continue
        ball, mark = seen
        if prev is not None:
            v = 0.6 * v + 0.4 * (ball - prev[0]) / (wb.t - prev[1])
        prev = (ball, wb.t)
        e = mark - ball
        if abs(e) < NEAR and abs(v) < STILL:
            held += 0.1
            if held >= HOLD:
                break
        else:
            held = 0.0
        tilt = float(np.clip(KP * e - KD * v, -TILT, TILT))
        _tilt(wb, level, tilt, 0.1)
        wb.wait(0.1)
        if len(notes["trace"]) < 400:
            notes["trace"].append(round(100 * e, 1))
    else:
        raise S.TaskFailed("the ball didn't settle on the mark")
    notes["time"] = round(wb.t - (t_end - 30.0), 1)
    # level, down onto the rests, let go
    _tilt(wb, level, 0.0, 0.4)
    wb.settle()
    for s in ("l", "r"):
        p, r = level[s]
        wb.reach(s, p - [0, 0, LIFT - 0.002], r, 1.2)
    wb.settle()
    for s in ("l", "r"):                            # let go slowly (a quick release jolts the beam: the ball rolls)
        wb.fingers(s, "lid_pre", dur=1.2)
    wb.settle()
    wb.wait(0.3)
    for s in ("l", "r"):
        p, r = wb.hand_goal[s]
        wb.reach(s, p + [0, 0, 0.07], r, 0.8)
    wb.settle()
    for s in ("l", "r"):                            # (up and in: a hand swung straight to READY drags a handle)
        wb.arm_joints_to(s, HIGH, 0.9)
    wb.settle()
    wb.look([LAMP[0], LAMP[1], T + 0.12], dur=0.6)
    wb.wait(2.0)
    notes["off"] = round(100 * (_ball_on_beam(m, d) - _mark(m)), 1)
    if not d.userdata[_U_OK]:
        raise S.TaskFailed(f"the ball ended {notes['off']} cm off its mark")


def _tilt(wb, level, tilt: float, dur: float) -> None:
    """Tilt the held beam by ``tilt`` (rad, + rolls the ball toward +y, the robot's left): the left hand down, the
    right up (or the other way), each palm turned with the beam."""
    c = np.array([sum(level[s][0][i] for s in ("l", "r")) / 2 for i in range(3)])
    rot = np.array([[1, 0, 0], [0, math.cos(tilt), math.sin(tilt)], [0, -math.sin(tilt), math.cos(tilt)]])
    for s in ("l", "r"):
        p, r = level[s]
        wb.reach(s, c + rot @ (p - c), rot @ r, dur)


def _see(wb, eye, plane):
    """The ball's and the mark's places along the beam (y), from the head camera: the red blob, the yellow blobs
    (one on each wall: their mean), each put on the plane of the ball's middle."""
    blobs = eye.find(wb.d, plane, min_area=4)
    red = [b for b in blobs if b.colour == "red" and abs(b.world[0] - BEAM_X) < 0.05 and abs(b.world[1]) < HALF_L + 0.02]
    yellow = [b for b in blobs if b.colour == "yellow" and abs(b.world[0] - BEAM_X) < 0.05 and abs(b.world[1]) < HALF_L]
    if not red or not yellow:
        return None
    ball = max(red, key=lambda b: b.area).world[1]
    mark = float(np.mean([b.world[1] for b in yellow]))
    return float(ball), mark


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["ball_beam"].plan = plan
