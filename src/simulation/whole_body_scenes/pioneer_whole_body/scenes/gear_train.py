"""Gear train (long horizon, whole body: turning round and walking with a part in each hand): fetch two gears from
the parts table behind, put them on the gearbox's shafts so they mesh, turn the train by hand, test it.

The robot walks up to the bench. On it, the gearbox: a base plate with two shafts side by side (each on a collar),
and the test button with its lamp. Behind the robot, the parts table: the two gears, lying on stands.

Steps, each checked from the simulation and latched in order (``progress``):
  1. fetch the gears           turn round, walk to the parts table, a gear in each hand (by its hub knob)
  2. bring them to the bench   walk back and turn round, both gears held
  3. gear 1 on its shaft       right hand: lowered over the right shaft onto its collar
  4. gear 2 on its shaft       left hand: onto the left shaft, its teeth in gear 1's
  5. turn the gear train       right hand twists gear 1 by its hub (strokes): gear 2 turns the other way
  6. test it                   the test button pressed with the train assembled and run in: the lamp comes up
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY, rot_z
from pioneer_whole_body.parts import KNOB, KNOB_H, KNOB_R, ORANGE, STEEL, add_weld, quat_z
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

BOX_POS = (0.52, 0.0)                    # the gearbox base
SHAFT_Y = {1: -0.08, 2: 0.08}            # gear 1 right, gear 2 left; centre distance = 2 x pitch radius
COLLAR_R, COLLAR_H = 0.014, 0.03
PITCH_R = 0.08
ROOT_R = 0.074                           # the gear's disc (collides); the teeth are drawn to OUTER_R
OUTER_R = 0.087
TEETH = 20
GEAR_T = 0.012                           # disc thickness
SEAT_R = 0.005                           # a gear within this of its shaft's axis, flat on the collar: on its shaft
TABLE_X = (-1.20, -0.60)                 # the parts table, behind the robot
PART_X = -0.82
PART_Y = {1: 0.20, 2: -0.20}             # gear 1 on the robot's right once it has turned round
STAND_R, STAND_H = 0.03, 0.03
WALK = 0.30
BUTTON_POS = (0.60, -0.30)
BUTTON_TRAVEL = 0.008
RUN_IN = math.pi                         # gear 2 turned half a turn: run in

STEPS = ("fetch the gears", "bring them to the bench", "gear 1 on its shaft", "gear 2 on its shaft",
         "turn the gear train", "test it")
_U_STEP, _U_ON, _U_MESH_K, _U_TURN0, _U_HELD, _U_O1, _U_O2 = 0, 1, 2, 3, 4, 5, 6


def _gear_yaw(model, data, k) -> float:
    r = data.xmat[model.body(f"gear{k}").id].reshape(3, 3)
    return math.atan2(r[1, 0], r[0, 0])


def seated(model, data, k) -> bool:
    return bool(data.eq_active[model.equality(f"seat{k}").id])


def _try_seat(model, data, k) -> None:
    g = data.xpos[model.body(f"gear{k}").id]
    flat = data.xmat[model.body(f"gear{k}").id][8] > math.cos(math.radians(6))
    sx, sy = BOX_POS[0], SHAFT_Y[k]
    if math.hypot(g[0] - sx, g[1] - sy) < SEAT_R and abs(g[2] - (T + COLLAR_H)) < 0.004 and flat:
        e = model.equality(f"seat{k}").id
        piv = model.body(f"pivot{k}").id
        q, qn = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qn, data.xquat[piv])
        mujoco.mju_mulQuat(q, qn, data.xquat[model.body(f"gear{k}").id])
        model.eq_data[e, 0:3] = 0
        model.eq_data[e, 3:6] = data.xmat[piv].reshape(3, 3).T @ (g - data.xpos[piv])
        model.eq_data[e, 6:10] = q
        model.eq_data[e, 10] = 1.0
        data.eq_active[e] = 1


def turned(model, data) -> float:
    """How far gear 2 has turned since the train meshed (rad)."""
    if data.userdata[_U_HELD] < 2:
        return 0.0
    return abs(data.qpos[model.joint("pivot2").qposadr[0]] - data.userdata[_U_TURN0])


def step(model, data) -> None:
    for k in (1, 2):
        if not seated(model, data, k):
            _try_seat(model, data, k)
    if seated(model, data, 1) and seated(model, data, 2):
        # the mesh: gear 2 turns with gear 1, the other way, its teeth in gear 1's gaps (a rigid coupling; the
        # teeth themselves aren't collided)
        a1, a2 = model.joint("pivot1").qposadr[0], model.joint("pivot2").qposadr[0]
        v1, v2 = model.joint("pivot1").dofadr[0], model.joint("pivot2").dofadr[0]
        p = 2 * math.pi / TEETH
        if not data.userdata[_U_HELD]:                 # just meshed: each gear's yaw = its pivot angle + offset;
            th1, th2 = _gear_yaw(model, data, 1), _gear_yaw(model, data, 2)       # the nearest tooth phase
            data.userdata[_U_O1] = th1 - data.qpos[a1]
            data.userdata[_U_O2] = th2 - data.qpos[a2]
            data.userdata[_U_MESH_K] = round((th2 + th1 + p / 2) / p)
            data.userdata[_U_HELD] = 1
        th1 = data.qpos[a1] + data.userdata[_U_O1]
        data.qpos[a2] = -th1 - p / 2 + data.userdata[_U_MESH_K] * p - data.userdata[_U_O2]
        data.qvel[v2] = -data.qvel[v1]
        if data.userdata[_U_HELD] == 1:                # where gear 2 starts from (after the snap into mesh)
            data.userdata[_U_TURN0] = data.qpos[a2]
            data.userdata[_U_HELD] = 2
    run_in = data.userdata[_U_STEP] >= 5                # (latched: the train was turned half a turn)
    if data.qpos[model.joint("button").qposadr[0]] < -0.6 * BUTTON_TRAVEL and run_in:
        data.userdata[_U_ON] = 1
    a = model.joint("lamp").qposadr[0]
    target = 0.0 if data.userdata[_U_ON] else -math.pi / 2
    data.qpos[a] += float(np.clip(target - data.qpos[a], -0.004, 0.004))
    data.qvel[model.joint("lamp").dofadr[0]] = 0.0
    k = int(data.userdata[_U_STEP])
    if k < len(STEPS) and _step_done(model, data, k):
        data.userdata[_U_STEP] = k + 1


def _held_up(model, data, k) -> bool:
    return data.xpos[model.body(f"gear{k}").id][2] > T + STAND_H + 0.04


def _step_done(model, data, k) -> bool:
    if k == 0:
        return _held_up(model, data, 1) and _held_up(model, data, 2)
    if k == 1:
        g = [data.xpos[model.body(f"gear{i}").id] for i in (1, 2)]
        return all(p[0] > 0.2 and p[2] > T + 0.03 for p in g)
    if k == 2:
        return seated(model, data, 1)
    if k == 3:
        return seated(model, data, 2)
    if k == 4:
        return turned(model, data) > RUN_IN
    return bool(data.userdata[_U_ON])


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _gear(world, name, pos, yaw, rgba):
    g = world.add_body(name=name, pos=list(pos), quat=quat_z(yaw))
    g.add_freejoint(name=name)
    g.add_geom(type=CYL, size=[ROOT_R, GEAR_T / 2, 0], pos=[0, 0, GEAR_T / 2], rgba=rgba, mass=0.06,
               friction=[0.8, 0.02, 0.002])
    tooth = [0.006, 0.0085, GEAR_T / 2 * 0.98]
    for i in range(TEETH):
        a = 2 * math.pi * i / TEETH
        rr = (ROOT_R + OUTER_R) / 2
        g.add_geom(type=BOX, size=[(OUTER_R - ROOT_R) / 2 + 0.001, tooth[1] / 2, tooth[2]],
                   pos=[rr * math.cos(a), rr * math.sin(a), GEAR_T / 2], quat=quat_z(a), rgba=rgba,
                   contype=0, conaffinity=0, mass=0.0005)
    for i in range(5):           # spokes / lightening holes (dark marks)
        a = 2 * math.pi * i / 5 + 0.3
        g.add_geom(type=CYL, size=[0.013, 0.0006, 0], pos=[0.045 * math.cos(a), 0.045 * math.sin(a), GEAR_T + 0.0006],
                   rgba=[0.12, 0.13, 0.15, 1], contype=0, conaffinity=0, mass=0.0001)
    # the hub knob on top (the hands take the gear by it)
    z = GEAR_T
    g.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[0, 0, z + KNOB_H / 2], rgba=KNOB, mass=0.02,
               friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):
        a = 2 * math.pi * i / 10
        g.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                   pos=[KNOB_R * math.cos(a), KNOB_R * math.sin(a), z + KNOB_H / 2], quat=quat_z(a), rgba=KNOB,
                   contype=0, conaffinity=0, mass=0.0001)
    g.add_geom(type=CYL, size=[KNOB_R * 0.45, 0.0006, 0], pos=[0, 0, z + KNOB_H + 0.0006], rgba=ORANGE,
               contype=0, conaffinity=0, mass=0.0001)
    g.add_geom(type=BOX, size=[0.004, 0.012, 0.0007], pos=[0, 0.024, z + KNOB_H + 0.0007], rgba=[1, 1, 1, 1],
               contype=0, conaffinity=0, mass=0.0001)                       # a mark: shows it turning
    g.add_site(name=f"{name}_top", pos=[0, 0, z + KNOB_H])
    g.add_site(name=f"{name}_base", pos=[0, 0, 0])
    return g


@scene("gear_train", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress,
       title="Pioneer v2  |  gear train: fetch the gears from behind, mesh them on their shafts, run it, test")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec, half_y=0.6)
    add_bench(spec, x=TABLE_X, half_y=0.45, name="table")
    # a light over the parts table, the robot's front lit when it works there (no shadow: one shadow pass on CPU)
    spec.worldbody.add_light(name="table_light", pos=[-2.0, -1.2, 2.8], dir=[0.6, 0.35, -0.7],
                             diffuse=[0.34, 0.34, 0.32], specular=[0.1, 0.1, 0.1], castshadow=False, cutoff=55,
                             exponent=2)
    spec.nuserdata = 7
    world = spec.worldbody
    grey, dark = [0.32, 0.34, 0.37, 1], [0.16, 0.22, 0.36, 1]

    # the gearbox base: a plate, two collars, the shafts (drawn), the pivots the gears seat onto
    bx, by = BOX_POS
    base = world.add_body(name="gearbox", pos=[bx, by, T])
    base.add_geom(type=BOX, size=[0.11, 0.2, 0.004], pos=[0, 0, 0.004], rgba=dark)
    for k, y in SHAFT_Y.items():
        base.add_geom(type=CYL, size=[COLLAR_R, (COLLAR_H - 0.008) / 2, 0], pos=[0, y, 0.008 + (COLLAR_H - 0.008) / 2],
                      rgba=STEEL)
        base.add_geom(type=CYL, size=[0.006, 0.024, 0], pos=[0, y, COLLAR_H + 0.024], rgba=STEEL, contype=0,
                      conaffinity=0)                                  # the shaft (drawn: it goes up into the hub)
        piv = base.add_body(name=f"pivot{k}", pos=[0, y, COLLAR_H])
        piv.add_joint(name=f"pivot{k}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.01,
                      frictionloss=0.01, armature=0.002)
        piv.add_geom(type=CYL, size=[0.002, 0.001, 0], rgba=STEEL, contype=0, conaffinity=0, mass=0.001)
        piv.gravcomp = 1.0
        add_weld(spec, f"seat{k}", f"pivot{k}", f"gear{k}", active=False)

    # the parts table: the gears on stands
    colors = {1: [0.93, 0.47, 0.16, 1], 2: [0.2, 0.45, 0.75, 1]}
    for k, y in PART_Y.items():
        stand = world.add_body(name=f"stand{k}", pos=[PART_X, y, T])
        stand.add_geom(type=CYL, size=[STAND_R, STAND_H / 2, 0], pos=[0, 0, STAND_H / 2], rgba=grey)
        _gear(world, f"gear{k}", [PART_X, y, T + STAND_H + 0.0005], 0.0, colors[k])

    # test button and its lamp
    bx_, by_ = BUTTON_POS
    housing = world.add_body(name="button_housing", pos=[bx_, by_, T])
    housing.add_geom(type=BOX, size=[0.03, 0.03, 0.02], pos=[0, 0, 0.02], rgba=grey)
    housing.add_geom(type=CYL, size=[0.024, 0.003, 0], pos=[0, 0, 0.043], rgba=[0.95, 0.8, 0.1, 1])
    cap = housing.add_body(name="button", pos=[0, 0, 0.046])
    cap.add_joint(name="button", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-BUTTON_TRAVEL, 0],
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=150.0, springref=0.0, damping=3.0)
    cap.add_geom(type=CYL, size=[0.02, 0.006, 0], pos=[0, 0, 0.006], rgba=[0.1, 0.65, 0.2, 1], mass=0.02)
    cap.add_site(name="button_top", pos=[0, 0, 0.012])
    spec.add_exclude(bodyname1="button_housing", bodyname2="button")
    post = world.add_body(name="lamp_post", pos=[bx + 0.13, by - 0.17, T])
    post.add_geom(type=CYL, size=[0.006, 0.05, 0], pos=[0, 0, 0.05], rgba=grey, contype=0, conaffinity=0)
    flag = post.add_body(name="lamp", pos=[0, 0, 0.1])
    flag.add_joint(name="lamp", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], ref=0.0)
    flag.add_geom(type=BOX, size=[0.003, 0.02, 0.03], pos=[0, 0, 0.03], rgba=[0.15, 0.85, 0.3, 1], contype=0,
                  conaffinity=0, mass=0.01)
    flag.gravcomp = 1.0


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S

    m, d = wb.m, wb.d
    d.qpos[m.joint("lamp").qposadr[0]] = -math.pi / 2
    wb.look([BOX_POS[0], 0.0, T], dur=0.8)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _hands(wb, HIGH, 0.8)
    wb.walk(WALK)
    wb.look([BOX_POS[0], 0.0, T + 0.05], dur=0.6)
    wb.wait(0.6)

    # 1. turn round to the parts table, walk up, a gear in each hand
    wb.look([0.0, 0.0, 1.2], dur=0.4)
    wb.turn(math.pi)
    wb.look([PART_X, 0.0, T], dur=0.6)
    wb.walk(-WALK)
    back = math.pi                                   # facing -x: hand yaws are turned round
    S.pick_knob(wb, "r", "gear1_top", "gear1", yaw=back + 0.6, lift=0.08)
    S.pick_knob(wb, "l", "gear2_top", "gear2", yaw=back - 0.6, lift=0.08)

    # 2. back to the bench with them: walk back, turn round
    wb.look([0.0, 0.0, 1.2], dur=0.4)
    wb.walk(WALK, carry=True)
    wb.turn(-math.pi, carry=True)
    wb.look([BOX_POS[0], 0.0, T], dur=0.6)

    # 3. / 4. each gear over its shaft, down onto the collar
    for side, k in (("r", 1), ("l", 2)):
        seat = np.array([BOX_POS[0], SHAFT_Y[k], T + COLLAR_H])
        S.carry(wb, side, f"gear{k}_base", seat, clear=0.03, dur=1.4, over=0.04)
        S.align_held(wb, side, f"gear{k}", f"gear{k}_base", seat, yaw=None, rounds=3)
        if not S.lower_until(wb, side, 0.045, lambda k=k: seated(m, d, k), dur=1.2):
            raise S.TaskFailed(f"gear {k} not on its shaft")
        wb.wait(0.2)
        S.release(wb, side, up=0.08)
        wb.arm_joints_to(side, HIGH, 0.8)
        wb.settle()

    # 5. run the train in: twist gear 1 by its hub, stroke after stroke
    notes["strokes"] = _crank(wb, "r", "gear1", yaw0=1.1, yaw1=0.25, max_strokes=8)
    wb.arm_joints_to("r", READY, 1.0)
    wb.settle()

    # 6. test
    top = S.site(wb, "button_top")
    wb.look(top, dur=0.5)
    wb.fingers("r", "fist", dur=0.4)
    S.press(wb, "r", "r_thumb_tip", top)
    wb.fingers("r", "relaxed", dur=0.4)
    wb.arm_joints_to("r", READY, 1.0)
    wb.look([BOX_POS[0] + 0.13, BOX_POS[1] - 0.17, T + 0.12], dur=0.6)
    wb.wait(2.0)
    notes["final"] = dict(seated=[seated(m, d, 1), seated(m, d, 2)], turned=round(turned(m, d), 2),
                          on=bool(d.userdata[_U_ON]))


def _crank(wb, side: str, gear: str, yaw0: float, yaw1: float, max_strokes: int = 8) -> int:
    """Twist ``gear`` by its hub knob: grip at ``yaw0``, turn the hand to ``yaw1``, let go, back, regrip."""
    from pioneer_whole_body import skills as S
    m, d = wb.m, wb.d
    axis = d.xpos[m.body(gear).id].copy()
    for k in range(max_strokes):
        if turned(m, d) > RUN_IN:
            return k
        top = S.site(wb, f"{gear}_top")
        top[:2] = axis[:2]
        p, r = S.knob_pose(top, yaw0, S.TWIST_ABOVE)
        wb.look(top, dur=0.3)
        wb.fingers(side, "knob_pre", dur=0.3)
        wb.reach(side, p + [0, 0, 0.05], r, 0.6)
        wb.settle()
        wb.reach(side, p, r, 0.4)
        wb.settle()
        wb.fingers(side, "knob", amount=S.TWIST_SQUEEZE, dur=0.3)
        wb.settle()
        a = yaw1 - yaw0
        wb.reach(side, top + rot_z(a) @ (p - top), rot_z(a) @ r, 0.9)
        wb.settle()
        wb.fingers(side, "knob_pre", dur=0.25)
        wb.settle()
        p3, r3 = wb.hand_goal[side]
        wb.reach(side, p3 + [0, 0, 0.05], r3, 0.3)
        wb.settle()
    if turned(m, d) <= RUN_IN:
        raise S.TaskFailed(f"gear train turned only {turned(m, d):.2f} rad")
    return max_strokes


def _hands(wb, pose, dur: float = 1.0) -> None:
    for side in ("l", "r"):
        wb.arm_joints_to(side, pose, dur)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["gear_train"].plan = plan
