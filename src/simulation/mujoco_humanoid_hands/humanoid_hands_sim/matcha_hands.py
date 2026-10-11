"""Matcha for the five-finger humanoid: humanoid_mujoco_scenes' matcha (same balls, same cup, same four
checks) with its tools re-laid out for hands.

The gripper layout has every handle pointing up for a jaw coming from above. A hand holds a jug by a side
handle and pours by rolling the forearm, so here the scoop ("ladle") and the pitcher have a jug handle on
their +Y side (the left hand's outside) and the cup sits on their -Y side: held in a handshake grip and
rolled, a tool empties over its -Y lip. The cup is a mug with its handle on -Y, for the right hand, which
holds it while the left hand whisks and then serves it on the tray (on the right).

Not registered with humanoid_mujoco_scenes' @scene: its make_model attaches the Pioneer arm, and this
scene is for the humanoid (humanoid_hands_sim.robot.make_model).
"""
from __future__ import annotations
import math
import mujoco
import numpy as np
from humanoid_mujoco_scenes import add_floor
from humanoid_mujoco_scenes.matcha import scene as M

# A counter at 0.78 m (the gripper scene's table is 0.705 m): with the elbow no lower than 0.96 m, this
# humanoid holds a handle at 0.705 m only with its wrist bent past its range.
T = 0.78
CUP_POS = (0.30, 0.10)
LADLE_POS = (0.22, 0.42)      # out from under the left forearm when it pours
PITCHER_POS = (0.42, 0.20)
WHISK_POS = (0.48, 0.47)
TRAY = ((0.24, 0.36), (-0.27, -0.15))
# Side handles for a hand: a square post standing HANDLE_GAP off the container wall on a base plate, like a
# jug's handle. The fingers wrap the post through the gap; rolled about the forearm, the container tips over
# its far lip. (The gripper scene's post sits on the rim: a hand there curls its fingers into the container.)
HANDLE_GAP = 0.03
POST_H = 0.11                                  # post height above the base plate
GRASP_BELOW_TOP = 0.04                         # where the hand holds the post: index and little finger both on it
STEPS = M.STEPS
_U_STEP = M._U_STEP
ROT_Y = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]   # tool +X (its handle side) -> world +Y
ROT_NEG_Y = [math.cos(math.pi / 4), 0, 0, -math.sin(math.pi / 4)]  # tool +X -> world -Y


def step(model, data):
    """Track whisking and advance the latched step (call before every physics step)."""
    M._track_whisk(model, data)
    k = int(data.userdata[_U_STEP])
    if k < len(STEPS) and _step_done(model, data, k):
        data.userdata[_U_STEP] = k + 1


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _step_done(model, data, k):
    if k < 3:
        return M._step_done(model, data, k)
    cup = data.xpos[model.body("cup").id]
    (tx0, tx1), (ty0, ty1) = TRAY
    on_tray = tx0 < cup[0] < tx1 and ty0 < cup[1] < ty1 and cup[2] < T + 0.02
    kept = M.balls_in_cup(model, data, "powder") + M.balls_in_cup(model, data, "water")
    return on_tray and M.cup_upright(model, data) and kept >= M.SERVE_KEEP * (M.POWDER_NEEDED + M.WATER_NEEDED)


# Wide shot of the whole robot at the counter (MuJoCo free-camera fields).
CAMERA = dict(lookat=[0.22, 0.08, 0.98], distance=1.55, azimuth=205, elevation=-18)


def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    spec.nuserdata = 5
    world = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX
    H = M.HANDLE
    x0, x1 = M.TABLE_X
    world.add_geom(name="table", type=box, size=[(x1 - x0) / 2, M.TABLE_HALF_Y, T / 2], pos=[(x0 + x1) / 2, 0, T / 2],
                   rgba=[0.55, 0.42, 0.3, 1])
    (tx0, tx1), (ty0, ty1) = TRAY
    world.add_geom(name="tray", type=box, size=[(tx1 - tx0) / 2, (ty1 - ty0) / 2, 0.003],
                   pos=[(tx0 + tx1) / 2, (ty0 + ty1) / 2, T + 0.003], rgba=[0.15, 0.15, 0.17, 1])
    ceramic, wood = [0.95, 0.94, 0.9, 1], [0.6, 0.45, 0.3, 1]

    # Matcha scoop and water pitcher: a container with a side handle (see _jug).
    ladle = world.add_body(name="ladle", pos=[*LADLE_POS, T], quat=ROT_Y)
    ladle.add_freejoint(name="ladle")
    _jug(ladle, "ladle", r=M.LADLE_R, h=M.LADLE_H, wall=0.002, n=12, rgba=wood, mass=0.03, handle_rgba=wood)
    M._balls(world, "powder", LADLE_POS, M.LADLE_R, T + 0.002, **M.POWDER)

    P = M.PITCHER
    pitcher = world.add_body(name="pitcher", pos=[*PITCHER_POS, T], quat=ROT_Y)
    pitcher.add_freejoint(name="pitcher")
    _jug(pitcher, "pitcher", **P, rgba=[0.75, 0.85, 0.95, 0.5], handle_rgba=[0.3, 0.3, 0.32, 1])
    M._balls(world, "water", PITCHER_POS, P["r"], T + P["wall"], **M.WATER)

    holder = world.add_body(name="whisk_holder", pos=[*WHISK_POS, T])
    M._container(holder, "holder", r=0.022, h=0.05, wall=0.003, n=14, rgba=[0.3, 0.32, 0.35, 1])
    whisk = world.add_body(name="whisk", pos=[WHISK_POS[0], WHISK_POS[1], T + 0.003 + 0.05])
    whisk.add_freejoint(name="whisk")
    M._handle(whisk, "whisk_handle", 0.0, 0.10, wood, mass=0.02)
    for i in range(6):
        a = i * math.pi / 3
        mid = [0.013 * math.cos(a), 0.013 * math.sin(a), -0.025]
        for p0, p1 in (([0, 0, 0.0], mid), (mid, [0, 0, -0.048])):
            whisk.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*p0, *p1], size=[0.0012, 0, 0],
                           rgba=[0.75, 0.68, 0.5, 1], mass=0.0005)
    whisk.add_site(name="whisk_tip", pos=[0, 0, -0.03])

    # The cup (same size and checks as the gripper scene's) as a mug: its handle on the -Y side, for the right
    # hand. A plain 66 mm cup is shorter than the hand's finger span: only the little finger reaches it.
    cup = world.add_body(name="cup", pos=[*CUP_POS, T], quat=ROT_NEG_Y)
    cup.add_freejoint(name="cup")
    _jug(cup, "cup", **M.CUP, rgba=ceramic, handle_rgba=ceramic)


def _jug(body, name, r, h, wall, n, rgba, mass, handle_rgba):
    """Open container (inner radius r) with a side handle on its +X side: base plate, then a post HANDLE_GAP
    from the wall. Geom `{name}_handle` is the post."""
    M._container(body, name, r=r, h=h, wall=wall, n=n, rgba=rgba, mass=mass)
    H = M.HANDLE
    post_x = r + wall + HANDLE_GAP + H / 2
    body.add_geom(name=f"{name}_plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(post_x + H / 2) / 2, H / 2, wall / 2],
                  pos=[(post_x + H / 2) / 2, 0, wall / 2], rgba=handle_rgba, mass=0.005)
    body.add_geom(name=f"{name}_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[H / 2, H / 2, POST_H / 2],
                  pos=[post_x, 0, wall + POST_H / 2], rgba=handle_rgba, mass=0.015, friction=M.HANDLE_FRICTION, condim=4)
