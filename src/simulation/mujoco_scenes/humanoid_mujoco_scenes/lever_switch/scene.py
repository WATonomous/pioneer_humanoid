"""Lever switch (table-top, the easy fallback): a toggle lever on a base, in front of the LEFT arm. Push the lever away from the robot to turn it on.

The lever hinges about the Y axis. Tilted toward the robot (q < 0) the switch is off, tilted away (q > 0) it is on.
A detent (a small torque in ``step``) pushes the lever to whichever end it is nearer, so it snaps and stays
there like a real toggle. Every reset puts it off and shifts the base a little.
``is_on``: the lever is tilted past ON_THRESHOLD (the dataset's success check holds this for a short time).
"""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene

BASE_POS = (0.38, 0.30)         # xy on the table; inside the left arm's gripper-down reach
BASE_SIZE = (0.09, 0.07, 0.02)  # x, y, z of the base block
BASE_JITTER = 0.03              # m, random shift of the base per reset (each axis)

LEVER_LENGTH = 0.10             # hinge to knob tip
LEVER_WIDTH = 0.018
KNOB_SIZE = (0.016, 0.03, 0.012)   # a wider knob at the tip: easy for the jaws to push
LEVER_MASS = 0.03
HINGE_LIMIT = 0.45              # rad, the two ends (off = -limit, on = +limit)
ON_THRESHOLD = 0.35             # rad, past this the switch counts as on
DETENT_TORQUE = 0.05            # N m, the snap toward the nearer end (about 0.5 N at the knob)


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """New episode: base shifted, lever off. Call after mj_resetData."""
    dx, dy = rng.uniform(-BASE_JITTER, BASE_JITTER, 2)
    model.body_pos[model.body("switch").id, :2] = [BASE_POS[0] + dx, BASE_POS[1] + dy]
    data.qpos[model.joint("lever").qposadr[0]] = -HINGE_LIMIT


def angle(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return float(data.qpos[model.joint("lever").qposadr[0]])


def is_on(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return angle(model, data) > ON_THRESHOLD


def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """The detent: push the lever toward the nearer end. State lives in data (qfrc_applied), so a reset clears it."""
    dof = model.joint("lever").dofadr[0]
    data.qfrc_applied[dof] = DETENT_TORQUE * np.sign(angle(model, data))


@scene("lever_switch", camera=dict(lookat=[0.38, 0.30, TABLE_TOP_Z + 0.06], distance=0.5, azimuth=200, elevation=-30),
       reset=reset, step=step, success=is_on)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    world = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX
    bx, by, bz = BASE_SIZE

    base = world.add_body(name="switch", pos=[BASE_POS[0], BASE_POS[1], TABLE_TOP_Z + bz / 2])
    base.add_geom(name="switch_base", type=box, size=[bx / 2, by / 2, bz / 2], rgba=[0.9, 0.9, 0.85, 1])

    # The lever hinges on the top of the base, about Y. Its geoms start just above the hinge and run up (+z).
    lever = base.add_body(name="lever", pos=[0, 0, bz / 2])
    lever.add_joint(name="lever", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.degrees(HINGE_LIMIT), np.degrees(HINGE_LIMIT)],   # MjSpec reads hinge ranges in degrees
                    limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.01, armature=0.0005)
    lever.add_geom(name="lever_stick", type=box, size=[LEVER_WIDTH / 2, LEVER_WIDTH / 2, LEVER_LENGTH / 2],
                   pos=[0, 0, 0.005 + LEVER_LENGTH / 2], mass=LEVER_MASS, friction=[1.0, 0.02, 0.001],
                   rgba=[0.8, 0.2, 0.2, 1])
    kx, ky, kz = KNOB_SIZE
    lever.add_geom(name="lever_knob", type=box, size=[kx / 2, ky / 2, kz / 2], pos=[0, 0, 0.005 + LEVER_LENGTH],
                   mass=0.01, friction=[1.0, 0.02, 0.001], rgba=[0.9, 0.3, 0.3, 1])
    spec.add_exclude(bodyname1="switch", bodyname2="lever")   # the lever turns on its hinge, it does not hit its own base
