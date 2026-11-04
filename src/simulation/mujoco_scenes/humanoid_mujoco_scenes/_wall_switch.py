"""Shared parts of the wall-switch scenes (toggle_switch, rocker_switch): a wall on the table, a wall plate, a hinged switch.

A wall stands on the table in front of the LEFT arm, facing the robot (the robot is at -X). A wall plate is
fixed on its face and the switch sits on the plate, hinged about the Y axis (left-right). Rotating the switch
by a positive angle moves its robot-side part up and the upper part away from the robot: ``on`` is a positive angle,
``off`` a negative one. A detent (a small torque in ``step``) snaps it to the nearer end and holds it there.
Every reset puts the switch off and moves the plate on the wall a little.
"""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_TOP_Z

WALL_FRONT_X = 0.50             # the wall's face toward the robot; inside the arm's reach with the gripper pointing forward
WALL_THICK = 0.02
WALL_HALF_Y = 0.15
WALL_HEIGHT = 0.32              # above the table
PLATE_SIZE = (0.004, 0.070, 0.115)   # x (thickness), y, z: a standard US wall plate (70 x 115 mm)
SWITCH_POS = (0.30, 0.15)       # y on the table, height above the table of the plate centre
JITTER = (0.03, 0.03)           # m, random shift of the plate (y, z) per reset

JOINT = "switch"
CAMERA = dict(lookat=[0.45, 0.30, TABLE_TOP_Z + 0.15], distance=0.7, azimuth=-35, elevation=-10)
# The same wall mirrored to the RIGHT arm (y < 0); the leader arm unit drives the right arm only.
CAMERA_RIGHT = dict(lookat=[0.45, -0.30, TABLE_TOP_Z + 0.15], distance=0.7, azimuth=35, elevation=-10)


def add_wall(spec: mujoco.MjSpec, side: float = 1.0) -> mujoco.MjsBody:
    """Backboard wall + wall plate (``side`` +1: left arm, -1: right arm). Returns the plate's body ("plate"; its origin is the centre of the plate's front face)."""
    world = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX
    world.add_geom(name="wall", type=box, size=[WALL_THICK / 2, WALL_HALF_Y, WALL_HEIGHT / 2],
                   pos=[WALL_FRONT_X + WALL_THICK / 2, side * SWITCH_POS[0], TABLE_TOP_Z + WALL_HEIGHT / 2],
                   rgba=[0.45, 0.55, 0.40, 1])
    px, py, pz = PLATE_SIZE
    plate = world.add_body(name="plate", pos=[WALL_FRONT_X - px, side * SWITCH_POS[0], TABLE_TOP_Z + SWITCH_POS[1]])
    plate.add_geom(name="plate", type=box, size=[px / 2, py / 2, pz / 2], pos=[px / 2, 0, 0], rgba=[0.95, 0.95, 0.92, 1])
    return plate


def add_hinge(spec: mujoco.MjSpec, plate: mujoco.MjsBody, limit: float) -> mujoco.MjsBody:
    """The switch body on the plate's front face with its hinge (about Y, +-``limit`` rad). The caller adds its geoms."""
    sw = plate.add_body(name="switch", pos=[0, 0, 0])
    sw.add_joint(name=JOINT, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.degrees(limit), np.degrees(limit)],
                 limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.005, armature=0.0003)   # MjSpec reads hinge ranges in degrees
    spec.add_exclude(bodyname1="plate", bodyname2="switch")   # it turns on its hinge, it does not hit its own plate
    return sw


def make_hooks(limit: float, threshold: float, torque: float, side: float = 1.0):
    """(reset, step, angle, is_on) for a switch with ends at +-``limit``, counted as on past ``threshold``, snapping with ``torque``."""
    def angle(model: mujoco.MjModel, data: mujoco.MjData) -> float:
        return float(data.qpos[model.joint(JOINT).qposadr[0]])

    def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
        """New episode: plate shifted on the wall, switch off. Call after mj_resetData."""
        dy, dz = rng.uniform(-1, 1, 2) * JITTER
        model.body_pos[model.body("plate").id, 1:] = [side * SWITCH_POS[0] + dy, TABLE_TOP_Z + SWITCH_POS[1] + dz]
        data.qpos[model.joint(JOINT).qposadr[0]] = -limit

    def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """The detent: push toward the nearer end. State lives in data (qfrc_applied), so a reset clears it."""
        data.qfrc_applied[model.joint(JOINT).dofadr[0]] = torque * np.sign(angle(model, data))

    def is_on(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        return angle(model, data) > threshold

    return reset, step, angle, is_on
