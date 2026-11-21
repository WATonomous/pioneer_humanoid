"""A movable red block and a fixed, hollow blue bowl in the lightbox workcell."""

import math

import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene
from pioneer_humanoid.mujoco_bimanual_arm import ARM_CONTYPE

BLOCK_HALF = 0.02
BLOCK_START = np.array([0.30, -0.23])
BLOCK_JITTER = 0.02
BOWL_XY = np.array([0.43, -0.32])
BOWL_RADIUS = 0.075
BOWL_THICKNESS = 0.008
BOWL_HEIGHT = 0.045
BOWL_FLOOR = 0.008
BOWL_SEGMENTS = 20
SETTLE_TIME = 0.3
MAX_LINEAR_SPEED = 0.02
MAX_ANGULAR_SPEED = 0.3
CONTACT_TOLERANCE = 0.001  # Allow small penetration from compliant contacts.

BLOCK_CORNERS = np.array([
    [x, y, z]
    for x in (-BLOCK_HALF, BLOCK_HALF)
    for y in (-BLOCK_HALF, BLOCK_HALF)
    for z in (-BLOCK_HALF, BLOCK_HALF)
])
_ANGLES = 2 * np.pi * np.arange(BOWL_SEGMENTS) / BOWL_SEGMENTS
WALL_NORMALS = np.column_stack((np.cos(_ANGLES), np.sin(_ANGLES)))


def reset(model, data, rng) -> None:
    """Randomize the block within +/-2 cm and clear its motion and settling timer."""
    joint = model.joint("pick_block_joint")
    qpos = int(joint.qposadr[0])
    dof = int(joint.dofadr[0])
    xy = BLOCK_START + rng.uniform(-BLOCK_JITTER, BLOCK_JITTER, 2)
    data.qpos[qpos:qpos + 7] = [*xy, TABLE_TOP_Z + BLOCK_HALF + 0.001, 1, 0, 0, 0]
    data.qvel[dof:dof + 6] = 0
    data.userdata[0] = -1.0


def block_inside_bowl(model, data) -> bool:
    """Require all eight rotated corners within the wall faces and below the rim."""
    geom = model.geom("pick_block_geom").id
    bowl = model.body("target_bowl").id
    corners = BLOCK_CORNERS @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    local = (corners - data.xpos[bowl]) @ data.xmat[bowl].reshape(3, 3)
    return bool(
        np.all(local[:, :2] @ WALL_NORMALS.T <= BOWL_RADIUS - BOWL_THICKNESS / 2)
        and np.all(local[:, 2] >= BOWL_FLOOR - CONTACT_TOLERANCE)
        and np.all(local[:, 2] <= BOWL_FLOOR + BOWL_HEIGHT)
    )


def _placed(model, data) -> bool:
    if not block_inside_bowl(model, data):
        return False
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                            model.body("pick_block").id, velocity, 0)
    if (np.linalg.norm(velocity[3:]) >= MAX_LINEAR_SPEED
            or np.linalg.norm(velocity[:3]) >= MAX_ANGULAR_SPEED):
        return False

    block = model.geom("pick_block_geom").id
    floor = model.geom("bowl_floor").id
    supported = False
    for contact in data.contact:
        if contact.dist > 0 or block not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == block else contact.geom1
        if model.geom_contype[other] & ARM_CONTYPE:
            return False
        normal_z = contact.frame[2] * (1 if contact.geom2 == block else -1)
        if other == floor and normal_z > 0.5:
            supported = True
    return supported


def step(model, data) -> None:
    """Track a continuous settling interval in this simulation's own state."""
    if not _placed(model, data):
        data.userdata[0] = -1.0
    elif data.userdata[0] < 0 or data.userdata[0] > data.time:
        data.userdata[0] = data.time


def success(model, data) -> bool:
    """True only while released, supported, and settled inside for 0.3 seconds."""
    started = data.userdata[0]
    return bool(started >= 0 and data.time - started >= SETTLE_TIME and _placed(model, data))


@scene(
    "block_bowl",
    reset=reset,
    step=step,
    success=success,
    camera=dict(
        lookat=[0.38, -0.28, TABLE_TOP_Z + 0.04],
        distance=0.75,
        azimuth=160,
        elevation=-40,
),
)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    spec.nuserdata = 1

    world = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX

    # Movable 4 cm block, weighing 50 g.
    block = world.add_body(
        name="pick_block",
        pos=[*BLOCK_START, TABLE_TOP_Z + BLOCK_HALF + 0.001],
    )
    block.add_freejoint(name="pick_block_joint")
    block.add_geom(
        name="pick_block_geom",
        type=box,
        size=[BLOCK_HALF] * 3,
        mass=0.05,
        friction=[1.0, 0.02, 0.001],
        condim=4,
        rgba=[0.9, 0.15, 0.1, 1],
    )

    # No joint: the bowl stays fixed on the table.
    bowl = world.add_body(
        name="target_bowl",
        pos=[*BOWL_XY, TABLE_TOP_Z],
    )
    blue = [0.15, 0.4, 0.9, 1]
    bowl.add_geom(
        name="bowl_floor",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, BOWL_FLOOR / 2],
        size=[BOWL_RADIUS + BOWL_THICKNESS / 2 + 0.001, BOWL_FLOOR / 2],
        rgba=blue,
    )

    radius = BOWL_RADIUS
    wall_thickness = BOWL_THICKNESS
    wall_height = BOWL_HEIGHT
    segments = BOWL_SEGMENTS
    # Slightly overlap neighboring segments so the collision ring has no gaps.
    wall_half_length = radius * math.tan(math.pi / segments) + 0.001
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        bowl.add_geom(
            name=f"bowl_wall_{i}",
            type=box,
            pos=[
                radius * math.cos(angle),
                radius * math.sin(angle),
                BOWL_FLOOR + wall_height / 2,
            ],
            size=[wall_thickness / 2, wall_half_length, wall_height / 2],
            quat=[math.cos(angle / 2), 0, 0, math.sin(angle / 2)],
            rgba=blue,
        )
