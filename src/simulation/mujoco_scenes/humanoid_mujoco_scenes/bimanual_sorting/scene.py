"""Sort coloured blocks across the table into matching trays."""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene
from pioneer_humanoid.mujoco_bimanual_arm import ARM_CONTYPE

COLORS = {"red": [0.85, 0.2, 0.15, 1], "blue": [0.15, 0.35, 0.85, 1]}
TRAYS = {"red": (0.38, -0.47), "blue": (0.38, 0.47)}
BLOCK_POSITIONS = ((0.33, 0.19), (0.40, 0.28))
BLOCK_HALF = 0.02
TRAY_HALF = 0.09
TRAY_WALL = 0.006
SETTLE_SECONDS = 0.3
MAX_LINEAR_SPEED = 0.01
MAX_ANGULAR_SPEED = 0.1
BLOCK_NAMES = tuple(f"{color}_{i}" for color in COLORS for i in range(2))


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """Start each colour on the opposite side from its tray."""
    data.userdata[:] = data.time
    for color, (_, target_y) in TRAYS.items():
        for i, (start_x, source_y) in enumerate(BLOCK_POSITIONS):
            start_y = -np.sign(target_y) * source_y
            offset_x, offset_y = rng.uniform(-0.015, 0.015, 2)
            position = [
                start_x + offset_x,
                start_y + offset_y,
                TABLE_TOP_Z + BLOCK_HALF,
            ]
            yaw = rng.uniform(-np.pi, np.pi)
            quaternion = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]  # MuJoCo order: w, x, y, z

            qpos_start = model.joint(f"{color}_{i}").qposadr[0]
            data.qpos[qpos_start:qpos_start + 3] = position
            data.qpos[qpos_start + 3:qpos_start + 7] = quaternion


def _placed_blocks(model: mujoco.MjModel, data: mujoco.MjData) -> set[str]:
    """Check bounds and motion like drawer_stow, then require released, supported placement."""
    placed = set()
    # Check lower blocks first so a stack's support is already known to be placed.
    for name in sorted(BLOCK_NAMES, key=lambda name: data.xpos[model.body(name).id, 2]):
        color = name.split("_")[0]
        geom = model.geom(f"{name}_geom").id
        pos = data.geom_xpos[geom]
        half_extent = np.abs(data.geom_xmat[geom].reshape(3, 3)) @ model.geom_size[geom]
        inside = np.all(np.abs(pos[:2] - TRAYS[color]) + half_extent[:2] <= TRAY_HALF - TRAY_WALL)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, model.body(name).id, velocity, 0)
        still = (np.linalg.norm(velocity[3:]) < MAX_LINEAR_SPEED
                 and np.linalg.norm(velocity[:3]) < MAX_ANGULAR_SPEED)
        if not (inside and still):
            continue

        supports = {model.geom(f"{color}_tray_floor").id}
        supports.update(model.geom(f"{lower}_geom").id for lower in placed if lower.startswith(f"{color}_"))
        supported, released = False, True
        for contact in data.contact:
            if geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == geom else contact.geom1
            if model.geom_contype[other] & ARM_CONTYPE:
                released = False
            normal_z = contact.frame[2] * (1 if contact.geom2 == geom else -1)
            if other in supports and normal_z > 0.5 and contact.pos[2] < pos[2]:
                supported = True
        if supported and released:
            placed.add(name)
    return placed


def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Remember the last time each block was not fully placed; resetData clears the timers."""
    placed = _placed_blocks(model, data)
    for i, name in enumerate(BLOCK_NAMES):
        if name not in placed:
            data.userdata[i] = data.time


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    placed = _placed_blocks(model, data)
    count = sum(int(name in placed and data.time - data.userdata[i] >= SETTLE_SECONDS)
                for i, name in enumerate(BLOCK_NAMES))
    return count, len(BLOCK_NAMES), "sort the blocks into matching trays"


@scene("bimanual_sorting", camera=dict(lookat=[0.05, 0.0, 0.99], distance=0.8, azimuth=180, elevation=-25),
       step=step, reset=reset, progress=progress)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    spec.nuserdata = len(BLOCK_NAMES)
    # Keep the observer camera inside the lightbox, clear of its back panel.
    spec.visual.global_.fovy = 75
    spec.worldbody.add_camera(name="front", pos=[0.775, 0.0, 1.328],
                              xyaxes=[0, 1, 0, -0.422618, 0, 0.906308], fovy=75)
    box = mujoco.mjtGeom.mjGEOM_BOX
    for color, (x, y) in TRAYS.items():
        tray = spec.worldbody.add_body(name=f"{color}_tray", pos=[x, y, TABLE_TOP_Z])
        tray.add_geom(name=f"{color}_tray_floor", type=box, size=[TRAY_HALF, TRAY_HALF, 0.003], pos=[0, 0, 0.003], rgba=COLORS[color])
        for axis in range(2):
            for sign in (-1, 1):
                pos = [0, 0, 0.016]
                size = [TRAY_HALF, TRAY_HALF, 0.016]
                pos[axis] = sign * (TRAY_HALF - TRAY_WALL / 2)
                size[axis] = TRAY_WALL / 2
                tray.add_geom(type=box, size=size, pos=pos, rgba=COLORS[color])
        for i, (start_x, source_y) in enumerate(BLOCK_POSITIONS):
            block = spec.worldbody.add_body(name=f"{color}_{i}",
                                           pos=[start_x, -np.sign(y) * source_y, TABLE_TOP_Z + BLOCK_HALF])
            block.add_freejoint(name=f"{color}_{i}")
            block.add_geom(name=f"{color}_{i}_geom", type=box, size=[BLOCK_HALF] * 3, mass=0.05,
                           friction=[1.0, 0.02, 0.001], condim=4, rgba=COLORS[color])
