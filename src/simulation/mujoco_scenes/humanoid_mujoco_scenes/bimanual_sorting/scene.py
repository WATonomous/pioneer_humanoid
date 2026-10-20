"""Sort coloured blocks across the table into matching trays."""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene

COLORS = {"red": [0.85, 0.2, 0.15, 1], "blue": [0.15, 0.35, 0.85, 1]}
TRAYS = {"red": (0.46, -0.42), "blue": (0.46, 0.42)}
BLOCK_X = (0.56, 0.70)
BLOCK_Y = 0.28
BLOCK_HALF = 0.02
TRAY_HALF = 0.09


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """Start each colour on the opposite side from its tray."""
    for color, (_, target_y) in TRAYS.items():
        start_y = -np.sign(target_y) * BLOCK_Y
        for i, start_x in enumerate(BLOCK_X):
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


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    count = 0
    for color, center in TRAYS.items():
        for i in range(2):
            pos = data.xpos[model.body(f"{color}_{i}").id]
            inside = np.all(np.abs(pos[:2] - center) < TRAY_HALF - BLOCK_HALF)
            resting = abs(pos[2] - (TABLE_TOP_Z + 0.006 + BLOCK_HALF)) < 0.01
            count += int(inside and resting)
    return count, 4, "sort the blocks into matching trays"


@scene("bimanual_sorting", camera=dict(lookat=[0.05, 0.0, 0.99], distance=0.8, azimuth=180, elevation=-25),
       reset=reset, progress=progress)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    # Keep the observer camera inside the lightbox, clear of its back panel.
    spec.visual.global_.fovy = 75
    spec.worldbody.add_camera(name="front", pos=[0.775, 0.0, 1.328],
                              xyaxes=[0, 1, 0, -0.422618, 0, 0.906308], fovy=75)
    box = mujoco.mjtGeom.mjGEOM_BOX
    for color, (x, y) in TRAYS.items():
        tray = spec.worldbody.add_body(name=f"{color}_tray", pos=[x, y, TABLE_TOP_Z])
        tray.add_geom(type=box, size=[TRAY_HALF, TRAY_HALF, 0.003], pos=[0, 0, 0.003], rgba=COLORS[color])
        for axis in range(2):
            for sign in (-1, 1):
                pos = [0, 0, 0.016]
                size = [TRAY_HALF, TRAY_HALF, 0.016]
                pos[axis] = sign * (TRAY_HALF - 0.003)
                size[axis] = 0.003
                tray.add_geom(type=box, size=size, pos=pos, rgba=COLORS[color])
        for i, start_x in enumerate(BLOCK_X):
            block = spec.worldbody.add_body(name=f"{color}_{i}",
                                           pos=[start_x, -np.sign(y) * BLOCK_Y, TABLE_TOP_Z + BLOCK_HALF])
            block.add_freejoint(name=f"{color}_{i}")
            block.add_geom(type=box, size=[BLOCK_HALF] * 3, mass=0.05,
                           friction=[1.0, 0.02, 0.001], condim=4, rgba=COLORS[color])
