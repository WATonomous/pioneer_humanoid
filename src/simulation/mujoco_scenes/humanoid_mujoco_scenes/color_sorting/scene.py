"""Put two loose blocks into their matching coloured bins, in any order.

Coordinates are from the robot's perspective: +X away from the robot, +Y to its left, +Z up.
Bins form a compact horizontal row behind two blocks on the robot's right.
Reset samples two distinct colours, shuffles the block slots, and randomises block yaw.
"""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene

# Simulation prop dimensions, not measured physical props.
CUBE_HALF_SIZE = 0.01  # 2 cm-wide footprint
CUBE_HALF_HEIGHT = 0.02  # 4 cm tall
CUBE_MASS = 0.05
COLOUR_POOL = {
    "red": [0.85, 0.2, 0.2, 1],
    "green": [0.2, 0.7, 0.3, 1],
    "blue": [0.2, 0.4, 0.85, 1],
    "yellow": [0.95, 0.85, 0.1, 1],
    "cyan": [0.1, 0.8, 0.85, 1],
    "magenta": [0.8, 0.15, 0.8, 1],
}
DEFAULT_PALETTE = ("red", "green")  # build-time appearance; reset samples a new palette
BIN_INNER = (0.09, 0.09)  # full X/Y lengths
BIN_WALL = 0.003          # floor and wall thickness
BIN_WALL_H = 0.01         # 1 cm rims leave clearance for the hand approaching cubes

# Layout mirrored across Y=0 for the right arm (negative Y).
# Compact bins run horizontally behind the cubes in the ego view.
# Outer size is 9.6 x 9.6 x 1.3 cm; bases sit directly on the table.
# Shift the row 1.5 cm away from the robot to clear pickup approaches on the raised table.
# The old table's sampled reach rectangle does not apply at the new workcell height.
# 12 cm centre spacing leaves 2.4 cm between the bin exteriors.
# Both pairs remain on the robot's right, including reset jitter.
BIN_POSITIONS = ((0.405, -0.24), (0.405, -0.12))
CUBE_POSITIONS = ((0.280, -0.12), (0.275, -0.24))
CUBE_JITTER = 0.004
INSTRUCTION = "put each coloured cube in the bin of the same colour"


def _box(body, name, lo, hi, rgba, **kw):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list((hi - lo) / 2),
                         pos=list((hi + lo) / 2), rgba=rgba, **kw)


def add_bin(world, bin_index: int, xy, rgba) -> None:
    """Fixed open bin made from five collision boxes; origin at its bottom centre."""
    body = world.add_body(name=f"bin_{bin_index}", pos=[*xy, TABLE_TOP_Z])
    x, y = np.asarray(BIN_INNER) / 2
    w, top = BIN_WALL, BIN_WALL + BIN_WALL_H
    _box(body, f"bin_{bin_index}_floor", [-x - w, -y - w, 0], [x + w, y + w, w], rgba)
    _box(body, f"bin_{bin_index}_near", [-x - w, -y - w, w], [-x, y + w, top], rgba)
    _box(body, f"bin_{bin_index}_far", [x, -y - w, w], [x + w, y + w, top], rgba)
    _box(body, f"bin_{bin_index}_right", [-x, -y - w, w], [x, -y, top], rgba)
    _box(body, f"bin_{bin_index}_left", [-x, y, w], [x, y + w, top], rgba)


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """Call after mj_resetData; slot spacing keeps even jittered cubes clear of each other."""
    # Ordered sampling without replacement also uniformly assigns the palette to bins.
    # cube_i and bin_i always share a sampled colour; the index is not a colour label.
    palette = rng.choice(list(COLOUR_POOL), size=len(BIN_POSITIONS), replace=False)
    for pair_index, sampled_colour in enumerate(palette):
        rgba = COLOUR_POOL[sampled_colour]
        model.geom(f"cube_{pair_index}").rgba[:] = rgba
        bin_id = model.body(f"bin_{pair_index}").id
        model.geom_rgba[model.geom_bodyid == bin_id] = rgba
    # Independent slot permutation gives an independent colour permutation for cubes.
    for pair_index, (x, y) in enumerate(rng.permutation(CUBE_POSITIONS)):
        adr = model.joint(f"cube_{pair_index}").qposadr[0]
        dx, dy = rng.uniform(-CUBE_JITTER, CUBE_JITTER, 2)
        yaw = rng.uniform(0, 2 * np.pi)
        data.qpos[adr:adr + 7] = [x + dx, y + dy, TABLE_TOP_Z + CUBE_HALF_HEIGHT,
                                np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def in_bin(model: mujoco.MjModel, data: mujoco.MjData, pair_index: int) -> bool:
    """Cube footprint inside its matching bin, nearly still and on the inner floor."""
    cube, bin_body = model.body(f"cube_{pair_index}").id, model.body(f"bin_{pair_index}").id
    rotation = data.xmat[bin_body].reshape(3, 3).T
    rel = rotation @ (data.xpos[cube] - data.xpos[bin_body])
    # Project the rotated cube half-extents into the bin frame.
    half = np.abs(rotation @ data.xmat[cube].reshape(3, 3)) @ np.array([CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_HEIGHT])
    # Cubes extend above these low rims; floor contact below still rejects hovering.
    if not (np.all(np.abs(rel[:2]) + half[:2] <= np.asarray(BIN_INNER) / 2)
            and rel[2] - half[2] >= BIN_WALL - 0.001):
        return False
    cube_geom, floor_geom = model.geom(f"cube_{pair_index}").id, model.geom(f"bin_{pair_index}_floor").id
    on_floor = any({int(c.geom1), int(c.geom2)} == {cube_geom, floor_geom} for c in data.contact)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, cube, velocity, 0)
    return bool(on_floor and np.linalg.norm(velocity[3:]) < 0.02 and np.linalg.norm(velocity[:3]) < 0.5)


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    """Live sorted count, not a latched sequence: removing a cube reduces progress."""
    count = len(CUBE_POSITIONS)
    return sum(in_bin(model, data, pair_index) for pair_index in range(count)), count, INSTRUCTION


@scene("color_sorting", camera=dict(lookat=[0.345, -0.12, TABLE_TOP_Z + 0.03], distance=0.5, azimuth=200, elevation=-35),
       reset=reset, progress=progress)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    world = spec.worldbody
    for pair_index, (colour, bin_xy, cube_xy) in enumerate(zip(DEFAULT_PALETTE, BIN_POSITIONS, CUBE_POSITIONS)):
        rgba = COLOUR_POOL[colour]
        add_bin(world, pair_index, bin_xy, rgba)
        cube = world.add_body(name=f"cube_{pair_index}", pos=[*cube_xy, TABLE_TOP_Z + CUBE_HALF_HEIGHT])
        cube.add_freejoint(name=f"cube_{pair_index}")
        cube.add_geom(name=f"cube_{pair_index}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_HEIGHT],
                      mass=CUBE_MASS, rgba=rgba, friction=[1.0, 0.02, 0.001], condim=4)
