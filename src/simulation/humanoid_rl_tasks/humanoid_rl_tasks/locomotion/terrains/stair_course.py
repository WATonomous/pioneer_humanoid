"""A straight ground → stairs up → raised walk → stairs down → ground course.

Unlike the stock pyramid generators, the origin is on the approach, not the
central platform. A positive-X traversal therefore encounters both stair flights
in the requested order. Every tread spans the full tile width, without holes.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


COURSE_SIZE = (16.0, 6.0)
COURSE_SPAWN_X = 0.9
COURSE_GOAL_X = 13.4
COURSE_GOAL_DISTANCE = COURSE_GOAL_X - COURSE_SPAWN_X


def course_goal_distance(cfg: MeshStairCourseTerrainCfg) -> float:
    """Forward displacement from spawn to one metre past the final tread."""
    return cfg.approach_length + 2.0 * cfg.num_steps * cfg.step_width + cfg.walkway_length + 1.0 - cfg.spawn_x


def stair_course_terrain(difficulty: float, cfg: MeshStairCourseTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build contiguous box meshes with matched ascending and descending risers.

    ``size`` and ``origin`` use Isaac Lab's local tile coordinates. The default
    course starts ascending at X=2 m, reaches a 2 m raised walkway at X=6.2 m,
    begins descending at X=8.2 m, and completes its last descent at X=12.05 m.
    The last descending tread (ending at X=12.4 m) is the ground itself. The
    returned origin is the ground-level spawn at (0.9, 3, 0), not the summit.
    """
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError("Stair-course difficulty must be finite and between 0 and 1.")
    if not isinstance(cfg.num_steps, int) or isinstance(cfg.num_steps, bool) or cfg.num_steps < 1:
        raise ValueError("Stair-course num_steps must be a positive integer.")

    size_x, size_y = cfg.size
    min_height, max_height = cfg.step_height_range
    dimensions = (
        size_x, size_y, min_height, max_height, cfg.step_width,
        cfg.approach_length, cfg.walkway_length, cfg.spawn_x,
    )
    if not all(math.isfinite(value) for value in dimensions):
        raise ValueError("Stair-course dimensions must all be finite.")
    if min_height < 0.0 or max_height < min_height:
        raise ValueError("Stair-course step_height_range must satisfy 0 <= min <= max.")
    if size_x <= 0.0 or size_y <= 1.0 or cfg.step_width <= 0.0 or cfg.walkway_length <= 0.0:
        raise ValueError("Stair-course tile, treads, and walkway must have positive usable dimensions.")
    # A half-metre margin supports the task's ground-only spawn jitter and avoids
    # putting the nominal spawn directly against the first vertical riser.
    if cfg.spawn_x < 0.5 or cfg.spawn_x > cfg.approach_length - 0.5:
        raise ValueError("Stair-course spawn_x must have 0.5 m flat clearance on both sides.")

    flight_length = cfg.num_steps * cfg.step_width
    walkway_start = cfg.approach_length + flight_length
    descending_start = walkway_start + cfg.walkway_length
    course_end = descending_start + flight_length
    if size_x < course_end + 1.0:
        raise ValueError("Stair-course tile must fit both flights, the walkway, and at least 1 m of exit ground.")

    step_height = min_height + difficulty * (max_height - min_height)
    summit_height = cfg.num_steps * step_height

    # Solid boxes terminate at Z=0; a continuous 20 cm-thick ground slab lies
    # below them. No negative-height final step or gaps are introduced.
    base = trimesh.creation.box(extents=(size_x, size_y, 0.2))
    base.apply_translation((0.5 * size_x, 0.5 * size_y, -0.1))
    meshes = [base]

    for index in range(cfg.num_steps):
        height = (index + 1) * step_height
        if height == 0.0:
            continue
        mesh = trimesh.creation.box(extents=(cfg.step_width, size_y, height))
        mesh.apply_translation((cfg.approach_length + (index + 0.5) * cfg.step_width,
                                0.5 * size_y, 0.5 * height))
        meshes.append(mesh)

    if summit_height > 0.0:
        walkway = trimesh.creation.box(extents=(cfg.walkway_length, size_y, summit_height))
        walkway.apply_translation((walkway_start + 0.5 * cfg.walkway_length,
                                   0.5 * size_y, 0.5 * summit_height))
        meshes.append(walkway)

    # The first descending tread is one riser below the walkway. The last is
    # ground-level and already covered by the base, so it needs no zero-size box.
    for index in range(cfg.num_steps - 1):
        height = (cfg.num_steps - index - 1) * step_height
        if height == 0.0:
            continue
        mesh = trimesh.creation.box(extents=(cfg.step_width, size_y, height))
        mesh.apply_translation((descending_start + (index + 0.5) * cfg.step_width,
                                0.5 * size_y, 0.5 * height))
        meshes.append(mesh)

    origin = np.array((cfg.spawn_x, 0.5 * size_y, 0.0), dtype=np.float64)
    return meshes, origin


@configclass
class MeshStairCourseTerrainCfg(SubTerrainBaseCfg):
    """Straight full-width flights and a level raised walkway between them.

    Difficulty changes riser height only; all horizontal geometry stays fixed.
    Spawn is safely before the first flight. Twelve 35 cm treads, 3–15 cm risers,
    and a 2 m walkway produce a summit 36–180 cm above the approach and exit.
    A (0, 0) height range yields a flat calibration tile with the same origin
    and course goal, omitting zero-volume step meshes entirely.
    """

    function = stair_course_terrain
    size: tuple[float, float] = COURSE_SIZE
    step_height_range: tuple[float, float] = (0.03, 0.15)
    step_width: float = 0.35
    num_steps: int = 12
    approach_length: float = 2.0
    walkway_length: float = 2.0
    spawn_x: float = COURSE_SPAWN_X
