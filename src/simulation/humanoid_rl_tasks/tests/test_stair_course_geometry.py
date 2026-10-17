"""Dependency-free numeric contracts for the straight up/walk/down generator.

The actual terrain function is executed with box-recording doubles in place of
NumPy/trimesh. These tests verify dimensions and top-surface profiles without
importing Isaac Lab, starting Kit, or touching CUDA. They do not replace mesh,
terrain-generator integration, spawn, or locomotion tests in the simulator.
"""

import ast
import __future__
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


GEOMETRY_FILE = (Path(__file__).resolve().parents[1]
                 / "humanoid_rl_tasks/locomotion/terrains/stair_course.py")


class _Box:
    """Record the exact extents/translations requested by the real function."""

    def __init__(self, extents):
        self.extents = tuple(extents)
        self.translation = (0.0, 0.0, 0.0)

    def apply_translation(self, translation):
        self.translation = tuple(a + b for a, b in zip(self.translation, translation))

    @property
    def bounds(self):
        return (tuple(center - extent / 2 for center, extent in zip(self.translation, self.extents)),
                tuple(center + extent / 2 for center, extent in zip(self.translation, self.extents)))


def _load_geometry():
    parsed = ast.parse(GEOMETRY_FILE.read_text(encoding="utf-8"), filename=str(GEOMETRY_FILE))
    config_class = next(node for node in parsed.body
                        if isinstance(node, ast.ClassDef) and node.name == "MeshStairCourseTerrainCfg")
    parsed.body = [node for node in parsed.body
                   if not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))]
    namespace = {
        "__name__": "stair_course_numeric_contract",
        "math": math,
        "np": SimpleNamespace(array=lambda values, **kwargs: tuple(values), float64=float),
        "trimesh": SimpleNamespace(creation=SimpleNamespace(box=lambda **kwargs: _Box(**kwargs))),
    }
    exec(compile(parsed, str(GEOMETRY_FILE), "exec", flags=__future__.annotations.compiler_flag), namespace)
    defaults = {}
    for node in config_class.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            defaults[node.target.id] = eval(compile(ast.Expression(node.value), str(GEOMETRY_FILE), "eval"),
                                           namespace)
    return namespace, defaults


def _surface_height(meshes, x, y=3.0):
    covering = [mesh.bounds[1][2] for mesh in meshes
                if mesh.bounds[0][0] <= x <= mesh.bounds[1][0]
                and mesh.bounds[0][1] <= y <= mesh.bounds[1][1]]
    if not covering:
        raise AssertionError(f"The course has no surface at ({x}, {y}).")
    return max(covering)


class StairCourseGeometryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace, cls.defaults = _load_geometry()

    def _cfg(self, **changes):
        values = deepcopy(self.defaults)
        values.update(changes)
        return SimpleNamespace(**values)

    def _build(self, difficulty=0.5, **changes):
        return self.namespace["stair_course_terrain"](difficulty, self._cfg(**changes))

    def test_default_spawn_is_on_the_approach_before_the_first_riser(self):
        meshes, origin = self._build()
        self.assertEqual(origin, (0.9, 3.0, 0.0))
        self.assertEqual(self.defaults["size"], (16.0, 6.0))
        self.assertEqual(_surface_height(meshes, origin[0], origin[1]), 0.0)
        for jitter in (-0.2, 0.0, 0.2):
            self.assertLess(origin[0] + jitter, self.defaults["approach_length"])
            self.assertEqual(_surface_height(meshes, origin[0] + jitter, origin[1] + jitter), 0.0)

    def test_traversal_is_twelve_risers_up_two_metre_walk_twelve_risers_down(self):
        meshes, _ = self._build(step_height_range=(0.08, 0.08))
        tread = self.defaults["step_width"]
        self.assertEqual(self.defaults["num_steps"], 12)
        self.assertEqual(len(meshes), 25)  # Base + twelve up + walkway + eleven raised down.
        self.assertEqual(_surface_height(meshes, 1.5), 0.0)
        for index in range(12):
            with self.subTest(flight="up", index=index):
                self.assertAlmostEqual(_surface_height(meshes, 2.0 + (index + 0.5) * tread),
                                       (index + 1) * 0.08)
            with self.subTest(flight="down", index=index):
                self.assertAlmostEqual(_surface_height(meshes, 8.2 + (index + 0.5) * tread),
                                       (11 - index) * 0.08)
        walkway = meshes[13]
        self.assertAlmostEqual(walkway.bounds[0][0], 6.2)
        self.assertAlmostEqual(walkway.bounds[1][0], 8.2)
        self.assertAlmostEqual(walkway.extents[0], 2.0)
        self.assertAlmostEqual(_surface_height(meshes, 7.2), 0.96)
        self.assertEqual(_surface_height(meshes, 12.7), 0.0)

    def test_every_vertical_transition_is_one_riser_in_the_correct_direction(self):
        meshes, _ = self._build(step_height_range=(0.08, 0.08))
        for index in range(12):
            up_x, down_x = 2.0 + index * 0.35, 8.2 + index * 0.35
            self.assertAlmostEqual(_surface_height(meshes, up_x + 1e-6)
                                   - _surface_height(meshes, up_x - 1e-6), 0.08)
            self.assertAlmostEqual(_surface_height(meshes, down_x + 1e-6)
                                   - _surface_height(meshes, down_x - 1e-6), -0.08)

    def test_ground_and_raised_boxes_are_full_width_positive_volume_and_in_tile(self):
        meshes, _ = self._build()
        base = meshes[0]
        self.assertEqual(base.bounds, ((0.0, 0.0, -0.2), (16.0, 6.0, 0.0)))
        for mesh in meshes[1:]:
            with self.subTest(bounds=mesh.bounds):
                self.assertTrue(all(extent > 0.0 for extent in mesh.extents))
                self.assertGreaterEqual(mesh.bounds[0][0], 0.0)
                self.assertLessEqual(mesh.bounds[1][0], 16.0)
                self.assertEqual(mesh.bounds[0][1], 0.0)
                self.assertEqual(mesh.bounds[1][1], 6.0)
                self.assertEqual(mesh.bounds[0][2], 0.0)

    def test_adjacent_raised_intervals_have_no_horizontal_gap(self):
        meshes, _ = self._build()
        for first, second in zip(meshes[1:], meshes[2:]):
            self.assertAlmostEqual(first.bounds[1][0], second.bounds[0][0])

    def test_difficulty_changes_only_riser_height_not_spawn_or_horizontal_layout(self):
        easiest, origin0 = self._build(0.0)
        middle, origin1 = self._build(0.5)
        hardest, origin2 = self._build(1.0)
        self.assertEqual(origin0, origin1)
        self.assertEqual(origin1, origin2)
        for meshes, summit in ((easiest, 0.36), (middle, 1.08), (hardest, 1.80)):
            self.assertAlmostEqual(max(mesh.bounds[1][2] for mesh in meshes), summit)
        self.assertEqual([mesh.bounds[0][:2] for mesh in easiest], [mesh.bounds[0][:2] for mesh in hardest])
        self.assertEqual([mesh.bounds[1][:2] for mesh in easiest], [mesh.bounds[1][:2] for mesh in hardest])

    def test_flat_calibration_uses_same_spawn_and_goal_without_zero_volume_meshes(self):
        flat, origin = self._build(step_height_range=(0.0, 0.0))
        self.assertEqual(len(flat), 1)
        self.assertEqual(origin, (0.9, 3.0, 0.0))
        self.assertAlmostEqual(self.namespace["course_goal_distance"](self._cfg()), 12.5)
        self.assertAlmostEqual(self.namespace["COURSE_GOAL_X"], 13.4)
        self.assertEqual(_surface_height(flat, 13.4), 0.0)

    def test_default_goal_is_one_metre_after_the_final_descending_tread(self):
        cfg = self._cfg()
        course_end = cfg.approach_length + 2 * cfg.num_steps * cfg.step_width + cfg.walkway_length
        goal_distance = self.namespace["course_goal_distance"](cfg)
        self.assertAlmostEqual(cfg.spawn_x + goal_distance - course_end, 1.0)
        self.assertAlmostEqual(goal_distance, self.namespace["COURSE_GOAL_DISTANCE"])
        self.assertLess(cfg.spawn_x + goal_distance, cfg.size[0])

    def test_extended_course_and_exit_fit_inside_longer_tile(self):
        cfg = self._cfg()
        meshes, _ = self._build(step_height_range=(0.08, 0.08))
        last_down_riser = cfg.approach_length + cfg.num_steps * cfg.step_width + cfg.walkway_length
        last_down_riser += (cfg.num_steps - 1) * cfg.step_width
        footprint_end = cfg.approach_length + 2 * cfg.num_steps * cfg.step_width + cfg.walkway_length
        self.assertAlmostEqual(last_down_riser, 12.05)
        self.assertAlmostEqual(footprint_end, 12.4)
        self.assertGreaterEqual(cfg.size[0], footprint_end + 1.0)
        self.assertEqual(_surface_height(meshes, self.namespace["COURSE_GOAL_X"]), 0.0)
        # The former finish is now on the descending flight, not the exit.
        self.assertGreater(_surface_height(meshes, 9.2), 0.0)

    def test_invalid_difficulty_is_rejected(self):
        for difficulty in (-0.001, 1.001, math.nan, math.inf, -math.inf):
            with self.subTest(difficulty=difficulty), self.assertRaises(ValueError):
                self._build(difficulty)

    def test_invalid_geometry_is_rejected(self):
        invalid = [
            {"num_steps": 0}, {"num_steps": True}, {"num_steps": 2.5},
            {"step_height_range": (-0.01, 0.15)}, {"step_height_range": (0.2, 0.1)},
            {"step_height_range": (0.03, math.nan)}, {"step_width": 0.0},
            {"walkway_length": 0.0}, {"size": (8.0, 6.0)}, {"size": (12.0, 6.0)},
            {"size": (16.0, 1.0)},
            {"size": (math.inf, 6.0)}, {"spawn_x": 0.4}, {"spawn_x": 1.6},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._build(**changes)


if __name__ == "__main__":
    unittest.main()
