"""CPU-only course completion/curriculum tests; Torch is optional on the host.

Run with the Isaac image's Python for the full suite. No Isaac imports, simulator,
CUDA tensors, or GPU initialization are needed. Hosts without Torch skip these
tests, while the source/geometry contracts remain standard-library-only.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


class _Scene(dict):
    pass


@unittest.skipIf(torch is None, "CPU Torch is supplied by the Isaac container")
class StairCourseEpisodeLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "humanoid_rl_tasks/locomotion/mdp/stair_course.py"
        spec = importlib.util.spec_from_file_location("course_terms_under_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        torch.set_num_threads(1)

    def make_env(self, positions, *, gravity=None, rollout=None, terminated=None):
        count = len(positions)
        origin = torch.tensor([[24.9, 15.0, 0.0]], dtype=torch.float32).repeat(count, 1)
        data = SimpleNamespace(
            root_pos_w=origin + torch.tensor(positions, dtype=torch.float32),
            projected_gravity_b=torch.tensor(gravity if gravity is not None else [[0, 0, -1]] * count,
                                            dtype=torch.float32),
        )
        scene = _Scene(robot=SimpleNamespace(data=data))
        scene.env_origins = origin
        calls = []
        scene.terrain = SimpleNamespace(
            terrain_levels=torch.ones(count, dtype=torch.long),
            update_env_origins=lambda env_ids, **kwargs: calls.append((env_ids, kwargs)),
        )
        return SimpleNamespace(
            scene=scene, calls=calls,
            episode_length_buf=torch.tensor(rollout if rollout is not None else [100] * count),
            termination_manager=SimpleNamespace(
                terminated=torch.tensor(terminated if terminated is not None else [False] * count)),
        )

    def test_only_full_upright_lane_traversal_completes(self):
        env = self.make_env([
            [6.3, 0, 1.8],  # preview summit 0.96 m + upright root 0.84 m
            [12.6, 0, 0.84],  # whole course traversed
            [12.6, 2.1, 0.84],  # left the lane
            [12.6, 0, 0.4],  # collapsed at exit
            [12.6, 0, 0.84],  # tipped at exit
            [float("nan"), 0, 0.84],
            [12.6, 0, 0.84],  # fully inverted at exit is not upright
        ], gravity=[[0, 0, -1]] * 4 + [[0.9, 0, -0.4], [0, 0, -1], [0, 0, 1]])
        self.assertEqual(self.module.course_complete(env, goal_distance=12.5).tolist(),
                         [False, True, False, False, False, False, False])

    def test_former_nine_point_two_metre_finish_does_not_complete_extended_course(self):
        # Positions are relative to the 0.9 m local-tile spawn. At local X=9.2,
        # the robot is now on a 0.72 m-high descending tread, not flat exit.
        env = self.make_env([[9.2 - 0.9, 0, 0.72 + 0.84], [12.6, 0, 0.84]])
        self.assertEqual(self.module.course_complete(env, goal_distance=12.5).tolist(), [False, True])
        self.module.complete_stair_course_levels(env, torch.arange(2), goal_distance=12.5)
        _, flags = env.calls[0]
        self.assertEqual(flags["move_up"].tolist(), [False, True])
        self.assertEqual(flags["move_down"].tolist(), [True, False])

    def test_sideways_backward_and_nonfinite_are_out_of_bounds(self):
        env = self.make_env([[0, 0, 0.84], [0, 2.1, 0.84], [-0.8, 0, 0.84],
                             [0, float("inf"), 0.84], [12.6, 0, 0.84]])
        self.assertEqual(self.module.course_out_of_bounds(env).tolist(), [False, True, True, True, False])

    def test_promotion_requires_whole_route_and_rejects_simultaneous_fall(self):
        env = self.make_env([[12.6, 0, 0.84], [6.3, 0, 1.8], [12.6, 0, 0.84]],
                            terminated=[False, False, True])
        result = self.module.complete_stair_course_levels(env, torch.arange(3), goal_distance=12.5)
        self.assertEqual(float(result), 1.0)
        self.assertEqual(len(env.calls), 1)
        _, flags = env.calls[0]
        self.assertEqual(flags["move_up"].tolist(), [True, False, False])
        self.assertEqual(flags["move_down"].tolist(), [False, True, True])

    def test_initial_reset_is_neutral_and_subset_updates_are_scoped(self):
        env = self.make_env([[12.6, 0, 0.84], [6.3, 0, 1.8], [12.6, 0, 0.84]], rollout=[0, 100, 0])
        ids = torch.tensor([0, 2])
        self.module.complete_stair_course_levels(env, ids, goal_distance=12.5)
        observed, flags = env.calls[0]
        self.assertTrue(torch.equal(observed, ids))
        self.assertEqual(flags["move_up"].tolist(), [False, False])
        self.assertEqual(flags["move_down"].tolist(), [False, False])


if __name__ == "__main__":
    unittest.main()
