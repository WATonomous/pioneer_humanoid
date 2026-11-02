"""A real pick through the whole stack: the dataset it records, and the deploy runner's guards on the same stack.

    python3 -m unittest scenarios/test_recording.py     # full image: needs LeRobot
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from arm_emulator import config  # noqa: F401 -- sys.path for the arm scripts and scenes
from harness import HOME_DEG, Client, Stack

SCENARIOS = Path(__file__).resolve().parent
_SRC = SCENARIOS.parents[2]
sys.path.insert(0, str(_SRC / "robot_learning"))
FORCE_SCALE = "2"


class TestSyntheticTake(unittest.TestCase):
    def test_one_episode_is_recorded_with_measured_state(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            root = Path(out) / "dataset"
            result = subprocess.run([sys.executable, str(SCENARIOS / "synthetic_demos.py"), "--episodes", "1", "--max-tries", "3",
                                     "--cameras", "wrist_left", "--gripper-force-scale", FORCE_SCALE, "--dataset_root", str(root)],
                                    capture_output=True, text=True, timeout=240)
            self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-3000:])

            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset("humanoid/pioneer_v1_left_arm", root=next(root.rglob("meta")).parent)
            self.assertEqual(dataset.num_episodes, 1)
            self.assertGreater(dataset.num_frames, 25)
            frames = [dataset[i] for i in range(0, dataset.num_frames, 25)]
            state = np.stack([f["observation.state"].numpy() for f in frames])
            action = np.stack([f["action"].numpy() for f in frames])
            self.assertEqual(state.shape[1], 7)
            self.assertEqual(action.shape[1], 7)
            # Measured, not a copy of the command: soft gains leave a visible gap.
            self.assertGreater(np.abs(state[:, :6] - action[:, :6]).max(), np.radians(0.5))
            image = frames[0]["observation.images.wrist_left"].numpy()
            self.assertEqual(image.shape, (3, 480, 640))
            self.assertGreater(image.std(), 0.01, "the wrist image must not be blank")


class TestDeployRunner(unittest.TestCase):
    """RealArmPolicyRunner's own checks, against the real joint_command and emulated feedback."""

    def setUp(self) -> None:
        from arm_feedback import ArmFeedbackState
        from arm_emulator.config import REPO_CALIBRATION
        from humanoid_robot_learning.deploy.real_arm import RealArmPolicyRunner
        from rclpy.node import Node

        self.stack = Stack(cameras="wrist_left")
        self.client = Client(self.stack)
        self.addCleanup(self.stack.stop)
        self.addCleanup(self.client.close)
        self.node = Node("real_arm_policy_runner")
        self.client.add_node(self.node)
        self.action = None
        self.runner = RealArmPolicyRunner(self.node, lambda obs: self.action, ArmFeedbackState(REPO_CALIBRATION), fps=25.0,
                                          cameras={"wrist_left": ("/wrist_cam/image_rect", 480, 640)}, live=True)
        self.client.measured()

    def wait_for_observation(self) -> None:
        deadline = time.monotonic() + 30.0
        while self.runner.observation() is None:
            self.assertLess(time.monotonic(), deadline, f"missing {self.runner.missing()}")
            time.sleep(0.1)

    def test_preflight_passes_on_a_healthy_stack(self) -> None:
        self.wait_for_observation()
        self.assertEqual(self.runner.preflight(), [])

    def test_refuses_a_first_target_far_from_the_arm(self) -> None:
        self.wait_for_observation()
        before = self.client.measured()
        self.action = np.append(np.radians(np.array(HOME_DEG) + [40.0, 0, 0, 0, 0, 0]), 0.0)
        self.assertIsNone(self.runner._policy_loop())
        self.client.hold(0.3, before)
        np.testing.assert_allclose(self.client.measured(), before, atol=6.0)

    def test_observation_state_matches_the_feedback(self) -> None:
        self.wait_for_observation()
        obs, q_deg = self.runner.observation()
        np.testing.assert_allclose(np.degrees(obs["observation.state"][:6]), q_deg, atol=1e-3)
        self.assertEqual(obs["observation.images.wrist_left"].shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
