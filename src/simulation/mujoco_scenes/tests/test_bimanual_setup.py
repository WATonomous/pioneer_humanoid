"""Checks for the bimanual MuJoCo setup, joint mapping and optional recording."""
import tempfile
import unittest
from pathlib import Path
import importlib.util

import mujoco
import numpy as np

from humanoid_robot_learning.bimanual import BimanualMapping, BIMANUAL_RECORD_JOINTS
from humanoid_robot_learning.schema import build_features, load_yaml
from humanoid_robot_learning.sim_teleop_record import BIMANUAL_SCHEMA, DEFAULT_SCHEMA
from humanoid_mujoco_scenes import make_model
from pioneer_humanoid.mujoco_bimanual_arm import set_home


class RecorderTests(unittest.TestCase):
    def test_schemas_preserve_single_arm_and_define_bimanual_order(self):
        left = build_features(load_yaml(DEFAULT_SCHEMA))
        both = build_features(load_yaml(BIMANUAL_SCHEMA))
        self.assertEqual(left["action"]["shape"], (7,))
        self.assertEqual(both["action"]["shape"], (14,))
        self.assertEqual(both["action"]["names"], BIMANUAL_RECORD_JOINTS)
        self.assertIn("observation.images.wrist_right", both)

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "Install the sim recording extra for the recorder test")
    def test_recording_preserves_both_arms(self):
        from humanoid_robot_learning.sim_recorder import SimLeRobotRecorder
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "recordings"
            recorder = SimLeRobotRecorder(
                task_name="metadata", repo_id="humanoid/pioneer_v1_bimanual",
                dataset_root=root, fps=25, device="cpu", joint_names=BIMANUAL_RECORD_JOINTS,
                cameras={}, buffer_capacity_s=0.2, rate_limit=False,
            )
            try:
                recorder.init_dataset()
                for i in range(3):
                    recorder.push_frame_to_buffer(np.arange(14, dtype=np.float32) + i,
                                                  np.arange(14, dtype=np.float32) - i, {})
                recorder.save_episode()
            finally:
                recorder.finalize()
            data = LeRobotDataset("humanoid/pioneer_v1_bimanual", root=root)
            np.testing.assert_array_equal(data[1]["action"], np.arange(14) + 1)
            np.testing.assert_array_equal(data[1]["observation.state"], np.arange(14) - 1)


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_model("bare")
        cls.mapping = BimanualMapping(cls.model)

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        set_home(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def test_independent_arms_gripper_mapping_and_atomic_validation(self):
        action = self.mapping.state(self.data)
        action[0], action[7], action[6], action[13] = 0.1, -0.2, 0.25, 0.75
        self.mapping.apply(self.data, action)
        for name, expected in (("joint1L", 0.1), ("joint1", -0.2),
                               ("joint7l", -0.0125), ("joint8l", 0.0125),
                               ("joint7", -0.0375), ("joint8", 0.0375)):
            self.assertAlmostEqual(self.data.ctrl[self.model.actuator(name).id], expected, places=6)
        before = self.data.ctrl.copy()
        action[13] = np.nan
        with self.assertRaises(ValueError):
            self.mapping.apply(self.data, action)
        np.testing.assert_array_equal(self.data.ctrl, before)
        with self.assertRaises(ValueError):
            self.mapping.apply(self.data, np.zeros(7))

    def test_measured_state_and_clamping(self):
        for arm, closure in zip(self.mapping.arms, (0.2, 0.8)):
            self.data.qpos[arm["finger_qpos"]] = arm["opened"] + closure * (arm["closed"] - arm["opened"])
        state = self.mapping.state(self.data)
        np.testing.assert_allclose(state[[6, 13]], [0.2, 0.8])
        bounded = self.mapping.apply(self.data, np.full(14, 100.0))
        np.testing.assert_array_equal(bounded[[6, 13]], [1, 1])
        self.assertLess(bounded[0], 100)

    def test_simultaneous_step_and_repeatable_reset(self):
        original = self.data.qpos.copy()
        action = self.mapping.state(self.data)
        action[[0, 7]] = [0.1, -0.1]
        self.mapping.apply(self.data, action)
        mujoco.mj_step(self.model, self.data, nstep=40)
        self.assertAlmostEqual(self.data.time, 0.04)
        for arm in self.mapping.arms:
            self.assertNotEqual(self.data.qpos[arm["qpos"][0]], original[arm["qpos"][0]])
        mujoco.mj_resetData(self.model, self.data)
        set_home(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        np.testing.assert_array_equal(self.data.qpos, original)


if __name__ == "__main__":
    unittest.main()
