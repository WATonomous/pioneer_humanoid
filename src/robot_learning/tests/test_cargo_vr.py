"""Exercise real VR mapping/IK and HG authority without sockets, Isaac or hardware."""
import time
import unittest
from types import SimpleNamespace

import numpy as np

from humanoid_robot_learning.sim.cargo_control import HumanGate
from humanoid_robot_learning.sim.cargo_vr import VrExpert
from control import Input
from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, DEFAULT_JOINT_POS


class Tensor:
    def __init__(self, values): self.values = np.array(values)
    def __getitem__(self, index): return Tensor(self.values[index])
    def cpu(self): return self
    def numpy(self): return self.values


class CargoVrTests(unittest.TestCase):
    def setUp(self):
        names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
        joints = [DEFAULT_JOINT_POS[j] for j in names]
        joints[3] = -2.1
        self.env = SimpleNamespace(
            arm_ids=list(range(6)), last=np.r_[joints[:6], 1.],
            robot=SimpleNamespace(joint_names=names, data=SimpleNamespace(
                joint_pos=Tensor([joints]), root_quat_w=Tensor([[1., 0., 0., 0.]]))),
            contract=lambda: {"ego_eye": [.9, 1.2, 1.7], "ego_target": [.38, .3, .87]})
        self.expert = VrExpert(self.env)
        self.expert.seed(self.env)
        self.saved = []
        self.resets = 0
        actor = SimpleNamespace(reset=self.reset_actor, action=lambda *_: self.env.last.copy())
        self.gate = HumanGate(self.env, self.expert, actor,
                              lambda frames, reason: self.saved.append((list(frames), reason)), 750)

    def reset_actor(self): self.resets += 1

    def sample(self, right=False, trigger=0., left_tracked=True, right_tracked=True):
        def control(clutch, grip):
            return Input(np.array([0., 1., 0.]), np.eye(3), clutch, grip, False, False)
        shared = self.expert.driver.shared
        with shared.lock:
            shared.packet = {"head": np.eye(3),
                             "left": control(False, trigger) if left_tracked else None,
                             "right": control(right, 0.) if right_tracked else None}
            shared.sequence += 1
            shared.received = time.monotonic()
        return self.expert.poll(self.env)

    def takeover(self):
        self.assertIsNone(self.sample())  # establish released state
        self.assertEqual(self.sample(right=True), "HUMAN")
        self.gate.toggle()
        self.sample(right=True)

    def test_takeover_preserves_carried_gripper_until_trigger_matches(self):
        self.takeover()
        self.assertEqual(self.gate.mode, "HUMAN")
        action = self.gate.select({}, "cargo")
        self.assertEqual(action.shape, (7,))  # right arm has no command channel
        self.assertEqual(action[6], 1.)  # relaxed trigger cannot drop held cargo
        self.sample(right=True, trigger=1.)
        self.assertEqual(self.gate.select({}, "cargo")[6], 1.)
        self.sample(right=True, trigger=.2)
        self.assertAlmostEqual(self.gate.select({}, "cargo")[6], .2)

    def test_release_saves_only_human_frames_and_resets_policy_queue(self):
        self.gate.record({}, self.env.last, "cargo")
        self.assertEqual(self.gate.frames, [])
        self.takeover()
        action = self.gate.select({}, "cargo")
        self.gate.record({"observation.state": np.zeros(7)}, action, "cargo")
        self.assertEqual(self.sample(right=False), "POLICY")
        self.gate.toggle()
        self.assertEqual(self.gate.mode, "POLICY")
        self.assertEqual(len(self.saved[0][0]), 1)
        self.assertEqual(self.saved[0][1], "hand_back")
        self.assertEqual(self.resets, 2)

    def test_stale_input_blocks_action_and_does_not_return_authority(self):
        self.takeover()
        self.expert.driver.shared.received = time.monotonic() - 1.
        self.assertFalse(self.expert.is_ready())
        with self.assertRaises(RuntimeError): self.gate.select({}, "cargo")
        self.assertIsNone(self.expert.poll(self.env))
        self.assertEqual(self.gate.mode, "HUMAN")
        # Reconnection with a released button is synchronization, not hand-back.
        self.assertIsNone(self.sample(right=False))
        self.assertEqual(self.sample(right=True), "HUMAN")
        self.assertEqual(self.sample(right=False), "POLICY")

    def test_missing_left_tracking_blocks_human_motion(self):
        self.takeover()
        self.sample(right=True, left_tracked=False)
        self.assertFalse(self.expert.is_ready())
        with self.assertRaises(RuntimeError): self.gate.select({}, "cargo")

    def test_startup_held_button_requires_deliberate_new_press(self):
        self.assertIsNone(self.sample(right=True))
        self.assertEqual(self.gate.mode, "POLICY")
        self.assertEqual(self.sample(right=False), "POLICY")
        self.assertEqual(self.sample(right=True), "HUMAN")

    def test_loss_arriving_during_render_blocks_cached_ready_state(self):
        self.takeover()
        self.assertTrue(self.expert.is_ready())
        shared = self.expert.driver.shared
        # Simulate a WebXR loss arriving after poll(), during camera rendering.
        with shared.lock:
            shared.packet = {**shared.packet, "right": None}
            shared.received = time.monotonic()
        self.assertFalse(self.expert.is_ready())
        with self.assertRaises(RuntimeError): self.gate.select({}, "cargo")

    def test_camera_mapping_is_rotation_and_published_views_reuse_policy_images(self):
        rotation = self.expert.driver.targets["left"].calibration
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        calls = []
        self.expert.driver.publish_cameras = lambda *images: calls.append(images)
        ego, wrist = np.zeros((240, 320, 3)), np.ones((240, 320, 3))
        self.expert.publish({"observation.images.ego": ego, "observation.images.wrist_left": wrist}, self.gate)
        self.assertTrue(calls[0][0] is ego and calls[0][1] is ego)
        self.assertTrue(calls[0][2] is wrist and calls[0][3] is wrist)


if __name__ == "__main__":
    unittest.main()
