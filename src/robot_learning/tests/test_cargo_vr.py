"""Exercise real VR mapping/IK and HG authority without sockets, Isaac or hardware."""
import time
import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
                              lambda frames, reason: self.saved.append((list(frames), reason)), 750,
                              save_on_hand_back=False)
        self._save_down = self._reset_down = False

    def reset_actor(self): self.resets += 1

    def sample(self, right=False, trigger=0., left_tracked=True, right_tracked=True,
               left=False, head_tracked=True, poll=True, save=False, reset=False):
        def control(clutch, grip, **buttons):
            return Input(np.array([0., 1., 0.]), np.eye(3), clutch, grip, False, False, **buttons)
        shared = self.expert.driver.shared
        with shared.lock:
            shared.packet = {"head": np.eye(3) if head_tracked else None,
                             "left": control(left, trigger, reset_scene=reset) if left_tracked else None,
                             "right": control(right, 0., record=save) if right_tracked else None}
            shared.record_presses += int(save and not self._save_down)
            shared.reset_sequence += int(reset and not self._reset_down)
            shared.sequence += 1
            shared.received = time.monotonic()
        self._save_down, self._reset_down = save, reset
        return self.expert.poll(self.env) if poll else None

    def test_startup_waits_for_complete_tracking_and_released_grips(self):
        self.assertTrue(self.expert.waiting_for_start)
        self.expert.poll(self.env)
        self.assertTrue(self.expert.waiting_for_start)
        self.sample(right_tracked=False)
        self.assertTrue(self.expert.waiting_for_start)
        self.sample(left_tracked=False)
        self.assertTrue(self.expert.waiting_for_start)
        self.sample(head_tracked=False)
        self.assertTrue(self.expert.waiting_for_start)
        self.sample(right=True)
        self.assertTrue(self.expert.waiting_for_start)
        self.sample(left=True)
        self.assertTrue(self.expert.waiting_for_start)
        self.sample()
        self.assertFalse(self.expert.waiting_for_start)

    def test_startup_wait_is_not_rearmed_by_takeover_reset_or_disconnect(self):
        self.sample()
        self.assertFalse(self.expert.waiting_for_start)
        self.expert.seed(self.env)
        self.expert.driver.shared.received = time.monotonic() - 1.
        self.expert.poll(self.env)
        self.assertFalse(self.expert.waiting_for_start)
        self.assertFalse(self.expert.is_ready())

    def test_collector_streams_but_does_not_act_or_record_until_vr_connects(self):
        # Exercise the actual collection loop without importing its Isaac launcher.
        path = Path(__file__).resolve().parents[1] / "humanoid_robot_learning/sim/hg_dagger.py"
        function = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "collect")
        actor = SimpleNamespace(reset=Mock(), action=Mock(return_value=self.env.last.copy()))
        writer = SimpleNamespace(episodes=0, save=Mock(), close=Mock())
        self.env.reset = Mock()
        self.env.sim = SimpleNamespace(is_playing=lambda: True, render=Mock())
        self.env.observe = Mock(return_value={"observation.images.ego": np.zeros((2, 2, 3)),
                                             "observation.images.wrist_left": np.zeros((2, 2, 3))})
        self.env.command = Mock(side_effect=lambda action: action)
        self.env.advance = Mock()
        self.expert.start, self.expert.close = Mock(), Mock()
        self.expert.driver.publish_cameras = Mock()
        iterations = iter((0, 1, 2))

        def running():
            tick = next(iterations)
            if tick == 1:
                actor.action.assert_not_called()
                self.env.command.assert_not_called()
                self.env.advance.assert_not_called()
                writer.save.assert_not_called()
                self.assertTrue(self.env.observe.called)  # browser preview is live while waiting
                self.sample(poll=False)
            return tick < 2

        args = SimpleNamespace(dataset_root="unused", resume=False, mode="collect", teleop="vr",
                               vr_port=8444, vr_gain=1., vr_calibration=None, seed=0,
                               headless=True, episodes=1, max_segment_frames=750)
        namespace = dict(args=args, EpisodeWriter=lambda *_: writer, load_actor=lambda _: actor,
                         app=SimpleNamespace(is_running=running), time=time, FPS=25, TASK="cargo")
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        with patch.dict(sys.modules, {"carb": SimpleNamespace()}), \
                patch("humanoid_robot_learning.sim.cargo_vr.VrExpert", return_value=self.expert), \
                patch("time.sleep"), patch("builtins.print"):
            namespace["collect"](self.env)
        actor.action.assert_called_once()
        self.env.command.assert_called_once()
        self.env.advance.assert_called_once()
        writer.save.assert_not_called()  # autonomous actions remain excluded
        writer.close.assert_called_once()

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

    def test_release_stages_human_frames_until_explicit_save_and_resets_policy_queue(self):
        self.gate.record({}, self.env.last, "cargo")
        self.assertEqual(self.gate.frames, [])
        self.takeover()
        action = self.gate.select({}, "cargo")
        self.gate.record({"observation.state": np.zeros(7)}, action, "cargo")
        self.assertEqual(self.sample(right=False), "POLICY")
        self.gate.toggle()
        self.assertEqual(self.gate.mode, "POLICY")
        self.assertEqual(self.saved, [])
        self.assertEqual(self.gate.pending_frames, 1)
        self.sample(save=True)
        self.assertTrue(self.expert.save_requested)
        self.gate.save("vr_operator_save")
        self.assertEqual(len(self.saved[0][0]), 1)
        self.assertEqual(self.saved[0][1], "vr_operator_save")
        self.sample(save=True)
        self.assertFalse(self.expert.save_requested)
        self.assertEqual(self.resets, 2)

    def test_discard_reset_holds_until_grip_release_without_saving(self):
        self.takeover()
        self.gate.record({}, self.env.last, "cargo")
        self.gate.reset(resume_policy=True)
        self.expert.waiting_for_release = True
        for _ in range(3):
            self.assertIsNone(self.sample(right=True))
            with self.assertRaises(RuntimeError):
                self.expert.action(self.env)
        self.assertEqual(self.sample(right=False), "POLICY")
        self.assertFalse(self.expert.waiting_for_release)
        self.assertEqual(self.gate.mode, "POLICY")
        self.assertEqual(self.gate.pending_frames, 0)
        self.assertEqual(self.saved, [])

    def _collect_script(self, packets, fail_save=False):
        path = Path(__file__).resolve().parents[1] / "humanoid_robot_learning/sim/hg_dagger.py"
        function = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "collect")
        saved = []
        writer = SimpleNamespace(episodes=0, close=Mock())
        def save(frames, provenance):
            if fail_save:
                raise RuntimeError("recording write failed")
            saved.append((list(frames), provenance))
            writer.episodes += 1
        writer.save = save
        actor = SimpleNamespace(reset=Mock(), action=Mock(return_value=self.env.last.copy()))
        self.env.reset = Mock()
        self.env.reset_metadata = {"profile": "test"}
        self.env.sim = SimpleNamespace(is_playing=lambda: True, render=Mock())
        tick = -1
        self.env.observe = lambda: {"observation.state": np.full(7, tick),
                                   "observation.images.ego": np.zeros((2, 2, 3)),
                                   "observation.images.wrist_left": np.zeros((2, 2, 3))}
        self.env.command = Mock(side_effect=lambda action: action)
        self.env.advance = Mock()
        self.expert.start, self.expert.close = Mock(), Mock()
        self.expert.driver.publish_cameras = Mock()
        def running():
            nonlocal tick
            tick += 1
            if tick == len(packets):
                return False
            self.sample(**packets[tick], poll=False)
            return True
        args = SimpleNamespace(dataset_root="unused", resume=False, mode="collect", teleop="vr",
                               vr_port=8444, vr_gain=1., vr_calibration=None, seed=0, policy="test",
                               round_id="test", headless=True, episodes=100, max_segment_frames=750)
        namespace = dict(args=args, EpisodeWriter=lambda *_: writer, load_actor=lambda _: actor,
                         app=SimpleNamespace(is_running=running), time=time, FPS=25, TASK="cargo")
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        with patch.dict(sys.modules, {"carb": SimpleNamespace()}), \
                patch("humanoid_robot_learning.sim.cargo_vr.VrExpert", return_value=self.expert), \
                patch("time.sleep"), patch("builtins.print"):
            namespace["collect"](self.env)
        return saved

    def test_collector_release_without_b_never_writes_a_demo(self):
        saved = self._collect_script([{}, {"right": True}, {"right": True}, {}, {}])
        self.assertEqual(saved, [])

    def test_collector_b_saves_only_human_actions_once_when_held(self):
        saved = self._collect_script([{}, {"right": True}, {"right": True}, {},
                                      {"save": True}, {"save": True}, {}])
        self.assertEqual(len(saved), 1)
        frames, metadata = saved[0]
        self.assertEqual([int(f["observation.state"][0]) for f in frames], [2])
        self.assertEqual(metadata["boundary"], "vr_operator_save")
        self.assertEqual(metadata["save_mode"], "explicit_button")
        self.assertEqual(metadata["seed"], 0)  # label the saved attempt before changing seeds
        self.assertEqual([call.args[0] for call in self.env.reset.call_args_list], [0, 1])

    def test_collector_b_can_save_while_right_grip_is_still_held(self):
        saved = self._collect_script([{}, {"right": True}, {"right": True},
                                      {"right": True, "save": True}])
        self.assertEqual(len(saved), 1)
        self.assertEqual([int(f["observation.state"][0]) for f in saved[0][0]], [2])

    def test_b_press_is_not_lost_if_tracking_disappears_before_next_poll(self):
        self.sample()
        shared = self.expert.driver.shared
        with shared.lock:
            shared.record_presses += 1  # transport received a tracked right-B press
            shared.packet = {"head": None, "left": None, "right": None}
            shared.received = time.monotonic()
        self.expert.poll(self.env)
        self.assertTrue(self.expert.save_requested)
        self.assertFalse(self.expert.is_ready())  # motion still cannot use lost tracking
        self.expert.poll(self.env)
        self.assertFalse(self.expert.save_requested)

    def test_old_browser_page_and_empty_save_are_visible_in_feedback(self):
        self.sample()
        self.expert.driver.shared.record_button_supported = False
        self.expert.feedback(self.gate)
        feedback = self.expert.driver.shared.feedback
        self.assertIn("REFRESH QUEST BROWSER PAGE", feedback["status"])
        self.assertFalse(feedback["vr_client_has_save_button"])
        self.gate.notice = "NOTHING TO SAVE"
        self.expert.feedback(self.gate)
        self.assertIn("NOTHING TO SAVE", feedback["status"])

    def test_collector_y_overrides_b_and_drops_both_failed_and_post_reset_frames(self):
        saved = self._collect_script([{}, {"right": True}, {"right": True}, {},
                                      {"reset": True, "save": True}, {"right": True},
                                      {"right": True}, {}, {},
                                      {"right": True}, {"right": True}, {}, {"save": True}])
        self.assertEqual(len(saved), 1)
        frames, metadata = saved[0]
        self.assertEqual([int(f["observation.state"][0]) for f in frames], [10])
        self.assertEqual(metadata["seed"], 1)

    def test_b_resets_even_when_no_human_correction_is_pending(self):
        saved = self._collect_script([{}, {"save": True}, {}])
        self.assertEqual(saved, [])
        self.assertEqual([call.args[0] for call in self.env.reset.call_args_list], [0, 1])

    def test_save_error_does_not_reset_before_write_success(self):
        with self.assertRaisesRegex(RuntimeError, "recording write failed"):
            self._collect_script([{}, {"right": True}, {"right": True}, {}, {"save": True}],
                                 fail_save=True)
        self.assertEqual([call.args[0] for call in self.env.reset.call_args_list], [0])

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
