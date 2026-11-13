import unittest
from types import SimpleNamespace
from humanoid_robot_learning.sim.cargo_control import HumanGate


class CargoGateTests(unittest.TestCase):
    def setUp(self):
        self.resets = 0
        self.seeds = []
        self.saved = []
        self.env = SimpleNamespace(last=[0, 0, 0, 0, 0, 0, .8])
        self.actor = SimpleNamespace(reset=self.reset_actor, action=lambda obs, task: "policy")
        self.expert = SimpleNamespace(seed=lambda env: self.seeds.append(env.last[-1]), action=lambda env: "human")
        self.gate = HumanGate(self.env, self.expert, self.actor,
                              lambda frames, reason: self.saved.append((list(frames), reason)), 3)

    def reset_actor(self):
        self.resets += 1

    def test_policy_actions_never_become_labels(self):
        self.assertEqual(self.gate.select({}, "cargo"), "policy")
        self.gate.record({}, "policy", "cargo")
        self.assertEqual(self.gate.frames, [])
        self.gate.toggle()
        self.assertEqual(self.seeds, [.8])  # takeover preserves a closed gripper
        self.assertEqual(self.gate.select({}, "cargo"), "human")
        self.gate.record({"observation.state": "before"}, "executed human", "cargo")
        self.gate.toggle()
        self.assertEqual(self.saved[0][0][0]["action"], "executed human")
        self.assertEqual(self.saved[0][1], "hand_back")
        self.assertEqual(self.resets, 2)  # stale chunks cleared on both transitions
        self.assertEqual(self.gate.frames, [])

    def test_two_interventions_are_separate_episodes(self):
        for label in ("first", "second"):
            self.gate.toggle()
            self.gate.record({}, label, "cargo")
            self.gate.toggle()
        self.assertEqual([s[0][0]["action"] for s in self.saved], ["first", "second"])

    def test_pause_does_not_admit_frames_or_execute_actions(self):
        self.gate.toggle()
        self.gate.pause()
        self.gate.record({}, "paused", "cargo")
        with self.assertRaises(RuntimeError):
            self.gate.select({}, "cargo")
        self.assertEqual(self.gate.frames, [])
        self.gate.pause()
        self.assertEqual(self.gate.mode, "HUMAN")

    def test_reset_discards_partial_correction(self):
        self.gate.toggle()
        self.gate.record({}, "discard me", "cargo")
        self.gate.reset()
        self.gate.toggle()
        self.assertEqual(self.saved, [])

    def test_memory_boundary_splits_without_autonomous_labels(self):
        self.gate.toggle()
        for i in range(4):
            self.gate.record({}, i, "cargo")
        self.assertEqual(len(self.saved[0][0]), 3)
        self.assertEqual(self.saved[0][1], "buffer_boundary")
        self.assertEqual(len(self.gate.frames), 1)

    def test_explicit_save_keeps_interventions_separate_across_policy_intervals(self):
        self.gate.save_on_hand_back = False
        for label in ("first", "second"):
            self.gate.toggle()
            self.gate.record({}, label, "cargo")
            self.gate.toggle()
            self.gate.record({}, "autonomous", "cargo")
        self.assertEqual(self.saved, [])
        self.assertEqual(self.gate.pending_frames, 2)
        self.gate.save("vr_operator_save")
        self.assertEqual([s[0][0]["action"] for s in self.saved], ["first", "second"])
        self.assertEqual(self.gate.pending_frames, 0)
        self.assertEqual(self.gate.saved_segments, 2)

    def test_explicit_save_buffer_limit_holds_instead_of_auto_saving(self):
        self.gate.save_on_hand_back = False
        self.gate.toggle()
        for i in range(4):
            self.gate.record({}, i, "cargo")
        self.assertTrue(self.gate.buffer_full)
        self.assertEqual(self.gate.pending_frames, 3)
        self.assertEqual(self.saved, [])
        with self.assertRaises(RuntimeError):
            self.gate.select({}, "cargo")
        self.gate.toggle()
        self.assertEqual(self.saved, [])
        self.assertEqual(self.gate.discard(), 3)
        self.assertFalse(self.gate.buffer_full)


if __name__ == "__main__":
    unittest.main()
