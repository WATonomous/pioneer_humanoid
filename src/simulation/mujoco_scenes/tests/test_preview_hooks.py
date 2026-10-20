"""Exercise the managed viewer's callback using real MuJoCo stepping without a GUI."""
import unittest
from unittest.mock import patch

import mujoco
import mujoco.viewer
import numpy as np

from humanoid_mujoco_scenes import make_model, scene_reset
from humanoid_mujoco_scenes import preview
from pioneer_humanoid.mujoco_bimanual_arm import set_home


class PreviewHookTests(unittest.TestCase):
    def setUp(self):
        self.previous = mujoco.get_mjcb_control()

    def tearDown(self):
        mujoco.set_mjcb_control(self.previous)

    def model_data(self, name):
        model = make_model(name)
        data = mujoco.MjData(model)
        set_home(model, data)
        reset = scene_reset(name)
        if reset is not None:
            reset(model, data, np.random.default_rng(0))
        mujoco.mj_forward(model, data)
        return model, data

    def test_zip_tie_pawl_and_welds_follow_pulled_carriage(self):
        from humanoid_mujoco_scenes.zip_tie import scene as zip_tie
        model, data = self.model_data("zip_tie")
        original_welds = data.eq_active.copy()

        def simulate(model, data):
            data.qpos[model.joint("zip_carriage").qposadr[0]] = 4 * zip_tie.TOOTH_PITCH
            mujoco.mj_step(model, data)
            self.assertGreater(data.ctrl[model.actuator("zip_pawl").id], 0)
            self.assertTrue(np.any(data.eq_active != original_welds))

        with patch.object(mujoco.viewer, "launch", simulate):
            preview._launch_interactive("zip_tie", model, data)
        self.assertIs(mujoco.get_mjcb_control(), self.previous)

    def test_duplo_snaps_seated_brick_and_reset_reinitializes_order(self):
        from humanoid_mujoco_scenes.duplo import scene as duplo
        model, data = self.model_data("duplo")

        def simulate(model, data):
            adr = model.joint("brick_red").qposadr[0]
            data.qpos[adr:adr + 7] = [*duplo.PLATE_POS, duplo.T + duplo.PLATE_T, 1, 0, 0, 0]
            mujoco.mj_step(model, data)
            self.assertTrue(data.eq_active[model.equality("snap_red_on_baseplate").id])
            mujoco.mj_step(model, data, nstep=20)
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            self.assertEqual(sorted(data.userdata[1:].tolist()), [0, 1, 2])
            self.assertAlmostEqual(data.ctrl[model.actuator("joint4l").id], -np.pi / 2)
            self.assertTrue(np.isfinite(data.qpos).all())

        with patch.object(mujoco.viewer, "launch", simulate):
            preview._launch_interactive("duplo", model, data)

    def test_paused_forward_calls_do_not_repeat_hook_and_rewind_resumes(self):
        model, data = self.model_data("bare")
        calls = []

        def simulate(model, data):
            mujoco.mj_step(model, data, nstep=21)
            self.assertTrue(np.allclose(calls, [0, 0.01, 0.02]))
            mujoco.mj_forward(model, data)
            count = len(calls)
            for _ in range(5):
                mujoco.mj_forward(model, data)
            self.assertEqual(len(calls), count)
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            self.assertEqual(calls[-1], 0)

        with patch.object(preview, "scene_step", return_value=lambda m, d: calls.append(d.time)), \
             patch.object(mujoco.viewer, "launch", simulate):
            preview._launch_interactive("bare", model, data)

    def test_previous_callback_runs_and_is_restored_on_error(self):
        model, data = self.model_data("zip_tie")
        previous_calls = []

        def previous(model, data):
            previous_calls.append(data.time)

        mujoco.set_mjcb_control(previous)

        def fail(model, data):
            mujoco.mj_step(model, data)
            raise RuntimeError("viewer failed")

        with patch.object(mujoco.viewer, "launch", fail):
            with self.assertRaisesRegex(RuntimeError, "viewer failed"):
                preview._launch_interactive("zip_tie", model, data)
        self.assertTrue(previous_calls)
        self.assertIs(mujoco.get_mjcb_control(), previous)

    def test_loading_another_model_does_not_apply_original_scene_hook(self):
        model, data = self.model_data("zip_tie")
        other_model, other_data = self.model_data("bare")

        def simulate(model, data):
            mujoco.mj_step(other_model, other_data, nstep=20)
            mujoco.mj_step(model, data, nstep=20)

        with patch.object(mujoco.viewer, "launch", simulate):
            preview._launch_interactive("zip_tie", model, data)
        self.assertAlmostEqual(other_data.time, 0.02)


if __name__ == "__main__":
    unittest.main()
