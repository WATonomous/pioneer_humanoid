"""Exercise the passive viewer loop with real MuJoCo physics and a headless viewer handle."""
import contextlib
import unittest
from unittest.mock import Mock, patch

import mujoco
import mujoco.viewer
import numpy as np

from humanoid_mujoco_scenes import make_model
from humanoid_mujoco_scenes import preview


class FakeViewer:
    def __init__(self, model, data, *, key_callback, frames=5, events=None):
        self.model, self.data, self.key_callback = model, data, key_callback
        self.frames, self.events = frames, events or {}
        self.cam = mujoco.MjvCamera()
        self.times = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def lock(self):
        return contextlib.nullcontext()

    def is_running(self):
        return len(self.times) < self.frames

    def sync(self):
        self.times.append(self.data.time)
        event = self.events.get(len(self.times))
        if event is not None:
            event(self)


class PreviewHookTests(unittest.TestCase):
    def model_data(self, name):
        model = make_model(name)
        data = mujoco.MjData(model)
        preview._reset_scene(name, model, data)
        return model, data

    def run_viewer(self, name, model, data, **options):
        handle = None

        def launch(model, data, *, key_callback):
            nonlocal handle
            handle = FakeViewer(model, data, key_callback=key_callback, **options)
            return handle

        with patch.object(mujoco.viewer, 'launch_passive', launch), patch.object(preview.time, 'sleep'):
            preview._launch_interactive(name, model, data)
        self.assertTrue(handle.closed)
        return handle

    def test_zip_tie_pawl_and_welds_follow_pulled_carriage(self):
        from humanoid_mujoco_scenes.zip_tie import scene as zip_tie
        model, data = self.model_data('zip_tie')
        original_welds = data.eq_active.copy()
        data.qpos[model.joint('zip_carriage').qposadr[0]] = 4 * zip_tie.TOOTH_PITCH
        self.run_viewer('zip_tie', model, data, frames=1)
        self.assertGreater(data.ctrl[model.actuator('zip_pawl').id], 0)
        self.assertTrue(np.any(data.eq_active != original_welds))

    def test_duplo_snaps_and_native_reset_reinitializes_scene(self):
        from humanoid_mujoco_scenes.duplo import scene as duplo
        model, data = self.model_data('duplo')
        adr = model.joint('brick_red').qposadr[0]
        data.qpos[adr:adr + 7] = [*duplo.PLATE_POS, duplo.T + duplo.PLATE_T, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        def check_snap(handle):
            self.assertTrue(data.eq_active[model.equality('snap_red_on_baseplate').id])

        def reset(handle):
            mujoco.mj_resetData(model, data)

        handle = self.run_viewer('duplo', model, data, events={1: check_snap, 3: reset})
        self.assertAlmostEqual(handle.times[3], 0.01)
        self.assertEqual(sorted(data.userdata[1:].tolist()), [0, 1, 2])
        self.assertAlmostEqual(data.ctrl[model.actuator('joint4l').id], -np.pi / 2)

    def test_space_pauses_physics_and_hooks_and_r_resets_while_paused(self):
        model, data = self.model_data('bare')
        calls = []
        events = {1: lambda h: h.key_callback(32),
                  2: lambda h: h.key_callback(82),
                  3: lambda h: h.key_callback(32)}
        with patch.object(preview, 'scene_step', return_value=lambda m, d: calls.append(d.time)):
            handle = self.run_viewer('bare', model, data, events=events)
        self.assertTrue(np.allclose(handle.times, [0.01, 0.01, 0, 0.01, 0.02]))
        self.assertTrue(np.allclose(calls, [0, 0, 0.01]))

    def test_scene_without_hook_advances_and_uses_scene_camera(self):
        model, data = self.model_data('bare')
        camera = dict(lookat=[0.2, 0.1, 0.8], distance=1.2, azimuth=210, elevation=-35)
        with patch.object(preview, 'scene_camera', return_value=camera):
            handle = self.run_viewer('bare', model, data)
        self.assertAlmostEqual(data.time, 0.05)
        self.assertTrue(np.allclose(handle.cam.lookat, camera['lookat']))
        self.assertEqual(handle.cam.azimuth, camera['azimuth'])

    def test_engine_callback_and_ui_actuator_targets_are_preserved(self):
        model, data = self.model_data('bare')
        before = mujoco.get_mjcb_control()
        calls = []

        def callback(model, data):
            calls.append(data.time)

        mujoco.set_mjcb_control(callback)
        try:
            actuator = model.actuator('joint1L').id
            self.run_viewer('bare', model, data,
                            events={1: lambda h: data.ctrl.__setitem__(actuator, 0.2)})
            self.assertEqual(data.ctrl[actuator], 0.2)
            self.assertTrue(calls)
            self.assertIs(mujoco.get_mjcb_control(), callback)
        finally:
            mujoco.set_mjcb_control(before)

    def test_viewer_closes_when_scene_hook_fails(self):
        model, data = self.model_data('bare')
        handle = FakeViewer(model, data, key_callback=lambda key: None)
        with patch.object(mujoco.viewer, 'launch_passive', return_value=handle), \
             patch.object(preview, 'scene_step', return_value=Mock(side_effect=RuntimeError('hook failed'))):
            with self.assertRaisesRegex(RuntimeError, 'hook failed'):
                preview._launch_interactive('bare', model, data)
        self.assertTrue(handle.closed)


if __name__ == '__main__':
    unittest.main()
