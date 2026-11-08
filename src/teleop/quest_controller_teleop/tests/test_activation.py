"""Controller activation regressions without Isaac, networking, or a GPU."""
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2]/"src/pioneer_humanoid")]
from control import Input
from scene_driver import ControllerSceneDriver
from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS


class ActivationTests(unittest.TestCase):
    def driver(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        driver = ControllerSceneDriver({"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS},
                                       1.5, 1., Path(folder.name)/"pivot.json")
        home = {side: solver.rest.copy() for side, solver in driver.solvers.items()}
        return driver, home

    def test_default_is_article_solver_with_article_target(self):
        from article_control import ArticleTarget, PioneerArticleIK
        driver, _ = self.driver()
        self.assertTrue(all(isinstance(s, PioneerArticleIK) for s in driver.solvers.values()))
        self.assertTrue(all(isinstance(t, ArticleTarget) for t in driver.targets.values()))

    def test_io_is_serviced_before_consuming_input_on_a_physics_tick(self):
        driver, home = self.driver()
        calls = []
        class Task:
            def done(self): return False
            def cancel(self): pass
        def schedule(coroutine):
            coroutine.close()
            return Task()
        def pump():
            calls.append(1)
            control = Input(np.array([0., 1., 0.]), np.eye(3), False, 0., False)
            with driver.shared.lock:
                driver.shared.packet = {"head": np.eye(3), "left": control, "right": None}
                driver.shared.received = time.monotonic()
                driver.shared.sequence += 1
        driver.start(schedule=schedule, pump=pump)
        try:
            driver.update(home)
            self.assertEqual(len(calls), 1)
            self.assertTrue(driver.targets["left"].active)
            driver.update(home)
            self.assertEqual(len(calls), 2)
        finally:
            driver.close()

    def test_kit_pump_is_not_nested_inside_a_running_event_loop(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch
        driver, home = self.driver()
        calls = []
        class Task:
            def done(self): return False
            def cancel(self): pass
        def schedule(coroutine):
            coroutine.close()
            return Task()
        kit = SimpleNamespace(run_coroutine=schedule)
        loop = SimpleNamespace(run_once=lambda: calls.append(1))
        with patch.dict(sys.modules, {"omni.kit.async_engine": kit}), patch.object(asyncio, "get_event_loop", return_value=loop):
            driver.start()
        async def nested():
            driver.update(home)
        try:
            asyncio.run(nested())
            self.assertEqual(calls, [])
            driver.update(home)
            self.assertEqual(calls, [1])
        finally:
            driver.close()

    def send(self, driver, home, *, grip=False, x=0., left=True, head=True, age=0., calibrate=False, reset=0):
        control = Input(np.array([x, 1., 0.]), np.eye(3), grip, .4, calibrate)
        with driver.shared.lock:
            driver.shared.packet = {"head": np.eye(3) if head else None,
                                    "left": control if left else None, "right": None}
            driver.shared.received = time.monotonic()-age
            driver.shared.sequence += 1
            driver.shared.reset_sequence = reset
        return driver.update(home), control

    def test_always_mode_engages_without_button_and_consumes_relative_motion(self):
        driver, home = self.driver()
        _, original = self.send(driver, home)
        target = driver.targets["left"]
        p, r, _, _ = driver.solvers["left"].kinematics(home["left"])
        self.assertTrue(target.active)
        self.assertFalse(original.clutch)  # shared packet was not mutated
        np.testing.assert_allclose(target.position, p)
        np.testing.assert_allclose(target.rotation, r)
        self.send(driver, home, x=.01)
        np.testing.assert_allclose(target.position, p+[0., -.015, 0.], atol=1e-10)
        self.assertEqual(driver.grippers["left"], .4)

    def test_always_mode_stale_or_lost_tracking_pauses_then_reanchors_without_jump(self):
        for loss in ({"age": 1.}, {"left": False}, {"head": False}):
            with self.subTest(loss=loss):
                driver, home = self.driver()
                self.send(driver, home)
                self.send(driver, home, x=.02)
                self.send(driver, home, **loss)
                self.assertFalse(driver.targets["left"].active)
                # Reacquisition far from the previous controller pose must not
                # inject that displacement into the target.
                self.send(driver, home, x=3.)
                p, r, _, _ = driver.solvers["left"].kinematics(home["left"])
                np.testing.assert_allclose(driver.targets["left"].position, p)
                np.testing.assert_allclose(driver.targets["left"].rotation, r)
                self.send(driver, home, x=3.002)
                np.testing.assert_allclose(driver.targets["left"].position, p+[0., -.003, 0.], atol=1e-10)

    def test_latched_loss_between_frames_is_not_hidden_by_reacquisition(self):
        driver, home = self.driver()
        self.send(driver, home)
        self.send(driver, home, x=.01)
        with driver.shared.lock:
            driver.shared.interrupted["left"] = True
        self.send(driver, home, x=2.)
        p, _, _, _ = driver.solvers["left"].kinematics(home["left"])
        np.testing.assert_allclose(driver.targets["left"].position, p)

    def test_scene_reset_reanchors_always_mode_without_a_button(self):
        driver, home = self.driver()
        self.send(driver, home)
        reset, _ = self.send(driver, home, reset=1, x=.1)
        self.assertTrue(reset)
        driver.reset(home)
        self.assertFalse(driver.targets["left"].active)
        reset, _ = self.send(driver, home, reset=1, x=2.)
        self.assertFalse(reset)
        p, _, _, _ = driver.solvers["left"].kinematics(home["left"])
        self.assertTrue(driver.targets["left"].active)
        np.testing.assert_allclose(driver.targets["left"].position, p)

    def test_calibration_pauses_always_mode_even_if_physical_grip_is_held(self):
        driver, home = self.driver()
        self.send(driver, home)
        self.send(driver, home, grip=True, calibrate=True, x=.1)
        self.assertTrue(driver.calibrating["left"])
        self.assertFalse(driver.targets["left"].active)
        self.send(driver, home, x=1.)  # short/invalid fit still resumes safely
        p, _, _, _ = driver.solvers["left"].kinematics(home["left"])
        self.assertTrue(driver.targets["left"].active)
        np.testing.assert_allclose(driver.targets["left"].position, p)



if __name__ == "__main__":
    unittest.main()
