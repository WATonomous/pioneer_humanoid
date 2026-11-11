"""Controller pose filter (input_filter.py): jitter, jump capping, lag, reset, and the driver wiring."""
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2]/"src/pioneer_humanoid")]
from control import Input, rotation_exp, rotation_vector
from input_filter import OneEuroPose

DT = 1 / 72   # Quest frame period


class FilterTests(unittest.TestCase):
    def test_still_hand_jitter_is_roughly_halved(self):
        rng = np.random.default_rng(0)
        f = OneEuroPose()
        raw, out = [], []
        for k in range(2000):
            p = np.array([0., 1., 0.]) + rng.normal(0, .0005, 3)
            raw.append(p)
            out.append(f(p, np.eye(3), k*DT)[0])
        ratio = np.std(out[200:], axis=0).mean() / np.std(raw[200:], axis=0).mean()
        self.assertLess(ratio, .6)

    def test_tracking_jump_is_spread_at_the_speed_cap(self):
        f = OneEuroPose(max_speed=1.)
        f(np.zeros(3), np.eye(3), 0.)
        positions = [f(np.array([.15, 0., 0.]), np.eye(3), k*DT)[0] for k in range(1, 40)]
        steps = np.linalg.norm(np.diff(np.vstack([np.zeros(3)] + positions), axis=0), axis=1)
        self.assertLessEqual(steps.max(), 1.*DT + 1e-9)
        self.assertAlmostEqual(positions[-1][0], .15, places=3)   # still arrives

    def test_turn_rate_is_capped(self):
        for beta_rot in (0., 8.):
            f = OneEuroPose(max_turn_rate=6., beta_rot=beta_rot)
            f(np.zeros(3), np.eye(3), 0.)
            _, r = f(np.zeros(3), rotation_exp([0., 0., 2.]), DT)
            self.assertLessEqual(np.linalg.norm(rotation_vector(r)), 6.*DT + 1e-9)

    def test_wrist_twist_lags_little(self):
        f = OneEuroPose()
        w = 3.   # rad/s, a brisk wrist roll with the hand still
        lag = []
        for k in range(150):
            target = rotation_exp([w*k*DT, 0., 0.])
            _, r = f(np.zeros(3), target, k*DT)
            lag.append(np.degrees(np.linalg.norm(rotation_vector(target @ r.T))))
        self.assertLess(max(lag[60:]), 6.)

    def test_moving_hand_lags_little(self):
        f = OneEuroPose()
        v = .5   # m/s, an ordinary teleop motion
        lag = []
        for k in range(200):
            p = np.array([v*k*DT, 0., 0.])
            lag.append(p[0] - f(p, np.eye(3), k*DT)[0][0])
        self.assertLess(max(lag[100:]), .015)   # under 1.5 cm behind at 0.5 m/s

    def test_reset_passes_next_sample_through_and_repeat_timestamps_hold(self):
        f = OneEuroPose()
        f(np.zeros(3), np.eye(3), 0.)
        held, _ = f(np.ones(3), np.eye(3), 0.)          # same time stamp: hold
        np.testing.assert_allclose(held, np.zeros(3))
        f.reset()
        p, _ = f(np.ones(3), np.eye(3), 1.)
        np.testing.assert_allclose(p, np.ones(3))

    def test_rejects_bad_parameters(self):
        for kwargs in ({"min_cutoff": 0.}, {"beta": -1.}, {"max_speed": float("nan")}):
            with self.assertRaises(ValueError):
                OneEuroPose(**kwargs)


class DriverWiringTests(unittest.TestCase):
    def test_filter_is_applied_and_reset_on_tracking_loss(self):
        from scene_driver import ControllerSceneDriver
        from pioneer_humanoid.arm_8dof_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        driver = ControllerSceneDriver({"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS}, 1.5, 1.,
                                       Path(folder.name)/"pivot.json", input_filter_factory=OneEuroPose)
        home = {side: solver.rest.copy() for side, solver in driver.solvers.items()}

        def send(position, at):
            control = Input(np.asarray(position, float), np.eye(3), False, 0., False, False)
            with driver.shared.lock:
                driver.shared.packet = {"head": np.eye(3), "left": control, "right": None}
                driver.shared.received = at
                driver.shared.sequence += 1

        now = time.monotonic()
        send([0., 1., 0.], now)
        driver.update(home)
        self.assertIsNotNone(driver.filters["left"].position)
        self.assertIsNone(driver.filters["right"].position)        # no controller: nothing held
        send([.3, 1., 0.], now + DT)                                  # a 30 cm one-frame jump
        driver.update(home)
        moved = np.linalg.norm(driver.filters["left"].position - [0., 1., 0.])
        self.assertLessEqual(moved, 1.*DT + 1e-9)                     # capped, not passed through
        send([.3, 1., 0.], now - 1.)                                  # stale packet: tracking lost
        driver.update(home)
        self.assertIsNone(driver.filters["left"].position)


if __name__ == "__main__":
    unittest.main()
