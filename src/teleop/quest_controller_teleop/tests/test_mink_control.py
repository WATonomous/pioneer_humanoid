"""mink teleop IK (mink_control.py) on pioneer_bimanual_arm_8dof: kinematics, tracking, limits, speed caps, keep-out."""
from pathlib import Path
import sys
import time
import unittest
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2]/"src/pioneer_humanoid")]
try:
    import mink  # noqa: F401
    import mujoco
except ImportError:  # pragma: no cover
    mink = None

from pioneer_humanoid.arm_8dof_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS

SIDES = (("left", LEFT_ARM_JOINTS), ("right", RIGHT_ARM_JOINTS))


@unittest.skipIf(mink is None, "mink / mujoco not installed")
class MinkIKTests(unittest.TestCase):
    def solver(self, side, names, **kw):
        from mink_control import PioneerMinkIK
        s = PioneerMinkIK(names, .9, 3.6, **kw)
        limits = s.physical_limits.copy()
        limits[3, 0] = np.radians(10.)
        s.set_teleop_limits(limits)
        return s

    @staticmethod
    def target(position, rotation):
        return SimpleNamespace(active=True, position=position, rotation=rotation, saturation=0.)

    def test_model_matches_urdf_kinematics(self):
        rng = np.random.default_rng(0)
        for side, names in SIDES:
            s = self.solver(side, names)
            d = mujoco.MjData(s.model)
            for _ in range(50):
                q = rng.uniform(*s.physical_limits.T)
                d.qpos[:] = q
                mujoco.mj_kinematics(s.model, d)
                p, r, _, _ = s.kinematics(q)
                np.testing.assert_allclose(d.site("anchor").xpos, p, atol=1e-5)
                np.testing.assert_allclose(d.site("tool").xmat.reshape(3, 3), r, atol=1e-5)   # gripper frame

    def test_reaches_reachable_poses_within_limits(self):
        rng = np.random.default_rng(1)
        for side, names in SIDES:
            s = self.solver(side, names)
            errors = []
            for _ in range(10):
                q_goal = np.clip(s.rest + rng.uniform(-.6, .6, 7), *s.limits.T)
                p, r, _, _ = s.kinematics(q_goal)
                s.command = s.rest.copy()
                for _ in range(400):
                    q = s.solve(None, self.target(p, r), .01)
                    self.assertTrue(np.all(q >= s.limits[:, 0] - 1e-9) and np.all(q <= s.limits[:, 1] + 1e-9))
                errors.append(np.linalg.norm(s.kinematics(q)[0] - p))
            self.assertLess(np.median(errors), .005)   # the home pull leaves ~1 mm (7th joint)

    def test_speed_caps_hold_every_tick(self):
        side, names = SIDES[0]
        s = self.solver(side, names)
        p, r, _, _ = s.kinematics(s.rest)
        far = self.target(p + np.array([.2, .1, .15]), r)
        q_prev = s.command.copy()
        for _ in range(50):
            q = s.solve(None, far, .01)
            self.assertTrue(np.all(np.abs(q - q_prev) <= s.velocity * .01 + 1e-9))
            q_prev = q

    def test_joint_outside_teleop_limit_returns_at_speed_cap(self):
        side, names = SIDES[0]
        s = self.solver(side, names)
        s.command = s.rest.copy()
        s.command[3] = s.limits[3, 0] - .3           # elbow 0.3 rad past its teleop limit (e.g. after a reset)
        p, r, _, _ = s.kinematics(s.command)
        before = s.command[3]
        q = s.solve(None, self.target(p, r), .01)
        self.assertGreater(q[3], before)
        self.assertLessEqual(q[3] - before, s.velocity[3] * .01 + 1e-9)

    def test_inactive_target_holds(self):
        side, names = SIDES[0]
        s = self.solver(side, names)
        np.testing.assert_allclose(s.solve(None, SimpleNamespace(active=False), .01), s.rest)

    def test_body_keepout_keeps_arm_out_of_the_torso_box(self):
        from mink_control import TORSO_CENTRE, TORSO_HALF
        side, names = SIDES[0]
        s = self.solver(side, names, body_keepout=True)
        p, r, _, _ = s.kinematics(s.rest)
        inward = self.target(np.array([.10, 0., -.15]), r)   # inside the box: the arm must stop outside it
        for _ in range(400):
            q = s.solve(None, inward, .01)
        d = mujoco.MjData(s.model)
        d.qpos[:] = q
        mujoco.mj_forward(s.model, d)
        torso = s.model.geom("torso").id
        for name in ("upper_arm", "forearm", "wrist"):
            g = s.model.geom(name).id
            dist = mujoco.mj_geomDistance(s.model, d, g, torso, 1., None)
            self.assertGreater(dist, .01, name)

    def test_solve_is_fast_enough_for_100_hz(self):
        side, names = SIDES[0]
        s = self.solver(side, names, body_keepout=True)
        p, r, _, _ = s.kinematics(s.rest)
        t = self.target(p + np.array([.1, .05, .1]), r)
        start = time.perf_counter()
        for _ in range(200):
            s.solve(None, t, .01)
        self.assertLess((time.perf_counter() - start) / 200, .005)


if __name__ == "__main__":
    unittest.main()
