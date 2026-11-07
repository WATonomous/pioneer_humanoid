"""Position-IK limits, mathematical reference checks, and Pioneer reach regressions."""
from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2]/"src/pioneer_humanoid")]
from control import Target, UrdfPioneerIK, bounded_dls, dls, rotation_exp, rotation_vector
from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS


class BoundedIKTests(unittest.TestCase):
    def test_blocked_joint_redistributes_motion_instead_of_discarding_it(self):
        j = np.array([[1., 1., 0., 0.], [0., 1., 1., 0.], [0., 0., 0., 1.]])
        error = np.array([.05, 0., 0.])
        lower, upper = np.full(4, -.1), np.array([0., .1, .1, .1])
        unconstrained, _ = dls(j, error)
        old = np.clip(unconstrained, lower, upper)
        new, _ = bounded_dls(j, error, lower, upper)
        self.assertLess(np.linalg.norm(j @ new-error), np.linalg.norm(j @ old-error)/20)
        self.assertLessEqual(new[0], 0.)

    def test_active_set_matches_exhaustive_box_qp_reference(self):
        rng = np.random.default_rng(81)
        for _ in range(50):
            j, error = rng.normal(size=(3, 4))*.2, rng.normal(size=3)*.05
            lower, upper = -rng.uniform(.001, .08, 4), rng.uniform(.001, .08, 4)
            upper[0] = 0.  # A joint already at its physical upper stop.
            if rng.random() < .5:
                lower[1] = upper[1] = 0.  # A fixed joint as well.
            step, rho = bounded_dls(j, error, lower, upper)
            h = j.T @ j+(.01**2+.12**2*rho**2)*np.eye(4)
            rhs = j.T @ error
            best = float("inf")
            # Independent oracle: enumerate every face of this 4-D box.
            for state in product((-1, 0, 1), repeat=4):
                state = np.array(state)
                free, fixed = np.flatnonzero(state == 0), np.flatnonzero(state != 0)
                x = np.where(state == -1, lower, upper).copy()
                if len(free):
                    x[free] = np.linalg.solve(h[np.ix_(free, free)], rhs[free]-h[np.ix_(free, fixed)] @ x[fixed])
                if np.all(x >= lower-1e-12) and np.all(x <= upper+1e-12):
                    best = min(best, .5*x @ h @ x-rhs @ x)
            self.assertAlmostEqual(.5*step @ h @ step-rhs @ step, best, places=11)
            self.assertTrue(np.all(step >= lower-1e-12))
            self.assertTrue(np.all(step <= upper+1e-12))

    def test_unbounded_case_matches_article_dls_with_elbow_bias(self):
        rng = np.random.default_rng(19)
        for _ in range(15):
            j = rng.normal(size=(3, 4))*.3
            error = rng.normal(size=3)*.01
            bias = np.array([0., 0., 0., .2])
            reference, rho = dls(j, error, bias=bias)
            actual, actual_rho = bounded_dls(j, error, np.full(4, -100.), np.full(4, 100.), bias=bias)
            np.testing.assert_allclose(actual, reference, atol=1e-12)
            self.assertEqual(rho, actual_rho)

    def test_both_arm_jacobians_match_finite_differences(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            solver = UrdfPioneerIK(names)
            q = solver.rest+np.array([.1, -.1, .15, .1, -.2, .1])
            p, r, jp, jr = solver.kinematics(q)
            for i in range(6):
                perturbed = q.copy()
                perturbed[i] += 1e-6
                pp, rr, _, _ = solver.kinematics(perturbed)
                np.testing.assert_allclose((pp-p)/1e-6, jp[:, i], atol=1e-6)
                np.testing.assert_allclose(rotation_vector(rr @ r.T)/1e-6, jr[:, i], atol=1e-6)
            np.testing.assert_allclose(jp[:, 4:], 0., atol=1e-12)

    def test_inward_reach_near_shoulder_stop_and_slower_speed_caps(self):
        for names, sign in ((RIGHT_ARM_JOINTS, 1.), (LEFT_ARM_JOINTS, -1.)):
            solver = UrdfPioneerIK(names, velocity=.9, wrist_velocity=3.6, orientation_mode="full")
            q = solver.rest.copy()
            p, r, _, _ = solver.kinematics(q)
            target = Target(active=True, position=p+[0., sign*.15, 0.], rotation=r)
            for _ in range(400):
                previous = q.copy()
                q = solver.solve(q, target, .02)
                self.assertTrue(np.all(np.abs(q-previous) <= solver.velocity*.02+1e-10))
                self.assertTrue(np.all(q >= solver.limits[:, 0]))
                self.assertTrue(np.all(q <= solver.limits[:, 1]))
            achieved, _, _, _ = solver.kinematics(q)
            self.assertLess(np.linalg.norm(achieved-target.position), .002)
            self.assertIn(2, solver.diagnostics["limited_joints"])

    def test_wrist_twist_still_preserves_position_in_both_directions(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            for sign in (-1., 1.):
                solver = UrdfPioneerIK(names, wrist_velocity=3.6)
                q = solver.rest.copy()
                p, r, _, jr = solver.kinematics(q)
                target = Target(active=True, position=p, rotation=rotation_exp(jr[:, -1]*sign*.3) @ r)
                for _ in range(100):
                    q = solver.solve(q, target, .02)
                pp, rr, _, _ = solver.kinematics(q)
                self.assertLess(np.linalg.norm(pp-p), .001)
                self.assertLess(np.linalg.norm(rotation_vector(target.rotation @ rr.T)), np.radians(1.))


if __name__ == "__main__":
    unittest.main()
