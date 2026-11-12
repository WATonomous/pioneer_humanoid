"""Controller orientation references must survive lag, lead limiting and holds."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2]/"src/pioneer_humanoid")]
from control import Input, Target, UrdfPioneerIK, rotation_exp, rotation_vector
from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS


def controller(rotation, grip=True, precision=False):
    return Input(np.zeros(3), rotation, grip, 0., precision, False)


class OrientationReferenceTests(unittest.TestCase):
    def engage(self, target, position, rotation):
        target.update(controller(np.eye(3), False), np.eye(3), position, rotation)
        target.update(controller(np.eye(3)), np.eye(3), position, rotation)

    def test_fast_rotation_hold_and_slow_return_reproduce_controller_pose(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            solver = UrdfPioneerIK(names, wrist_velocity=3.6)
            q = solver.rest.copy()
            p, initial_r, _, jr = solver.kinematics(q)
            axis = jr[:, 5]
            target = Target(calibration=np.eye(3))
            self.engage(target, p, initial_r)
            for degree in [60.]*150+list(np.linspace(60., 0., 151))+[0.]*100:
                p, r, _, _ = solver.kinematics(q)
                rc = rotation_exp(axis*np.radians(degree))
                target.update(controller(rc), np.eye(3), p, r)
                np.testing.assert_allclose(target.requested_rotation, rc @ initial_r, atol=1e-12)
                self.assertLessEqual(np.linalg.norm(rotation_vector(target.rotation @ r.T)), .6+1e-12)
                q = solver.solve(q, target, .02)
                if degree == 60.:
                    held_q = q.copy()
            # Old clipping lost ~25.6 degrees from this gesture and overshot
            # the return by that amount despite perfectly noise-free input.
            self.assertAlmostEqual(abs(np.degrees(held_q[5]-solver.rest[5])), 60., places=5)
            np.testing.assert_allclose(q, solver.rest, atol=1e-6)

    def test_stationary_request_is_independent_of_measured_robot_motion(self):
        target = Target(calibration=np.eye(3))
        p, initial_r = np.zeros(3), rotation_exp(np.array([.1, .2, -.1]))
        self.engage(target, p, initial_r)
        rc = rotation_exp(np.array([.7, .3, -.1]))
        expected = rc @ initial_r
        for angle in np.linspace(0., -.7, 100):
            measured = rotation_exp(np.array([0., angle, .2])) @ initial_r
            target.update(controller(rc), np.eye(3), p, measured)
            np.testing.assert_allclose(target.requested_rotation, expected, atol=1e-12)
            self.assertLessEqual(np.linalg.norm(rotation_vector(target.rotation @ measured.T)), .6+1e-12)
        target.update(controller(rc), np.eye(3), p, expected)
        np.testing.assert_allclose(target.rotation, expected, atol=1e-12)
        self.assertFalse(target.orientation_limited)
        self.assertEqual(target.controller_rotation_delta_deg, 0.)

    def test_multi_axis_mapping_matches_relative_pose_after_large_excursions(self):
        mapping = rotation_exp(np.array([.3, -.2, .4]))
        initial_r = rotation_exp(np.array([-.2, .5, .1]))
        target = Target(calibration=mapping)
        self.engage(target, np.zeros(3), initial_r)
        rng = np.random.default_rng(14)
        for _ in range(100):
            rc = rotation_exp(rng.normal(size=3))
            target.update(controller(rc), np.eye(3), np.zeros(3), initial_r)
            np.testing.assert_allclose(target.requested_rotation,
                                       mapping @ rc @ mapping.T @ initial_r, atol=1e-12)
        target.update(controller(np.eye(3)), np.eye(3), np.zeros(3), initial_r)
        np.testing.assert_allclose(target.requested_rotation, initial_r, atol=1e-12)

    def test_precision_and_reengagement_start_from_the_correct_reference(self):
        target = Target(calibration=np.eye(3))
        self.engage(target, np.zeros(3), np.eye(3))
        rc = rotation_exp(np.array([0., .8, 0.]))
        target.update(controller(rc, precision=True), np.eye(3), np.zeros(3), np.eye(3))
        np.testing.assert_allclose(target.requested_rotation, rotation_exp(np.array([0., .2, 0.])), atol=1e-12)
        measured = rotation_exp(np.array([-.1, .3, .2]))
        target.lose_tracking()
        target.update(controller(rc, False), np.eye(3), np.zeros(3), measured)
        target.update(controller(rc), np.eye(3), np.zeros(3), measured)
        np.testing.assert_allclose(target.requested_rotation, measured, atol=1e-12)
        self.assertEqual(target.engagement_count, 2)

    def test_held_unreachable_rotation_settles_without_proximal_creep(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            solver = UrdfPioneerIK(names, wrist_velocity=3.6)
            q = solver.rest.copy()
            initial = q.copy()
            p, initial_r, _, _ = solver.kinematics(q)
            target = Target(calibration=np.eye(3))
            self.engage(target, p, initial_r)
            rc = rotation_exp(initial_r @ np.array([.8, .3, .2]))
            previous = q.copy()
            for i in range(600):
                p, r, _, _ = solver.kinematics(q)
                target.update(controller(rc), np.eye(3), p, r)
                q = solver.solve(q, target, .02)
                np.testing.assert_allclose(q[:4], initial[:4], atol=1e-8)
                if i > 500:
                    np.testing.assert_allclose(q, previous, atol=1e-8)
                previous = q.copy()
            self.assertGreater(solver.diagnostics["requested_orientation_error_deg"], 10.)


if __name__ == "__main__":
    unittest.main()
