"""Distinguish wrist geometry from unintended controller-pivot translation."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2]/"src/pioneer_humanoid")]
from control import Input, Target, UrdfPioneerIK, rotation_exp, rotation_vector
from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS


class WristOrientationTests(unittest.TestCase):
    def test_each_distal_axis_tracks_without_proximal_motion(self):
        # Check every intermediate command, not just eventual convergence:
        # compensating proximal excursions would also hurt teleop dexterity.
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            for axis in (4, 5):
                for sign in (-1., 1.):
                    with self.subTest(arm=names[0], joint=axis+1, sign=sign):
                        solver = UrdfPioneerIK(names, wrist_velocity=3.6)
                        initial = solver.rest.copy()
                        q = initial.copy()
                        p, r, _, jr = solver.kinematics(q)
                        target = Target(active=True, position=p,
                                        rotation=rotation_exp(jr[:, axis]*sign*np.radians(20)) @ r)
                        for _ in range(100):
                            previous = q.copy()
                            q = solver.solve(q, target, .02)
                            np.testing.assert_allclose(q[:4], initial[:4], atol=1e-5)
                            self.assertTrue(np.all(np.abs(q-previous) <= solver.velocity*.02+1e-10))
                            pp, rr, _, _ = solver.kinematics(q)
                            self.assertLess(np.linalg.norm(pp-p), 1e-5)
                        self.assertLess(np.linalg.norm(rotation_vector(target.rotation @ rr.T)), np.radians(.1))

    def test_third_rotation_axis_requires_proximal_motion(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            solver = UrdfPioneerIK(names, wrist_velocity=3.6, orientation_mode="full")
            q = solver.rest.copy()
            initial = q.copy()
            p, r, _, jr = solver.kinematics(q)
            # This direction is orthogonal to both available distal axes.
            direction = np.cross(jr[:, 4], jr[:, 5])
            direction /= np.linalg.norm(direction)
            distal, *_ = np.linalg.lstsq(jr[:, 4:], direction, rcond=None)
            self.assertGreater(np.linalg.norm(jr[:, 4:] @ distal-direction), .99)
            target = Target(active=True, position=p,
                            rotation=rotation_exp(direction*np.radians(20)) @ r)
            for _ in range(200):
                q = solver.solve(q, target, .02)
                pp, rr, _, _ = solver.kinematics(q)
                self.assertLess(np.linalg.norm(pp-p), .001)
                self.assertTrue(np.all(q >= solver.limits[:, 0]))
                self.assertTrue(np.all(q <= solver.limits[:, 1]))
            self.assertGreater(np.linalg.norm(q[:4]-initial[:4]), np.radians(10))
            self.assertLess(np.linalg.norm(rotation_vector(target.rotation @ rr.T)), np.radians(.1))

    def test_pivot_calibration_removes_translation_from_stationary_wrist_twist(self):
        pivot = np.array([.1, 0., 0.])
        wrist = np.array([0., 1., 0.])
        rotated = rotation_exp(np.array([0., 0., np.radians(20)]))
        positions = []
        for saved_pivot in (np.zeros(3), pivot):
            target = Target(calibration=np.eye(3), pivot=saved_pivot)
            def control(rotation, clutch):
                return Input(wrist-rotation @ pivot, rotation, clutch, 0., False)
            target.update(control(np.eye(3), False), np.eye(3), wrist, np.eye(3))
            target.update(control(np.eye(3), True), np.eye(3), wrist, np.eye(3))
            target.update(control(rotated, True), np.eye(3), wrist, np.eye(3))
            positions.append(target.position.copy())
            np.testing.assert_allclose(target.rotation, rotated, atol=1e-12)
        self.assertGreater(np.linalg.norm(positions[0]-wrist), .05)
        np.testing.assert_allclose(positions[1], wrist, atol=1e-12)

    def test_default_missing_axis_does_not_creep_into_proximal_joints(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            solver = UrdfPioneerIK(names, wrist_velocity=3.6)
            initial = solver.rest.copy()
            q = initial.copy()
            p, r, _, jr = solver.kinematics(q)
            direction = np.cross(jr[:, 4], jr[:, 5])
            direction /= np.linalg.norm(direction)
            target = Target(active=True, position=p,
                            rotation=rotation_exp(direction*np.radians(20)) @ r)
            for _ in range(500):
                q = solver.solve(q, target, .02)
                np.testing.assert_allclose(q[:4], initial[:4], atol=1e-8)
            self.assertGreater(solver.diagnostics["command_orientation_error_deg"], 19.)
            self.assertEqual(solver.diagnostics["orientation_mode"], "wrist")

    def test_combined_wrist_targets_track_in_nonhome_poses_without_extra_proximal_motion(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            for sign in (-1., 1.):
                solver = UrdfPioneerIK(names, wrist_velocity=3.6)
                baseline = UrdfPioneerIK(names, wrist_velocity=3.6)
                initial = solver.rest+np.array([.15, .1, -.15, 0., .1, -.2])
                solver.command = baseline.command = initial.copy()
                q = initial.copy()
                bq = initial.copy()
                p, r, _, _ = solver.kinematics(q)
                goal = q.copy()
                goal[4:] += sign*np.array([.25, .3])
                _, goal_r, _, _ = solver.kinematics(goal)
                target = Target(active=True, position=p, rotation=goal_r)
                hold = Target(active=True, position=p, rotation=r)
                for _ in range(200):
                    previous = q.copy()
                    q = solver.solve(q, target, .02)
                    bq = baseline.solve(bq, hold, .02)
                    np.testing.assert_allclose(q[:4], bq[:4], atol=1e-9)
                    self.assertTrue(np.all(np.abs(q-previous) <= solver.velocity*.02+1e-10))
                    self.assertTrue(np.all(q >= solver.limits[:, 0]))
                    self.assertTrue(np.all(q <= solver.limits[:, 1]))
                pp, rr, _, _ = solver.kinematics(q)
                self.assertLess(np.linalg.norm(pp-p), .001)
                self.assertLess(np.linalg.norm(rotation_vector(goal_r @ rr.T)), np.radians(.1))

    def test_one_wrist_at_stop_still_allows_the_other_to_rotate(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            solver = UrdfPioneerIK(names, wrist_velocity=3.6)
            q = solver.rest.copy()
            q[4] = solver.limits[4, 1]
            solver.command = q.copy()
            p, r, _, _ = solver.kinematics(q)
            goal = q.copy()
            goal[4:] += [.15, .25]  # First wrist target exceeds its stop.
            _, goal_r, _, _ = solver.kinematics(goal)
            target = Target(active=True, position=p, rotation=goal_r)
            for _ in range(150):
                previous = q.copy()
                q = solver.solve(q, target, .02)
                self.assertTrue(np.all(np.abs(q-previous) <= solver.velocity*.02+1e-10))
                self.assertTrue(np.all(q >= solver.limits[:, 0]))
                self.assertTrue(np.all(q <= solver.limits[:, 1]))
            self.assertGreater(q[5], .15)
            pp, _, _, _ = solver.kinematics(q)
            self.assertLess(np.linalg.norm(pp-p), .001)

    def test_wrist_mode_still_reaches_inward_position_targets(self):
        for names, sign in ((LEFT_ARM_JOINTS, -1.), (RIGHT_ARM_JOINTS, 1.)):
            solver = UrdfPioneerIK(names, wrist_velocity=3.6)
            q = solver.rest.copy()
            p, r, _, _ = solver.kinematics(q)
            target = Target(active=True, position=p+[0., sign*.15, 0.], rotation=r)
            for _ in range(300):
                previous = q.copy()
                q = solver.solve(q, target, .02)
                self.assertTrue(np.all(np.abs(q-previous) <= solver.velocity*.02+1e-10))
                self.assertTrue(np.all(q >= solver.limits[:, 0]))
                self.assertTrue(np.all(q <= solver.limits[:, 1]))
            pp, _, _, _ = solver.kinematics(q)
            self.assertLess(np.linalg.norm(pp-target.position), .002)

    def test_invalid_orientation_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            UrdfPioneerIK(LEFT_ARM_JOINTS, orientation_mode="unknown")


if __name__ == "__main__":
    unittest.main()
