import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pose_math as pm


# Rotation matrices about one axis, angle in degrees
def Rx(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class testPoseMath(unittest.TestCase):
    def test_rho(self):
        # rho takes a position error (3,) in meters and a rotation error (3,) in radians
        def pos(mm):
            return np.array([mm / 1000, 0.0, 0.0])

        def rot(deg):
            return np.array([0.0, 0.0, np.radians(deg)])

        zero = np.zeros(3)

        self.assertAlmostEqual(pm.rho(pos(1), zero), 1)
        self.assertAlmostEqual(pm.rho(zero, rot(0.2)), 1)
        self.assertAlmostEqual(pm.rho(pos(2), zero), 2)
        self.assertTrue(pm.rho(pos(0.5), zero) < 1)
        self.assertTrue(pm.rho(zero, rot(0.5)) > 1)
        self.assertTrue(pm.rho(zero, rot(0.1)) < 1)
        # both errors present: the larger one wins
        self.assertAlmostEqual(pm.rho(pos(2), rot(0.1)), 2)
        self.assertAlmostEqual(pm.rho(pos(0.5), rot(0.4)), 2)

    def test_rho_uses_vector_length(self):
        # 3 mm and 4 mm on two axes is a 5 mm error, so rho = 5
        self.assertAlmostEqual(pm.rho(np.array([0.003, 0.004, 0.0]), np.zeros(3)), 5)

    def test_rot_log_identity_is_zero(self):
        self.assertTrue(np.allclose(pm.rot_log(np.eye(3)), 0))

    def test_rot_log_single_axis(self):
        # the rotation vector is the axis times the angle (radians)
        self.assertTrue(np.allclose(pm.rot_log(Rx(30)), [np.radians(30), 0, 0]))
        self.assertTrue(np.allclose(pm.rot_log(Ry(30)), [0, np.radians(30), 0]))
        self.assertTrue(np.allclose(pm.rot_log(Rz(30)), [0, 0, np.radians(30)]))

    def test_rot_log_sign(self):
        self.assertTrue(np.allclose(pm.rot_log(Rz(-30)), [0, 0, -np.radians(30)]))

    def test_rot_log_tiny_and_near_half_turn(self):
        self.assertTrue(np.allclose(pm.rot_log(Rz(1e-5)), [0, 0, np.radians(1e-5)], atol=1e-12))
        self.assertTrue(np.allclose(pm.rot_log(Rz(179)), [0, 0, np.radians(179)]))

    def test_pose_errors_zero_at_goal(self):
        p = np.array([0.3, 0.1, -0.2])
        e_pos, e_rot = pm.pose_errors(p, Rz(40), p, Rz(40))
        self.assertTrue(np.allclose(e_pos, 0))
        self.assertTrue(np.allclose(e_rot, 0))

    def test_pose_errors_unrotated_gripper(self):
        e_pos, e_rot = pm.pose_errors(np.zeros(3), np.eye(3), np.array([0.01, 0, 0]), np.eye(3))
        self.assertTrue(np.allclose(e_pos, [0.01, 0, 0]))
        self.assertTrue(np.allclose(e_rot, 0))

    def test_pose_errors_are_in_the_gripper_frame(self):
        # gripper turned 90 deg about z: base +x is the gripper's -y
        e_pos, _ = pm.pose_errors(np.zeros(3), Rz(90), np.array([0.01, 0, 0]), Rz(90))
        self.assertTrue(np.allclose(e_pos, [0, -0.01, 0]))

    def test_rotation_error_is_in_the_gripper_frame(self):
        # goal = gripper rotated a further 10 deg about its own x axis
        _, e_rot = pm.pose_errors(np.zeros(3), Rz(90), np.zeros(3), Rz(90) @ Rx(10))
        self.assertTrue(np.allclose(e_rot, [np.radians(10), 0, 0]))

    def test_scaled_divides_and_clips(self):
        out = pm.scaled(np.array([0.005, -1.0, 0.0]), np.array([0.0, 0.0, 5.0]), (0.01, 0.1))
        self.assertEqual(out.shape, (6,))
        self.assertTrue(np.allclose(out, [0.5, -1.0, 0.0, 0.0, 0.0, 1.0]))

    def test_pose_feedback_paper_example(self):
        # 0.2 deg of rotation fills 80% of the fine range (0.25 deg) and 1% of the coarse range (20 deg)
        coarse, fine = pm.pose_feedback(np.zeros(3), np.eye(3), np.zeros(3), Rz(0.2))
        self.assertAlmostEqual(fine[5], 0.8, places=4)
        self.assertAlmostEqual(coarse[5], 0.01, places=4)
        self.assertTrue(np.allclose(fine[:3], 0))

    def test_pose_feedback_position_scales_and_clipping(self):
        # 5 mm: half of the coarse range (1 cm), far past the fine range (0.5 mm) so clipped to 1
        coarse, fine = pm.pose_feedback(np.zeros(3), np.eye(3), np.array([0.005, 0, 0]), np.eye(3))
        self.assertEqual(coarse.shape, (6,))
        self.assertEqual(fine.shape, (6,))
        self.assertAlmostEqual(coarse[0], 0.5)
        self.assertEqual(fine[0], 1.0)
        # a large negative error clips to -1 in both
        coarse, fine = pm.pose_feedback(np.zeros(3), np.eye(3), np.array([-1.0, 0, 0]), np.eye(3))
        self.assertEqual(coarse[0], -1.0)
        self.assertEqual(fine[0], -1.0)

    def test_keypoint_errors_zero_at_goal(self):
        out = pm.keypoint_errors(np.array([0.1, 0.2, 0.3]), Rz(25), np.array([0.1, 0.2, 0.3]), Rz(25))
        self.assertEqual(out.shape, (9,))
        self.assertTrue(np.allclose(out, 0))

    def test_keypoint_errors_pure_translation(self):
        # every point moves with the gripper: 1 cm along x for all three
        out = pm.keypoint_errors(np.zeros(3), np.eye(3), np.array([0.01, 0, 0]), np.eye(3))
        self.assertTrue(np.allclose(out, [0.01, 0, 0] * 3))

    def test_keypoint_errors_pure_rotation(self):
        # goal turned 90 deg about z, same position. The points are 0.3 m out along x, y, z:
        # the x point goes to +y, the y point goes to -x, the z point doesn't move
        out = pm.keypoint_errors(np.zeros(3), np.eye(3), np.zeros(3), Rz(90))
        self.assertTrue(np.allclose(out, [-0.3, 0.3, 0, -0.3, -0.3, 0, 0, 0, 0]))

    def test_best_so_far_first_step(self):
        best, features = pm.best_so_far(None, 10.0)
        self.assertEqual(best, 10.0)
        self.assertTrue(np.allclose(features, [0.5, 0.0]))     # 10/20 and no change yet

    def test_best_so_far_gets_worse(self):
        best, features = pm.best_so_far(10.0, 15.0)
        self.assertEqual(best, 10.0)                           # the best never goes up
        self.assertTrue(np.allclose(features, [0.5, 0.25]))    # 10/20 and (15-10)/20

    def test_best_so_far_gets_better(self):
        best, features = pm.best_so_far(10.0, 4.0)
        self.assertEqual(best, 4.0)
        self.assertTrue(np.allclose(features, [0.5, 0.0]))     # a negative change clips to 0

    def test_best_so_far_clips_to_one(self):
        _, features = pm.best_so_far(100.0, 500.0)
        self.assertTrue(np.allclose(features, [1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
