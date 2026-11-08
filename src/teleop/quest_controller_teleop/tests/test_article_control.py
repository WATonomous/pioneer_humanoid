"""Article rebuild: three-axis tracking, slipping clutch and asset consistency."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2]/"src/pioneer_humanoid")]
from article_control import ArticleTarget, PioneerArticleIK
from control import Input, Target, quaternion_matrix, rotation_exp, rotation_vector
from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS


def control(rotation, clutch=True, position=None):
    return Input(np.zeros(3) if position is None else position, rotation, clutch, 0., False)


def engage(target, position, rotation):
    target.update(control(np.eye(3), False), np.eye(3), position, rotation)
    target.update(control(np.eye(3)), np.eye(3), position, rotation)


class ArticleControlTests(unittest.TestCase):
    def step(self, solver, q, target):
        result = solver.solve(q, target, .02)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(np.abs(result-solver.command) < 1e-12))
        self.assertTrue(np.all(np.abs(result-q) <= solver.velocity*.02+1e-10))
        self.assertTrue(np.all(result >= solver.limits[:, 0]))
        self.assertTrue(np.all(result <= solver.limits[:, 1]))
        return result

    def test_pioneer_split_is_full_rank_at_home(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            solver = PioneerArticleIK(names)
            _, _, jp, jr = solver.kinematics(solver.rest)
            self.assertGreater(abs(np.linalg.det(jp[:, solver.position_ids])), .005)
            self.assertGreater(abs(np.linalg.det(jr[:, solver.orientation_ids])), .99)
            # Literal DK1 indices would lose one rotation direction here.
            self.assertLess(abs(np.linalg.det(jr[:, 3:6])), 1e-10)

    def test_all_three_controller_axes_track_and_hold_in_both_directions(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            for axis in np.eye(3):
                for sign in (-1., 1.):
                    with self.subTest(arm=names[0], axis=axis, sign=sign):
                        solver = PioneerArticleIK(names, wrist_velocity=3.6)
                        q = solver.rest.copy()
                        p0, r0, _, _ = solver.kinematics(q)
                        target = ArticleTarget(calibration=np.eye(3))
                        engage(target, p0, r0)
                        requested = r0 @ rotation_exp(axis*sign*np.radians(20))
                        held = None
                        for i in range(320):
                            angle = min(1., (i+1)/80)*sign*np.radians(20)
                            rc = rotation_exp((r0 @ axis)*angle)
                            p, r, _, _ = solver.kinematics(q)
                            target.update(control(rc), np.eye(3), p, r)
                            self.assertFalse(target.orientation_limited)
                            q = self.step(solver, q, target)
                            if i == 250:
                                held = q.copy()
                        p, r, _, _ = solver.kinematics(q)
                        np.testing.assert_allclose(target.rotation, requested, atol=1e-10)
                        self.assertLess(np.linalg.norm(rotation_vector(requested @ r.T)), np.radians(.1))
                        self.assertLess(np.linalg.norm(p-p0), .001)
                        np.testing.assert_allclose(q, held, atol=1e-6)

    def test_reachable_mixed_poses_converge_from_home(self):
        rng = np.random.default_rng(42)
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            for _ in range(10):
                solver = PioneerArticleIK(names, wrist_velocity=3.6)
                q = solver.rest.copy()
                goal = np.clip(q+rng.uniform(-.2, .2, 6), solver.limits[:, 0], solver.limits[:, 1])
                p, r, _, _ = solver.kinematics(goal)
                target = Target(active=True, position=p, rotation=r)
                for _ in range(250):
                    q = self.step(solver, q, target)
                achieved_p, achieved_r, _, _ = solver.kinematics(q)
                self.assertLess(np.linalg.norm(p-achieved_p), .001)
                self.assertLess(np.linalg.norm(rotation_vector(r @ achieved_r.T)), np.radians(.1))

    def test_slipping_clutch_absorbs_angular_excess_and_reverses_immediately(self):
        target = ArticleTarget(calibration=np.eye(3))
        engage(target, np.zeros(3), np.eye(3))
        target.update(control(rotation_exp(np.array([0., 1., 0.]))), np.eye(3), np.zeros(3), np.eye(3))
        self.assertTrue(target.orientation_limited)
        np.testing.assert_allclose(rotation_vector(target.rotation), [0., .6, 0.], atol=1e-12)
        # No hidden backlog: a hold does not keep pursuing the original 1 rad.
        for _ in range(50):
            target.update(control(rotation_exp(np.array([0., 1., 0.]))), np.eye(3), np.zeros(3), target.rotation.copy())
        np.testing.assert_allclose(rotation_vector(target.rotation), [0., .6, 0.], atol=1e-12)
        target.update(control(rotation_exp(np.array([0., .9, 0.]))), np.eye(3), np.zeros(3), target.rotation.copy())
        np.testing.assert_allclose(rotation_vector(target.rotation), [0., .5, 0.], atol=1e-12)

    def test_multi_axis_conjugation_and_reengagement(self):
        mapping = rotation_exp(np.array([.3, .2, -.2]))
        r0 = rotation_exp(np.array([-.1, .3, .2]))
        target = ArticleTarget(calibration=mapping)
        engage(target, np.zeros(3), r0)
        for vec in ([.1, .1, 0.], [-.1, .15, .1], [0., 0., 0.]):
            rc = rotation_exp(np.array(vec))
            target.update(control(rc), np.eye(3), np.zeros(3), r0)
            np.testing.assert_allclose(target.rotation, mapping @ rc @ mapping.T @ r0, atol=1e-12)
        target.lose_tracking()
        engage(target, np.zeros(3), np.eye(3))
        np.testing.assert_allclose(target.rotation, np.eye(3), atol=1e-12)

    def test_pivot_calibration_still_removes_translation_from_rotation(self):
        pivot = np.array([.1, -.03, .04])
        wrist = np.array([0., 1., 0.])
        target = ArticleTarget(calibration=np.eye(3), pivot=pivot)
        for grip in (False, True):
            target.update(control(np.eye(3), grip, position=wrist-pivot), np.eye(3), wrist, np.eye(3))
        r = rotation_exp(np.array([.2, -.1, .1]))
        target.update(control(r, position=wrist-r @ pivot), np.eye(3), wrist, np.eye(3))
        np.testing.assert_allclose(target.position, wrist, atol=1e-12)
        np.testing.assert_allclose(target.rotation, r, atol=1e-12)

    def test_singularity_and_antipode_are_bounded(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            for sign in (-1., 1.):
                solver = PioneerArticleIK(names, velocity=100., wrist_velocity=100.)
                q = solver.rest.copy()
                q[3] = sign*np.radians(1.)  # Near straight elbow: roll axes align.
                solver.command = q.copy()
                p, r, _, _ = solver.kinematics(q)
                target = Target(active=True, position=p, rotation=rotation_exp(np.array([.01, .01, .01])) @ r)
                result = solver.solve(q, target, 1.)
                self.assertLess(np.max(np.abs(result-q)), np.radians(5.))
            solver = PioneerArticleIK(names, wrist_velocity=3.6)
            q = solver.rest.copy()
            p, r, _, _ = solver.kinematics(q)
            target = Target(active=True, position=p, rotation=rotation_exp(np.array([0., 2.7, 0.])) @ r)
            np.testing.assert_allclose(solver.solve(q, target, .02), q, atol=1e-12)

    def test_joint_stop_does_not_freeze_the_other_orientation_joints(self):
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            solver = PioneerArticleIK(names, wrist_velocity=3.6)
            q = solver.rest.copy()
            q[4] = solver.limits[4, 1]
            solver.command = q.copy()
            p, r, _, _ = solver.kinematics(q)
            goal = q.copy()
            goal[4:] += [.15, .25]
            _, goal_r, _, _ = solver.kinematics(goal)
            target = Target(active=True, position=p, rotation=goal_r)
            initial_error = np.linalg.norm(rotation_vector(goal_r @ r.T))
            for _ in range(200):
                q = self.step(solver, q, target)
            _, r, _, _ = solver.kinematics(q)
            self.assertLess(np.linalg.norm(rotation_vector(goal_r @ r.T)), initial_error*.8)

    def test_usd_joint_frames_match_the_urdf_model(self):
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            self.skipTest("USD Python bindings unavailable")
        stage = Usd.Stage.Open(str(HERE.parents[2]/"assets/pioneer_bimanual_arm/usd/pioneer_bimanual_arm.usd"))
        joints = {p.GetName(): UsdPhysics.RevoluteJoint(p) for p in stage.Traverse()
                  if p.IsA(UsdPhysics.RevoluteJoint)}
        def matrix(quat):
            return quaternion_matrix([*quat.GetImaginary(), quat.GetReal()])
        for names in (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS):
            solver = PioneerArticleIK(names)
            for i, name in enumerate(names):
                joint = joints[name]
                r0, r1 = matrix(joint.GetLocalRot0Attr().Get()), matrix(joint.GetLocalRot1Attr().Get())
                rotation = r0 @ r1.T
                translation = np.array(joint.GetLocalPos0Attr().Get())-rotation @ np.array(joint.GetLocalPos1Attr().Get())
                usd_axis = r0 @ np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
                np.testing.assert_allclose(rotation, solver.origins[i][:3, :3], atol=1e-6)
                np.testing.assert_allclose(translation, solver.origins[i][:3, 3], atol=1e-6)
                np.testing.assert_allclose(usd_axis, rotation @ solver.axes[i], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
