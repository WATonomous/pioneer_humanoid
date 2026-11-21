"""Controller-to-target mapping (article_control.AnchoredTarget): lead limit without drift, conjugation, pivot calibration."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2]/"src/pioneer_humanoid")]
from article_control import AnchoredTarget
from control import Input, rotation_exp, rotation_vector


def control(rotation, precision=False, position=None):
    return Input(np.zeros(3) if position is None else position, rotation, 0., precision, False)


def engage(target, position, rotation):
    target.update(control(np.eye(3)), np.eye(3), position, rotation)


class ArticleControlTests(unittest.TestCase):

    def test_lead_limit_caps_the_goal_not_the_request(self):
        target = AnchoredTarget(calibration=np.eye(3))
        engage(target, np.zeros(3), np.eye(3))
        self.assertTrue(target.active)
        target.update(control(rotation_exp(np.array([0., 1., 0.]))), np.eye(3), np.zeros(3), np.eye(3))
        np.testing.assert_allclose(rotation_vector(target.rotation), [0., .6, 0.], atol=1e-12)
        # No drift: once the arm reaches the capped goal, the goal moves on to the full 1 rad the hand asks for.
        target.update(control(rotation_exp(np.array([0., 1., 0.]))), np.eye(3), np.zeros(3), target.rotation.copy())
        np.testing.assert_allclose(rotation_vector(target.rotation), [0., 1., 0.], atol=1e-12)
        target.update(control(rotation_exp(np.array([0., .9, 0.]))), np.eye(3), np.zeros(3), target.rotation.copy())
        np.testing.assert_allclose(rotation_vector(target.rotation), [0., .9, 0.], atol=1e-12)

    def test_multi_axis_conjugation_precision_and_reengagement(self):
        mapping = rotation_exp(np.array([.3, .2, -.2]))
        r0 = rotation_exp(np.array([-.1, .3, .2]))
        target = AnchoredTarget(calibration=mapping)
        engage(target, np.zeros(3), r0)
        for vec in ([.1, .1, 0.], [-.1, .15, .1], [0., 0., 0.]):
            rc = rotation_exp(np.array(vec))
            target.update(control(rc), np.eye(3), np.zeros(3), r0)
            np.testing.assert_allclose(target.rotation, mapping @ rc @ mapping.T @ r0, atol=1e-12)
        target.update(control(rotation_exp(np.array([.2, 0., 0.])), precision=True), np.eye(3), np.zeros(3), r0)
        np.testing.assert_allclose(target.rotation, mapping @ rotation_exp(np.array([.05, 0., 0.])) @ mapping.T @ r0, atol=1e-12)
        target.lose_tracking()
        self.assertFalse(target.active)
        engage(target, np.zeros(3), np.eye(3))
        np.testing.assert_allclose(target.rotation, np.eye(3), atol=1e-12)

    def test_pivot_calibration_still_removes_translation_from_rotation(self):
        pivot = np.array([.1, -.03, .04])
        wrist = np.array([0., 1., 0.])
        target = AnchoredTarget(calibration=np.eye(3), pivot=pivot)
        target.update(control(np.eye(3), position=wrist-pivot), np.eye(3), wrist, np.eye(3))
        r = rotation_exp(np.array([.2, -.1, .1]))
        target.update(control(r, position=wrist-r @ pivot), np.eye(3), wrist, np.eye(3))
        np.testing.assert_allclose(target.position, wrist, atol=1e-12)
        np.testing.assert_allclose(target.rotation, r, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
