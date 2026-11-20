"""Tests for sampling.py (task B1)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_model import LeftArm                                                 
from sampling import WORKSPACE_HI, WORKSPACE_LO, in_workspace, load_eval_set, make_eval_set, too_close

N = 20   # small sets keep the tests fast

def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
class SamplingTests(unittest.TestCase):

    def setUp(self):
        # Runs before EVERY test: a fresh arm and a temporary folder that's deleted afterwards.
        self.arm = LeftArm()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "eval.npz")

    def tearDown(self):
        self.tmp.cleanup()

    # --- Test 1 (example) ---
    def test_in_workspace(self):
        centre = (WORKSPACE_LO + WORKSPACE_HI) / 2
        self.assertTrue(in_workspace(centre))
        for axis in range(3):                        # push one axis just outside, each side
            too_low = centre.copy()
            too_low[axis] = WORKSPACE_LO[axis] - 0.01
            self.assertFalse(in_workspace(too_low))
            too_high = centre.copy()
            too_high[axis] = WORKSPACE_HI[axis] + 0.01
            self.assertFalse(in_workspace(too_high))

    # --- Test 2 ---
    def test_goal_matches_fk(self):
        make_eval_set(N, 0, self.path)
        s = load_eval_set(self.path)
        for i in range(N):
            p, R = self.arm.tcp_pose(s["q_goal"][i])
            np.testing.assert_allclose(p, s["p_goal"][i], atol=1e-9)
            np.testing.assert_allclose(R, s["R_goal"][i], atol=1e-9)

    # --- Test 3 ---
    def test_goals_in_workspace_and_angles_in_limits(self):
        make_eval_set(N, 0, self.path)
        s = load_eval_set(self.path)
        lo, hi = self.arm.limits[:, 0], self.arm.limits[:, 1]
        for i in range(N):
            self.assertTrue(in_workspace(s["p_goal"][i]), msg=f"goal {i} outside the workspace: {s['p_goal'][i]}")
            self.assertTrue(np.all(s["q_start"][i] >= lo) and np.all(s["q_start"][i] <= hi))
            self.assertTrue(np.all(s["q_goal"][i] >= lo) and np.all(s["q_goal"][i] <= hi))

    # --- Test 4 ---
    def test_same_seed_same_set(self):
        make_eval_set(N, 0, self.path)
        s = load_eval_set(self.path)
        path2 = str(Path(self.tmp.name) / "eval2.npz")
        make_eval_set(N, 0, path2)
        s2 = load_eval_set(path2)
        for key in s2:
            np.testing.assert_array_equal(s[key], s2[key])
        path3 = str(Path(self.tmp.name) / "eval3.npz")

        make_eval_set(N, 1, path3)
        s3 = load_eval_set(path3)
        self.assertFalse(np.array_equal(s["q_goal"], s3["q_goal"]))


    # --- Test 5 ---
    def test_save_load_round_trip(self):
        make_eval_set(N, 0, self.path)
        s = load_eval_set(self.path)
        self.assertEqual(int(s["seed"]), 0)
        self.assertIn("q_start", s)
        self.assertIn("q_goal", s)
        self.assertIn("p_goal", s)
        self.assertIn("R_goal", s)
        self.assertEqual(s["q_start"].shape, (N, 7))
        self.assertEqual(s["q_goal"].shape, (N, 7))
        self.assertEqual(s["p_goal"].shape, (N, 3))
        self.assertEqual(s["R_goal"].shape, (N, 3, 3))

    # --- Test 6 ---
    def test_too_close(self):
        p = np.array([0.3, 0.2, -0.2])
        R = rot_z(0.5)
        self.assertTrue(too_close(p, R, np.array([p]), np.array([R])))
        
        p2 = p + np.array([0.002, 0, 0])
        self.assertFalse(too_close(p2, R, np.array([p]), np.array([R])))
        R2 = R @ rot_z(np.radians(1.0))
        self.assertFalse(too_close(p, R2, np.array([p]), np.array([R])))
        p3 = p + np.array([0.0005, 0, 0])
        R3 = R @ rot_z(np.radians(0.1))
        self.assertTrue(too_close(p3, R3, np.array([p]), np.array([R])))



if __name__ == "__main__":
    unittest.main()
