"""Tests for metrics.py (task B2): pose errors, per-episode scoring, and the summary over tasks.

The error traces are made by hand, so the right answer can be worked out on paper.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import CONTROL_DT, episode_metrics, pose_errors, rotation_angle, summarize  # noqa: E402

T = 160          # control steps per episode (5.3 s at 30 Hz)
MM = 1e-3        # 1 mm in metres
DEG = np.radians(1.0)


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class PoseErrorTests(unittest.TestCase):

    def test_same_pose_has_zero_error(self):
        p, R = np.array([0.3, 0.2, -0.2]), rot_z(0.5)
        e_p, e_R = pose_errors(p, R, p, R)
        self.assertAlmostEqual(e_p, 0.0)
        self.assertAlmostEqual(e_R, 0.0, places=6)   # arccos is a little imprecise right at 0

    def test_position_error_is_straight_line_distance(self):
        p, R = np.array([0.3, 0.2, -0.2]), np.eye(3)
        self.assertAlmostEqual(pose_errors(p + [3 * MM, 0, 0], R, p, R)[0], 3 * MM)
        # 3 mm along x and 4 mm along y is 5 mm away, not 7 mm
        self.assertAlmostEqual(pose_errors(p + [3 * MM, 4 * MM, 0], R, p, R)[0], 5 * MM)

    def test_rotation_error_is_the_angle(self):
        p, R = np.zeros(3), rot_z(0.5)
        self.assertAlmostEqual(pose_errors(p, R @ rot_z(DEG), p, R)[1], DEG, places=9)

    def test_rotation_angle_is_symmetric(self):
        R_a, R_b = rot_z(0.5), rot_z(0.8)
        self.assertAlmostEqual(rotation_angle(R_a, R_b), rotation_angle(R_b, R_a))


class EpisodeMetricsTests(unittest.TestCase):

    def test_zero_error_passes_everything(self):
        m = episode_metrics(np.zeros(T), np.zeros(T))
        self.assertTrue(m["strict"])
        self.assertTrue(m["coarse"])
        self.assertTrue(m["held"])
        self.assertAlmostEqual(m["entry_time"], CONTROL_DT)   # inside from the very first sample
        self.assertAlmostEqual(m["final_pos_err"], 0.0)

    def test_only_the_final_second_is_scored(self):
        e_p = np.zeros(T)
        e_p[:T - 30] = 50 * MM                    # far away for the first 130 steps, then perfect
        m = episode_metrics(e_p, np.zeros(T))
        self.assertTrue(m["strict"])
        self.assertTrue(m["held"])
        self.assertAlmostEqual(m["final_pos_err"], 0.0)

    def test_entry_time_is_measured_from_the_start_of_the_episode(self):
        e_p = np.full(T, 5 * MM)
        e_p[50:] = 0.3 * MM                       # first inside at sample 50, i.e. after step 51
        m = episode_metrics(e_p, np.zeros(T))
        self.assertAlmostEqual(m["entry_time"], 51 * CONTROL_DT)

    def test_settling_too_late_fails_strict(self):
        e_p = np.full(T, 2 * MM)
        e_p[-15:] = 0.5 * MM                      # inside only for the last half second
        m = episode_metrics(e_p, np.zeros(T))
        self.assertAlmostEqual(m["final_pos_err"], 1.25 * MM)   # (15 x 2 + 15 x 0.5) / 30
        self.assertFalse(m["strict"])
        self.assertFalse(m["held"])
        self.assertTrue(m["coarse"])
        self.assertAlmostEqual(m["entry_time"], (T - 15 + 1) * CONTROL_DT)

    def test_one_spike_breaks_held_but_not_strict(self):
        e_p = np.full(T, 0.5 * MM)
        e_p[-10] = 1.5 * MM                       # one bad sample inside the final second
        m = episode_metrics(e_p, np.zeros(T))
        self.assertAlmostEqual(m["final_pos_err"], (29 * 0.5 + 1.5) / 30 * MM)
        self.assertTrue(m["strict"])
        self.assertFalse(m["held"])

    def test_orientation_alone_can_fail(self):
        m = episode_metrics(np.zeros(T), np.full(T, 0.3 * DEG))   # perfect position, 0.3° off
        self.assertFalse(m["strict"])
        self.assertFalse(m["held"])
        self.assertTrue(m["coarse"])

    def test_never_entering_gives_nan_entry_time(self):
        m = episode_metrics(np.full(T, 5 * MM), np.zeros(T))
        self.assertTrue(np.isnan(m["entry_time"]))
        self.assertFalse(m["strict"])

    def test_coarse_but_not_strict(self):
        m = episode_metrics(np.full(T, 5 * MM), np.full(T, 1 * DEG))
        self.assertTrue(m["coarse"])
        self.assertFalse(m["strict"])

    def test_beyond_coarse_fails_both(self):
        m = episode_metrics(np.full(T, 20 * MM), np.zeros(T))
        self.assertFalse(m["coarse"])
        self.assertFalse(m["strict"])


class SummarizeTests(unittest.TestCase):

    def test_rates_and_medians_over_all_tasks(self):
        def result(strict, coarse, held, entry, pos, rot):
            return dict(strict=strict, coarse=coarse, held=held, entry_time=entry,
                        final_pos_err=pos, final_rot_err=rot)

        results = [
            result(True, True, True, 1.0, 0.2 * MM, 0.05 * DEG),       # success
            result(True, True, False, 2.0, 0.6 * MM, 0.10 * DEG),      # success, not held
            result(False, True, False, np.nan, 4.0 * MM, 0.50 * DEG),  # coarse only, never entered
            result(False, False, False, np.nan, 50 * MM, 5.0 * DEG),   # failure
        ]
        s = summarize(results)
        self.assertEqual(s["n_tasks"], 4)
        self.assertAlmostEqual(s["strict_rate"], 50.0)
        self.assertAlmostEqual(s["coarse_rate"], 75.0)
        self.assertAlmostEqual(s["held_rate"], 25.0)
        self.assertAlmostEqual(s["median_entry_time"], 1.5)          # only the tasks that entered
        # all 4 tasks count, failures too: the median of 4 values is the mean of the middle two
        self.assertAlmostEqual(s["median_final_pos_err"], (0.6 + 4.0) / 2 * MM)
        self.assertAlmostEqual(s["median_final_rot_err"], (0.10 + 0.50) / 2 * DEG)


if __name__ == "__main__":
    unittest.main()
