import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward import PoseReward, RewardConfig, U, keypoint_error_centered, limit_penalty
from pose_math import TOL_POS

C = RewardConfig
LO, HI = -np.ones(7), np.ones(7)     # joint limits used in the tests
EYE = np.eye(3)


def Rz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make(rho0=10.0):
    """A reward that has just been reset on a goal that starts rho0 tolerance units away."""
    r = PoseReward()
    r.reset(rho0)
    return r


def step_full(r, rho_val, a=None, q=None, qd=None, q_rest=None):
    """One step with the goal straight ahead, rho_val tolerance units (rho_val mm) away, same orientation.
    Returns (total, terms)."""
    p_goal = np.array([rho_val * TOL_POS, 0.0, 0.0])
    a = np.zeros(7) if a is None else a
    q = np.zeros(7) if q is None else q
    qd = np.zeros(7) if qd is None else qd
    return r.step(np.zeros(3), EYE, p_goal, EYE, q, qd, a, LO, HI, q_rest=q_rest)


def step_at(r, rho_val, **kw):
    return step_full(r, rho_val, **kw)[1]     # just the terms


class TestHelpers(unittest.TestCase):
    def test_U_is_a_smooth_step(self):
        self.assertAlmostEqual(U(1.0), 0.5)
        self.assertTrue(U(0.0) > 0.95)
        self.assertTrue(U(2.0) < 0.05)
        self.assertTrue(U(0.5) > U(1.0) > U(1.5))      # falls as the error grows

    def test_keypoint_error_zero_at_goal(self):
        self.assertAlmostEqual(keypoint_error_centered(np.zeros(3), Rz(30), np.zeros(3), Rz(30), 0.45), 0.0)

    def test_keypoint_error_pure_translation_is_the_distance(self):
        # every point moves with the gripper, so the average distance is the translation distance
        e = keypoint_error_centered(np.zeros(3), EYE, np.array([0.01, 0.0, 0.0]), EYE, 0.45)
        self.assertAlmostEqual(e, 0.01)

    def test_keypoint_error_sees_rotation(self):
        # goal turned 90 deg about z at the same tool point. The three points are at
        # kp*(2/3,-1/3,-1/3), kp*(-1/3,2/3,-1/3), kp*(-1/3,-1/3,2/3). Turning 90 deg about z moves a point at
        # distance r from the z axis by sqrt(2)*r, and r is kp*sqrt(5)/3, kp*sqrt(5)/3, kp*sqrt(2)/3.
        kp = 0.45
        expected = np.sqrt(2) * kp / 3 * (2 * np.sqrt(5) + np.sqrt(2)) / 3
        e = keypoint_error_centered(np.zeros(3), EYE, np.zeros(3), Rz(90), kp)
        self.assertAlmostEqual(e, expected)
        # a bigger turn gives a bigger error
        self.assertTrue(keypoint_error_centered(np.zeros(3), EYE, np.zeros(3), Rz(10), kp) < e)

    def test_limit_penalty(self):
        lo, hi = -np.ones(3), np.ones(3)       # range 2, so the 5% margin is 0.1 from a limit
        self.assertEqual(limit_penalty(np.zeros(3), lo, hi, 0.05), 0.0)                  # middle
        self.assertAlmostEqual(limit_penalty(np.array([1.0, 0, 0]), lo, hi, 0.05), 1.0)  # at the upper limit
        self.assertAlmostEqual(limit_penalty(np.array([0, -1.0, 0]), lo, hi, 0.05), 1.0)  # at the lower limit
        self.assertAlmostEqual(limit_penalty(np.array([0.9, 0, 0]), lo, hi, 0.05), 0.0)   # just outside the margin
        self.assertAlmostEqual(limit_penalty(np.array([0.95, 0, 0]), lo, hi, 0.05), 0.5)  # half way into it
        # only the worst joint counts
        self.assertAlmostEqual(limit_penalty(np.array([0.95, 0.99, 0]), lo, hi, 0.05), 0.9)


class TestReset(unittest.TestCase):
    def test_reset_sets_the_memory(self):
        r = make(10.0)
        self.assertEqual(r.best, 10.0)
        self.assertEqual(r.phi_prev, -10.0)
        self.assertTrue(np.array_equal(r.a_prev, np.zeros(7)))
        self.assertTrue(r.fresh)

    def test_reset_caps_the_starting_potential(self):
        self.assertEqual(make(500.0).phi_prev, -C.phi_cap)

    def test_reset_with_a_given_previous_action(self):
        r = PoseReward()
        r.reset(10.0, a_prev=np.ones(7))
        self.assertTrue(np.array_equal(r.a_prev, np.ones(7)))

    def test_a_new_goal_pays_nothing_on_its_first_step_again(self):
        r = make(10.0)
        step_at(r, 10.0)
        step_at(r, 5.0)
        r.reset(8.0)
        t = step_at(r, 0.5)
        for k in ("improvement", "shaping", "backslide", "dwell"):
            self.assertEqual(t[k], 0.0)


class TestProgressTerms(unittest.TestCase):
    def test_first_step_pays_nothing(self):
        t = step_at(make(10.0), 0.5)           # even inside the tolerance
        for k in ("improvement", "shaping", "backslide", "dwell"):
            self.assertEqual(t[k], 0.0)

    def test_improvement_telescopes(self):
        r = make(10.0)
        step_at(r, 10.0)                                       # first step: pays nothing
        # 6 is a new best, 8 is a step back (pays nothing), 5 is a new best
        total = sum(step_at(r, x)["improvement"] for x in (6.0, 8.0, 5.0))
        self.assertAlmostEqual(total, C.w_improve * (U(5.0) - U(10.0)))

    def test_improvement_pays_nothing_without_a_new_best(self):
        r = make(10.0)
        step_at(r, 10.0)
        self.assertEqual(step_at(r, 10.0)["improvement"], 0.0)   # equal to the best
        self.assertEqual(step_at(r, 14.0)["improvement"], 0.0)   # worse than the best

    def test_shaping_pays_for_getting_closer_and_charges_for_going_back(self):
        r = make(10.0)
        step_at(r, 10.0)                                       # memory: phi_prev = -10
        self.assertAlmostEqual(step_at(r, 8.0)["shaping"], C.gamma * -8.0 - -10.0)    # +2.08
        self.assertAlmostEqual(step_at(r, 10.0)["shaping"], C.gamma * -10.0 - -8.0)   # -1.90
        r2 = make(8.0)
        step_at(r2, 8.0)
        self.assertAlmostEqual(step_at(r2, 8.0)["shaping"], C.gamma * -8.0 - -8.0)    # +0.08 for standing still

    def test_shaping_potential_is_capped(self):
        r = make(500.0)
        step_at(r, 500.0)
        # both ends are capped at -50, so only the discount is left: 0.99*(-50) - (-50)
        self.assertAlmostEqual(step_at(r, 400.0)["shaping"], C.gamma * -C.phi_cap + C.phi_cap)

    def test_backslide_only_when_worse_than_the_best(self):
        r = make(10.0)
        step_at(r, 10.0)
        self.assertAlmostEqual(step_at(r, 12.0)["backslide"], -C.w_back * 2.0)
        self.assertEqual(step_at(r, 9.0)["backslide"], 0.0)

    def test_backslide_is_capped(self):
        r = make(10.0)
        step_at(r, 10.0)
        self.assertAlmostEqual(step_at(r, 100.0)["backslide"], -C.w_back * C.back_cap)

    def test_backslide_is_measured_from_the_best_not_the_last_step(self):
        r = make(10.0)
        step_at(r, 10.0)
        step_at(r, 12.0)                                       # step back, the best is still 10
        self.assertAlmostEqual(step_at(r, 13.0)["backslide"], -C.w_back * 3.0)

    def test_dwell_only_inside_the_tolerance(self):
        r = make(10.0)
        step_at(r, 10.0)
        self.assertEqual(step_at(r, 0.9)["dwell"], C.w_dwell)
        self.assertEqual(step_at(r, 1.1)["dwell"], 0.0)


class TestCostTerms(unittest.TestCase):
    def test_action_rate_is_measured_against_the_previous_action(self):
        r = make(10.0)
        a = np.array([0.1, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(step_at(r, 10.0, a=a)["action rate"], -C.w_action_rate_near * 0.01)  # vs the reset zeros (rho 10 counts as near)
        self.assertEqual(step_at(r, 10.0, a=a)["action rate"], 0.0)                            # same action again

    def test_action_rate_sums_over_joints(self):
        r = make(10.0)
        self.assertAlmostEqual(step_at(r, 10.0, a=np.array([3.0, 4.0, 0, 0, 0, 0, 0]))["action rate"],
                               -C.w_action_rate_near * 25.0)

    def test_previous_action_is_a_copy(self):
        r = make(10.0)
        a = np.zeros(7)
        step_at(r, 10.0, a=a)
        a[0] = 5.0                                             # the caller changes its array afterwards
        self.assertAlmostEqual(step_at(r, 10.0, a=a)["action rate"], -C.w_action_rate_near * 25.0)

    def test_action_rate_is_lighter_far_from_the_goal(self):
        a = np.array([3.0, 4.0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(step_at(make(20.0), 20.0, a=a)["action rate"], -C.w_action_rate * 25.0)

    def test_action_size_is_only_penalized_near_the_goal(self):
        a = np.array([3.0, 4.0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(step_at(make(5.0), 5.0, a=a)["action size"], -C.w_action_near * 25.0)
        self.assertEqual(step_at(make(20.0), 20.0, a=a)["action size"], 0.0)
        self.assertEqual(step_at(make(5.0), 5.0)["action size"], 0.0)

    def test_time_cost_stops_once_within_tolerance(self):
        self.assertAlmostEqual(step_at(make(20.0), 20.0)["time"], -C.w_time)
        self.assertAlmostEqual(step_at(make(2.0), 2.0)["time"], -C.w_time)
        self.assertEqual(step_at(make(0.5), 0.5)["time"], 0.0)
        self.assertEqual(step_at(make(1.0), 1.0)["time"], 0.0)

    def test_joint_velocity(self):
        qd = np.array([1.0, 2.0, 0, 0, 0, 0, 0])
        # far from the goal: the light cost. Within near_rho of it: the heavy one.
        self.assertAlmostEqual(step_at(make(20.0), 20.0, qd=qd)["joint velocity"], -C.w_joint_vel * 5.0)
        self.assertAlmostEqual(step_at(make(5.0), 5.0, qd=qd)["joint velocity"], -C.w_joint_vel_near * 5.0)
        self.assertEqual(step_at(make(10.0), 10.0)["joint velocity"], 0.0)

    def test_posture_cost_is_a_fraction_of_each_joints_range(self):
        q = np.zeros(7); q[0] = 1.0                      # joint 0 is 1 rad from rest; the range is 2, so 0.5 of it
        rest = np.zeros(7)
        self.assertAlmostEqual(step_at(make(20.0), 20.0, q=q, q_rest=rest)["posture"], -C.w_posture * 0.25)
        q[1] = -1.0                                      # two joints: the squares add
        self.assertAlmostEqual(step_at(make(20.0), 20.0, q=q, q_rest=rest)["posture"], -C.w_posture * 0.5)

    def test_no_posture_cost_at_rest_or_without_a_rest_pose(self):
        q = np.full(7, 0.3)
        self.assertEqual(step_at(make(20.0), 20.0, q=q, q_rest=q)["posture"], 0.0)
        self.assertEqual(step_at(make(20.0), 20.0, q=q)["posture"], 0.0)

    def test_joint_limit_penalty_only_applies_far_from_the_goal(self):
        q = np.zeros(7)
        q[0] = 1.0                                             # joint 0 sits at its upper limit
        far = step_at(make(20.0), 20.0, q=q)                   # rho 20 > the gate (10)
        near = step_at(make(5.0), 5.0, q=q)                    # rho 5, penalty switched off
        self.assertAlmostEqual(far["joint limits"], -C.w_limit)
        self.assertEqual(near["joint limits"], 0.0)
        self.assertEqual(step_at(make(20.0), 20.0)["joint limits"], 0.0)   # far, but away from the limits


class TestPositionTerms(unittest.TestCase):
    def test_position_terms_at_the_goal(self):
        t = step_at(make(10.0), 0.0)
        self.assertAlmostEqual(t["linear position"], 0.0)
        self.assertAlmostEqual(t["fine position"], C.w_pos_tanh)
        self.assertAlmostEqual(t["keypoint wide"], C.w_kp_wide)
        self.assertAlmostEqual(t["keypoint sharp"], C.w_kp_sharp)

    def test_position_terms_10_cm_away(self):
        d = 0.10                                               # 100 rho units straight ahead
        t = step_at(make(100.0), 100.0)
        self.assertAlmostEqual(t["linear position"], -d * C.w_pos_linear)
        self.assertAlmostEqual(t["fine position"], (1 - np.tanh(d / C.std_pos)) * C.w_pos_tanh)
        # a pure translation moves all three keypoints by d
        self.assertAlmostEqual(t["keypoint wide"], (1 - np.tanh(d / C.std_kp_wide)) * C.w_kp_wide)
        self.assertAlmostEqual(t["keypoint sharp"], (1 - np.tanh(d / C.std_kp_sharp)) * C.w_kp_sharp)

    def test_sharp_term_falls_faster_than_the_wide_one(self):
        t = step_at(make(100.0), 100.0)
        self.assertTrue(t["keypoint sharp"] < t["keypoint wide"])


class TestStepOverall(unittest.TestCase):
    def test_total_is_the_sum_of_the_terms(self):
        r = make(10.0)
        step_full(r, 10.0)
        total, terms = step_full(r, 7.0, a=np.ones(7) * 0.1, qd=np.ones(7))
        self.assertAlmostEqual(total, sum(terms.values()))
        self.assertEqual(len(terms), 14)

    def test_holding_at_the_goal_pays_a_fixed_amount_per_step(self):
        # linear 0 + fine 0.1 + wide 1 + sharp 1 + dwell 1; progress terms are 0 when nothing changes
        r = make(0.0)
        step_full(r, 0.0)                                      # first step, pays nothing extra
        total, _ = step_full(r, 0.0)
        self.assertAlmostEqual(total, C.w_pos_tanh + C.w_kp_wide + C.w_kp_sharp + C.w_dwell)

    def test_memory_updates_after_each_step(self):
        r = make(10.0)
        step_at(r, 7.0, a=np.ones(7))
        self.assertAlmostEqual(r.best, 7.0)
        self.assertAlmostEqual(r.phi_prev, -7.0)
        self.assertTrue(np.array_equal(r.a_prev, np.ones(7)))
        self.assertFalse(r.fresh)
        step_at(r, 9.0)
        self.assertAlmostEqual(r.best, 7.0)                    # the best never goes up
        self.assertAlmostEqual(r.phi_prev, -9.0)               # the potential follows the latest step


if __name__ == "__main__":
    unittest.main()
