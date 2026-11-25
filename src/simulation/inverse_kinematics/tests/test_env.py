"""Tests for env.py (task A4): reset, step, the observation layout, the gain beta, and the shared sim settings.

These need mujoco (like test_sampling.py), so run them with the same Python that has it:
    python -m unittest tests.test_env -v
"""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pose_math as pm
from arm_sim import EPISODE_STEPS, PHYSICS_DT, SUBSTEPS
from env import EnvConfig, PolicyIKEnv, gain

# Where each block sits in the 54-number observation (the order of Eq. 3 of the paper)
JOINTS, VEL, GOAL = slice(0, 7), slice(7, 14), slice(14, 17)
KEYPOINTS, COARSE, FINE = slice(17, 26), slice(26, 32), slice(32, 38)
BEST, COMMAND, PREV_ACTION = slice(38, 40), slice(40, 47), slice(47, 54)

TERM_NAMES = {"linear position", "fine position", "keypoint wide", "keypoint sharp", "improvement", "shaping",
              "backslide", "dwell", "action rate", "joint velocity", "joint limits"}


def rho_now(env):
    """The error in tolerance units from the arm's current pose, worked out independently of the env."""
    p, R = env._tool_pose()
    return pm.rho(*pm.pose_errors(p, R, env.p_goal, env.R_goal))


class TestSettings(unittest.TestCase):
    def test_uses_the_shared_sim_settings(self):
        env = PolicyIKEnv()
        self.assertAlmostEqual(env.model.opt.timestep, PHYSICS_DT)
        self.assertEqual(env.model.opt.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        self.assertEqual(SUBSTEPS, 20)                       # 20 substeps of 1/600 s = one 1/30 s control step
        self.assertEqual(env.cfg.max_steps, EPISODE_STEPS)


class TestGain(unittest.TestCase):
    L, U, B = 0.5e-3, 9.31e-3, 0.1

    def test_gain_limits_and_midpoint(self):
        self.assertAlmostEqual(gain(0.0, self.L, self.U, self.B), self.B)             # at or below l: beta_min
        self.assertAlmostEqual(gain(self.L, self.L, self.U, self.B), self.B)
        self.assertAlmostEqual(gain(self.U, self.L, self.U, self.B), 1.0)             # at or above u: full step
        self.assertAlmostEqual(gain(1.0, self.L, self.U, self.B), 1.0)
        self.assertAlmostEqual(gain((self.L + self.U) / 2, self.L, self.U, self.B), (self.B + 1.0) / 2)

    def test_beta_is_the_larger_of_the_two_gains(self):
        c = EnvConfig()
        def beta(mm, deg):
            return max(gain(mm / 1000, c.l_pos, c.u_pos, c.beta_min),
                       gain(np.radians(deg), c.l_rot, c.u_rot, c.beta_min))
        self.assertAlmostEqual(beta(1, 0.1), 0.151, places=3)       # the position error sets it
        self.assertAlmostEqual(beta(5, 0.2), 0.560, places=3)
        self.assertAlmostEqual(beta(0.3, 2.328), 1.0)               # the rotation error alone is enough for a full step
        self.assertAlmostEqual(beta(0.1, 0.05), c.beta_min)         # both small: the floor


class TestReset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PolicyIKEnv(seed=0)

    def test_observation_shape_type_and_values(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape, (54,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_goal_block_is_the_goal_position(self):
        obs = self.env.reset()
        self.assertTrue(np.allclose(obs[GOAL], self.env.p_goal, atol=1e-6))

    def test_arm_starts_at_rest_and_the_reference_equals_the_joints(self):
        obs = self.env.reset()
        self.assertTrue(np.allclose(obs[VEL], 0.0))                 # no motion
        self.assertTrue(np.allclose(obs[COMMAND], 0.0, atol=1e-6))  # z - q = 0
        self.assertTrue(np.all(self.env.z >= self.env.lo) and np.all(self.env.z <= self.env.hi))

    def test_no_previous_action_at_the_start(self):
        self.assertTrue(np.all(self.env.reset()[PREV_ACTION] == 0))

    def test_previous_action_does_not_leak_into_the_next_episode(self):
        self.env.reset()
        self.env.step(np.ones(7))
        self.assertTrue(np.all(self.env.reset()[PREV_ACTION] == 0))

    def test_best_so_far_features_on_the_first_step(self):
        obs = self.env.reset()
        rho0 = rho_now(self.env)
        self.assertAlmostEqual(float(obs[BEST][0]), min(rho0 / 20.0, 1.0), places=5)   # best / 20, clipped
        self.assertEqual(float(obs[BEST][1]), 0.0)                                    # no gap yet

    def test_pose_error_blocks_are_clipped_to_plus_minus_one(self):
        obs = self.env.reset()
        self.assertTrue(np.all(np.abs(obs[COARSE]) <= 1.0) and np.all(np.abs(obs[FINE]) <= 1.0))

    def test_reward_is_reset_for_the_new_goal(self):
        self.env.reset()
        self.assertTrue(self.env.reward.fresh)
        self.assertAlmostEqual(self.env.reward.best, rho_now(self.env), places=6)
        self.assertEqual(self.env.t, 0)

    def test_same_seed_gives_the_same_first_observation(self):
        a, b = PolicyIKEnv(seed=7).reset(), PolicyIKEnv(seed=7).reset()
        self.assertTrue(np.array_equal(a, b))

    def test_new_resets_give_new_goals(self):
        self.env.reset(); g1 = self.env.p_goal.copy()
        self.env.reset(); g2 = self.env.p_goal.copy()
        self.assertFalse(np.allclose(g1, g2))


class TestStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PolicyIKEnv(seed=1)

    def setUp(self):
        self.env.reset()

    def test_returns_five_values_and_a_valid_observation(self):
        obs, reward, terminated, truncated, terms = self.env.step(np.zeros(7))
        self.assertEqual(obs.shape, (54,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)) and np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(set(terms), TERM_NAMES)
        self.assertAlmostEqual(reward, sum(terms.values()))

    def test_zero_action_leaves_the_reference_unchanged(self):
        z0 = self.env.z.copy()
        self.env.step(np.zeros(7))
        self.assertTrue(np.array_equal(self.env.z, z0))

    def test_actions_are_clipped_and_stored_as_the_previous_action(self):
        obs = self.env.step(np.array([5.0, -5.0, 0.5, 0, 0, 0, -0.25]))[0]
        self.assertTrue(np.allclose(obs[PREV_ACTION], [1.0, -1.0, 0.5, 0, 0, 0, -0.25]))

    def test_reference_moves_by_at_most_eta_per_step(self):
        for _ in range(20):
            z0 = self.env.z.copy()
            self.env.step(np.ones(7) * 5.0)                       # a huge action, clipped to 1
            self.assertTrue(np.all(np.abs(self.env.z - z0) <= self.env.cfg.eta + 1e-12))

    def test_reference_stays_inside_the_joint_limits(self):
        for sign in (1.0, -1.0):
            self.env.reset()
            for _ in range(120):                                  # pushing hard in one direction until it saturates
                self.env.step(np.ones(7) * sign)
            self.assertTrue(np.all(self.env.z >= self.env.lo - 1e-12) and np.all(self.env.z <= self.env.hi + 1e-12))

    def test_a_pushed_reference_moves_the_arm(self):
        q0 = self.env.data.qpos[self.env.arm.qadr].copy()
        for _ in range(15):
            self.env.step(np.ones(7))
        self.assertGreater(np.abs(self.env.data.qpos[self.env.arm.qadr] - q0).max(), 0.05)

    def test_command_state_block_is_reference_minus_joints(self):
        obs = self.env.step(np.ones(7) * 0.5)[0]
        q = self.env.data.qpos[self.env.arm.qadr]
        self.assertTrue(np.allclose(obs[COMMAND], self.env.z - q, atol=1e-6))

    def test_first_step_pays_no_progress_reward_and_the_second_does(self):
        terms1 = self.env.step(np.zeros(7))[4]
        for name in ("improvement", "shaping", "backslide", "dwell"):
            self.assertEqual(terms1[name], 0.0)
        self.assertFalse(self.env.reward.fresh)
        self.assertNotEqual(self.env.step(np.zeros(7))[4]["shaping"], 0.0)   # the potential has changed, even a little

    def test_best_so_far_is_the_running_minimum_of_rho(self):
        rhos = [rho_now(self.env)]
        rng = np.random.default_rng(0)
        for _ in range(15):
            self.env.step(rng.uniform(-1, 1, 7))
            rhos.append(rho_now(self.env))
        self.assertAlmostEqual(self.env.best_so_far, min(rhos), places=6)

    def test_the_arm_barely_drifts_when_holding_still(self):
        p0 = self.env._tool_pose()[0]
        for _ in range(60):                                       # 2 s of zero actions: gravity sag, not a runaway
            self.env.step(np.zeros(7))
        drift_mm = 1000 * np.linalg.norm(self.env._tool_pose()[0] - p0)
        self.assertLess(drift_mm, 15.0)

    def test_step_counter(self):
        self.env.step(np.zeros(7)); self.env.step(np.zeros(7))
        self.assertEqual(self.env.t, 2)
        self.env.reset()
        self.assertEqual(self.env.t, 0)


class TestEpisodeEnd(unittest.TestCase):
    def test_truncates_exactly_at_max_steps_and_never_terminates(self):
        env = PolicyIKEnv(EnvConfig(max_steps=10), seed=2)
        env.reset()
        flags = [env.step(np.zeros(7))[2:4] for _ in range(10)]
        self.assertEqual([f[1] for f in flags], [False] * 9 + [True])      # truncated only on step 10
        self.assertFalse(any(f[0] for f in flags))                          # terminated never

    def test_a_full_random_episode_stays_finite(self):
        env = PolicyIKEnv(seed=3)
        env.reset()
        rng = np.random.default_rng(1)
        for i in range(EPISODE_STEPS):
            obs, reward, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 7))
            self.assertTrue(np.all(np.isfinite(obs)) and np.isfinite(reward))
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
