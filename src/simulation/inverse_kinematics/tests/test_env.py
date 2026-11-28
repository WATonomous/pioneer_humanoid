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
from sampling import collision_free

# Where each block sits in the 54-number observation (the order of Eq. 3 of the paper)
JOINTS, VEL, GOAL = slice(0, 7), slice(7, 14), slice(14, 17)
KEYPOINTS, COARSE, FINE = slice(17, 26), slice(26, 32), slice(32, 38)
BEST, COMMAND, PREV_ACTION = slice(38, 40), slice(40, 47), slice(47, 54)

TERM_NAMES = {"linear position", "fine position", "keypoint wide", "keypoint sharp", "improvement", "shaping",
              "backslide", "dwell", "action rate", "joint velocity", "joint limits", "posture", "action size", "time"}


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
        # thresholds: position 0.5 mm .. 30 mm, rotation 0.415 deg .. 7.5 deg, beta_min 0.1 (worked out by hand below)
        self.assertAlmostEqual(beta(1, 0.1), 0.1 + 0.9 * (0.5 / 29.5), places=4)          # 0.1153: the position error sets it
        self.assertAlmostEqual(beta(5, 0.2), 0.1 + 0.9 * (4.5 / 29.5), places=4)          # 0.2373
        self.assertAlmostEqual(beta(0.3, 2.328), 0.1 + 0.9 * (1.913 / 7.085), places=3)   # 0.343: the rotation error sets it
        self.assertAlmostEqual(beta(0.3, 7.5), 1.0)                 # rotation at its upper threshold alone: a full step
        self.assertAlmostEqual(beta(30, 0.0), 1.0)                  # position at its upper threshold alone: a full step
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


class TestGoalResampling(unittest.TestCase):
    """cfg.resample_goals: a new goal every goal_every steps, with the arm left where it is."""
    ACTION = np.linspace(-0.5, 0.5, 7)

    def make(self, **kw):
        env = PolicyIKEnv(EnvConfig(max_steps=20, goal_every=5, **kw), seed=4)
        env.reset()
        return env

    def test_off_by_default_the_goal_never_changes(self):
        env = self.make(resample_goals=False)
        g0 = env.p_goal.copy()
        for _ in range(19):
            env.step(self.ACTION)
            self.assertTrue(np.array_equal(env.p_goal, g0))

    def test_when_on_the_goal_changes_exactly_every_goal_every_steps(self):
        env = self.make(resample_goals=True)
        changed = []
        for step in range(1, 21):
            before = env.p_goal.copy()
            env.step(self.ACTION)
            if not np.array_equal(env.p_goal, before):
                changed.append(step)
        self.assertEqual(changed, [5, 10, 15])            # not on the last step (20), where the episode ends

    def test_resampling_does_not_move_the_arm(self):
        on, off = self.make(resample_goals=True), self.make(resample_goals=False)
        for _ in range(5):                                # the goal changes at the end of step 5 in `on` only
            on.step(self.ACTION)
            off.step(self.ACTION)
        self.assertFalse(np.allclose(on.p_goal, off.p_goal))
        self.assertTrue(np.array_equal(on.data.qpos, off.data.qpos))
        self.assertTrue(np.array_equal(on.data.qvel, off.data.qvel))
        self.assertTrue(np.array_equal(on.z, off.z))

    def test_the_observation_after_a_resample_is_for_the_new_goal(self):
        env = self.make(resample_goals=True)
        for _ in range(4):
            env.step(self.ACTION)
        obs = env.step(self.ACTION)[0]                    # step 5
        self.assertTrue(np.allclose(obs[GOAL], env.p_goal, atol=1e-6))
        self.assertTrue(np.allclose(obs[PREV_ACTION], self.ACTION))      # the last action is kept, not zeroed

    def test_per_goal_memory_restarts_on_the_new_goal(self):
        env = self.make(resample_goals=True)
        for _ in range(5):
            env.step(self.ACTION)
        rho_new = rho_now(env)
        self.assertTrue(env.reward.fresh)
        self.assertAlmostEqual(env.reward.best, rho_new, places=6)
        self.assertAlmostEqual(env.best_so_far, rho_new, places=6)        # set by the observation built after step 5
        terms6 = env.step(self.ACTION)[4]                                  # the first step against the new goal
        for name in ("improvement", "shaping", "backslide", "dwell"):
            self.assertEqual(terms6[name], 0.0)
        self.assertNotEqual(env.step(self.ACTION)[4]["shaping"], 0.0)      # and progress counts again from step 7

    def test_the_action_rate_cost_continues_across_a_new_goal(self):
        env = self.make(resample_goals=True)
        for _ in range(5):
            env.step(self.ACTION)
        self.assertEqual(env.step(self.ACTION)[4]["action rate"], 0.0)     # same action as before: no jump in the cost


class TestObservationNoise(unittest.TestCase):
    """cfg.obs_noise: uniform noise on the joint-position and joint-velocity blocks only."""
    NOISE = 0.01
    ACTIONS = [np.linspace(-1, 1, 7) * s for s in (0.3, 0.6, 0.9)]

    def run_env(self, noise, seed=6):
        env = PolicyIKEnv(EnvConfig(obs_noise=noise), seed=seed)
        out = [(env.reset(), 0.0)]
        for a in self.ACTIONS:
            obs, reward, *_ = env.step(a)
            out.append((obs, reward))
        return env, out

    def test_zero_noise_gives_the_true_joint_values(self):
        env, out = self.run_env(0.0)
        obs = out[-1][0]
        q, qd = env.data.qpos[env.arm.qadr], env.data.qvel[env.arm.qadr]
        self.assertTrue(np.allclose(obs[JOINTS], q - env.arm.rest, atol=1e-6))
        self.assertTrue(np.allclose(obs[VEL], qd, atol=1e-6))

    def test_noise_changes_only_the_joint_position_and_velocity_blocks(self):
        _, clean = self.run_env(0.0)
        _, noisy = self.run_env(self.NOISE)
        for (o_clean, _), (o_noisy, _) in zip(clean, noisy):
            keep = np.ones(54, dtype=bool)
            keep[JOINTS] = keep[VEL] = False
            self.assertTrue(np.array_equal(o_clean[keep], o_noisy[keep]))     # goal, errors, best, z - q, previous action
            self.assertFalse(np.array_equal(o_clean[JOINTS], o_noisy[JOINTS]))
            self.assertFalse(np.array_equal(o_clean[VEL], o_noisy[VEL]))

    def test_the_noise_stays_inside_plus_minus_the_setting(self):
        _, clean = self.run_env(0.0)
        _, noisy = self.run_env(self.NOISE)
        for (o_clean, _), (o_noisy, _) in zip(clean, noisy):
            for block in (JOINTS, VEL):
                self.assertTrue(np.all(np.abs(o_noisy[block] - o_clean[block]) <= self.NOISE + 1e-6))

    def test_noise_does_not_touch_the_physics_or_the_reward(self):
        env_clean, clean = self.run_env(0.0)
        env_noisy, noisy = self.run_env(self.NOISE)
        self.assertTrue(np.array_equal(env_clean.data.qpos, env_noisy.data.qpos))
        self.assertEqual([r for _, r in clean], [r for _, r in noisy])

    def test_noise_does_not_use_the_goal_generator(self):
        env_clean, _ = self.run_env(0.0)
        env_noisy, _ = self.run_env(self.NOISE)
        # The generator that picks goals and starts must be in exactly the same state with or without noise. (Comparing
        # goals alone is not enough: goal sampling rejects candidates in groups of 7 draws, and the noise uses 14 per
        # observation, so a shared generator can still land on the same goal by coincidence.)
        self.assertEqual(env_clean.rng.bit_generator.state, env_noisy.rng.bit_generator.state)
        self.assertTrue(np.array_equal(env_clean.p_goal, env_noisy.p_goal))
        env_clean.reset(); env_noisy.reset()
        self.assertTrue(np.array_equal(env_clean.p_goal, env_noisy.p_goal))
        self.assertEqual(env_clean.rng.bit_generator.state, env_noisy.rng.bit_generator.state)

    def test_noise_is_reproducible_and_fresh_each_time(self):
        _, a = self.run_env(self.NOISE, seed=3)
        _, b = self.run_env(self.NOISE, seed=3)
        self.assertTrue(all(np.array_equal(x[0], y[0]) for x, y in zip(a, b)))       # same seed, same noise
        env = PolicyIKEnv(EnvConfig(obs_noise=self.NOISE), seed=3)
        env.reset()
        self.assertFalse(np.array_equal(env._observation()[JOINTS], env._observation()[JOINTS]))   # new draw each call


class TestNearGoalStarts(unittest.TestCase):
    """cfg.near_goal_prob: some episodes start close to their goal (the paper's reverse curriculum)."""

    def distances(self, prob, n=20, seed=5):
        env = PolicyIKEnv(EnvConfig(near_goal_prob=prob), seed=seed)
        out = []
        for _ in range(n):
            env.reset()
            p, _ = env._tool_pose()
            out.append(1000 * np.linalg.norm(p - env.p_goal))      # mm
        return env, np.array(out)

    def test_off_by_default_starts_are_far(self):
        _, d = self.distances(0.0)
        self.assertGreater(d.min(), 100.0)

    def test_when_always_on_every_start_is_close(self):
        env, d = self.distances(1.0)
        self.assertLess(d.max(), 100.0)
        self.assertTrue(np.all(np.abs(env.z - env.q_goal) <= env.cfg.near_goal_noise + 1e-12))   # last episode's joints

    def test_a_mix_gives_roughly_the_chosen_share(self):
        _, d = self.distances(0.5, n=60)
        self.assertTrue(0.25 < np.mean(d < 100.0) < 0.75)

    def test_near_starts_are_collision_free_and_inside_the_limits(self):
        env = PolicyIKEnv(EnvConfig(near_goal_prob=1.0), seed=2)
        for _ in range(25):
            env.reset()
            self.assertTrue(collision_free(env.goal_arm, env.z))
            self.assertTrue(np.all(env.z >= env.lo) and np.all(env.z <= env.hi))

    def test_the_goal_itself_does_not_depend_on_the_option(self):
        on, off = PolicyIKEnv(EnvConfig(near_goal_prob=1.0), seed=8), PolicyIKEnv(EnvConfig(near_goal_prob=0.0), seed=8)
        on.reset(); off.reset()
        self.assertTrue(np.array_equal(on.p_goal, off.p_goal))

    def test_the_arm_starts_at_rest_and_the_reward_is_fresh(self):
        env = PolicyIKEnv(EnvConfig(near_goal_prob=1.0), seed=3)
        obs = env.reset()
        self.assertTrue(np.allclose(obs[VEL], 0.0))
        self.assertTrue(env.reward.fresh)
        self.assertLess(env.reward.best, 200.0)                  # a small starting error, in tolerance units


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


class TestHardCases(unittest.TestCase):
    """cfg.edge_goal_prob / cfg.hard_start_prob: goals near a joint limit, and starts with joints pinned at limits."""

    def frac(self, env, q):
        return np.min(np.minimum(q - env.lo, env.hi - q) / (env.hi - env.lo))

    def test_edge_goals_have_a_joint_near_a_limit(self):
        env = PolicyIKEnv(EnvConfig(edge_goal_prob=1.0), seed=4)
        n = 0
        for _ in range(20):
            env.reset()
            n += self.frac(env, env.q_goal) < 0.05
        self.assertGreaterEqual(n, 18)          # (a draw that never finds one keeps its last goal; that is rare)

    def test_edge_goal_matches_its_joint_angles(self):
        env = PolicyIKEnv(EnvConfig(edge_goal_prob=1.0), seed=6)
        env.reset()
        p, R = env.goal_arm.tcp_pose(env.q_goal)
        self.assertTrue(np.allclose(p, env.p_goal) and np.allclose(R, env.R_goal))

    def test_pinned_starts_have_a_joint_at_a_limit_and_are_valid(self):
        env = PolicyIKEnv(EnvConfig(hard_start_prob=1.0), seed=3)
        n = 0
        for _ in range(20):
            env.reset()
            self.assertTrue(collision_free(env.goal_arm, env.z))
            self.assertTrue(np.all(env.z >= env.lo) and np.all(env.z <= env.hi))
            n += self.frac(env, env.z) < 0.01
        self.assertGreaterEqual(n, 15)

    def test_hard_starts_do_not_replace_near_goal_starts(self):
        env = PolicyIKEnv(EnvConfig(near_goal_prob=1.0, hard_start_prob=1.0), seed=2)
        for _ in range(10):
            env.reset()
            self.assertTrue(np.all(np.abs(env.z - env.q_goal) <= env.cfg.near_goal_noise + 1e-12))

    def test_off_by_default_changes_nothing(self):
        a, b = PolicyIKEnv(EnvConfig(), seed=9), PolicyIKEnv(EnvConfig(edge_goal_prob=0.0, hard_start_prob=0.0), seed=9)
        self.assertTrue(np.array_equal(a.reset(), b.reset()))


class TestGoalBands(unittest.TestCase):
    """cfg.band_goals: goals only from the 20%+ band or the 5-20% band of distance to the joint limits."""

    def fracs(self, n, **cfg):
        env = PolicyIKEnv(EnvConfig(band_goals=True, **cfg), seed=4)
        out = []
        for _ in range(n):
            env.reset()
            out.append(np.min(np.minimum(env.q_goal - env.lo, env.hi - env.q_goal) / (env.hi - env.lo)))
        return np.array(out)

    def test_no_goal_is_within_5_percent_of_a_limit(self):
        self.assertTrue(np.all(self.fracs(30) >= 0.05))

    def test_the_two_bands_are_mixed_as_chosen(self):
        f = self.fracs(40)
        self.assertTrue(0.3 < np.mean(f >= 0.20) < 0.7)

    def test_all_centered_or_none(self):
        self.assertTrue(np.all(self.fracs(8, center_goal_prob=1.0) >= 0.20))
        f = self.fracs(20, center_goal_prob=0.0)
        self.assertTrue(np.all((f >= 0.05) & (f < 0.20)))

    def test_resampled_goals_use_the_bands_too(self):
        env = PolicyIKEnv(EnvConfig(band_goals=True, resample_goals=True, goal_every=3), seed=1)
        env.reset()
        for _ in range(12):
            env.step(np.zeros(7))
            f = np.min(np.minimum(env.q_goal - env.lo, env.hi - env.q_goal) / (env.hi - env.lo))
            self.assertGreaterEqual(f, 0.05)

    def test_off_by_default_changes_nothing(self):
        a, b = PolicyIKEnv(EnvConfig(), seed=9), PolicyIKEnv(EnvConfig(band_goals=False), seed=9)
        self.assertTrue(np.array_equal(a.reset(), b.reset()))


if __name__ == "__main__":
    unittest.main()
