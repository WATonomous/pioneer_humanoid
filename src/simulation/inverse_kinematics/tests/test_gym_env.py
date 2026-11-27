"""Tests for gym_env.py: the Gymnasium wrapper around PolicyIKEnv. Needs gymnasium and mujoco."""
import sys
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pose_math as pm
from env import EnvConfig
from gym_env import ACTION_DIM, OBS_DIM, PolicyIKGymEnv
from metrics import pose_errors

ACTION = np.linspace(-1, 1, ACTION_DIM).astype(np.float32) * 0.6


def make(seed=0, **cfg):
    return PolicyIKGymEnv(EnvConfig(**cfg), seed=seed)


class TestSpaces(unittest.TestCase):
    def test_spaces(self):
        env = make()
        self.assertEqual(env.observation_space.shape, (54,))
        self.assertEqual(env.observation_space.dtype, np.float32)
        self.assertEqual(env.action_space.shape, (7,))
        self.assertEqual(env.action_space.dtype, np.float32)
        self.assertTrue(np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1))

    def test_passes_the_gymnasium_checker(self):
        check_env(make(max_steps=6), skip_render_check=True)

    def test_observations_are_in_the_space(self):
        env = make()
        obs, info = env.reset(seed=1)
        self.assertIsInstance(info, dict)
        self.assertTrue(env.observation_space.contains(obs))
        obs, *_ = env.step(env.action_space.sample())
        self.assertTrue(env.observation_space.contains(obs))


class TestStep(unittest.TestCase):
    def test_returns_gymnasium_style_values_and_info(self):
        env = make()
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(ACTION)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated or truncated)
        self.assertEqual(len(info["reward_terms"]), 14)
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        self.assertGreater(info["rho"], 0)
        self.assertTrue(0.1 <= info["beta"] <= 1.0)
        self.assertNotIn("is_success", info)                 # only reported when the episode ends

    def test_the_episode_ends_by_truncation_with_a_success_score(self):
        env = make(max_steps=10)
        env.reset(seed=1)
        for i in range(10):
            _, _, terminated, truncated, info = env.step(ACTION)
            if i < 9:
                self.assertFalse(truncated)
                self.assertNotIn("is_success", info)
        self.assertTrue(truncated and not terminated)
        self.assertIn("is_success", info)
        self.assertFalse(info["is_success"])                  # a random far start cannot succeed in 10 steps
        self.assertGreater(info["final_pos_err"], 0.05)

    def test_the_final_error_is_the_mean_over_the_final_window(self):
        env = make(max_steps=12)
        env.reset(seed=2)
        e_p = []
        for _ in range(12):
            _, _, _, truncated, info = env.step(ACTION)
            p, R = env.env._tool_pose()
            e_p.append(pose_errors(p, R, env.env.p_goal, env.env.R_goal)[0])
        self.assertAlmostEqual(info["final_pos_err"], float(np.mean(e_p[-30:])), places=9)   # fewer than 30 steps: all of them


class TestSeeding(unittest.TestCase):
    def rollout(self, seed, n=8):
        env = make()
        obs, _ = env.reset(seed=seed)
        out = [obs]
        for _ in range(n):
            obs, reward, *_ = env.step(ACTION)
            out += [obs, reward]
        return out

    def test_the_same_seed_gives_the_same_episode(self):
        a, b = self.rollout(5), self.rollout(5)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))

    def test_different_seeds_give_different_goals(self):
        self.assertFalse(np.array_equal(self.rollout(5)[0], self.rollout(6)[0]))

    def test_reset_without_a_seed_keeps_going_and_gives_new_goals(self):
        env = make()
        first, _ = env.reset(seed=3)
        second, _ = env.reset()
        self.assertFalse(np.array_equal(first[14:17], second[14:17]))       # the goal-position block

    def test_reseeding_in_the_same_env_repeats_the_first_episode(self):
        env = make()
        a, _ = env.reset(seed=9)
        env.step(ACTION)
        b, _ = env.reset(seed=9)
        self.assertTrue(np.array_equal(a, b))

    def test_noise_uses_its_own_generator_after_reseeding(self):
        clean, noisy = make(obs_noise=0.0), make(obs_noise=0.01)
        a, _ = clean.reset(seed=4)
        b, _ = noisy.reset(seed=4)
        self.assertTrue(np.array_equal(clean.env.p_goal, noisy.env.p_goal))   # same goal with or without noise
        self.assertFalse(np.array_equal(a, b))                                # but the joint blocks differ


class TestVectorEnv(unittest.TestCase):
    def test_runs_in_a_synchronous_vector_env_across_episode_ends(self):
        venv = gym.vector.SyncVectorEnv([lambda s=s: make(seed=s, max_steps=5) for s in range(3)])
        obs, _ = venv.reset(seed=0)
        self.assertEqual(obs.shape, (3, 54))
        ended = 0
        for _ in range(12):                                  # past two episode ends: automatic resets must work
            obs, reward, terminated, truncated, info = venv.step(np.stack([ACTION] * 3))
            ended += int(truncated.sum())
            self.assertTrue(np.all(np.isfinite(obs)))
        self.assertEqual(ended, 6)                           # 3 envs x 2 episodes of 5 steps
        venv.close()


if __name__ == "__main__":
    unittest.main()
