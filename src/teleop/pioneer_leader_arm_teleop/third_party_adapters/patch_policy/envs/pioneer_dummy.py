"""Placeholder environment for training on Pioneer demonstrations (not part of the Patch Policy release).

The trainer needs an environment object, but our MuJoCo scenes run in another Python environment, so the trained
policy is rolled out separately (pioneer_adapter/rollout.py). Self-contained: the release's envs/dummy_env.py
imports a helper (utils.aggregate_dct) that the release does not include.
"""
import gym
import numpy as np


class PioneerDummyEnv(gym.Env):
    def __init__(self, id=None, action_dim=7, state_dim=7, views=2, size=224, **_):
        super().__init__()
        self.id, self.action_dim, self.state_dim, self.shape = id, action_dim, state_dim, (views, 3, size, size)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def reset(self, *args, **kwargs):
        return np.zeros(self.shape, dtype=np.float32)

    def step(self, action):
        return np.zeros(self.shape, dtype=np.float32), 0.0, True, {"state": np.zeros(self.state_dim, dtype=np.float32)}
