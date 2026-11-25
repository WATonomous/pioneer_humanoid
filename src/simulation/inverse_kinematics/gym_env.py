"""PolicyIK as a Gymnasium environment (start of task A5), so PPO libraries (Stable-Baselines3, CleanRL, ...) can train on it.

    env = PolicyIKGymEnv(EnvConfig(obs_noise=0.01, resample_goals=True, max_steps=360), seed=0)   # training
    env = PolicyIKGymEnv(seed=0)                                                                   # evaluation
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Observation: 54 float32 numbers (see PolicyIKEnv._observation). Action: 7 float32 numbers in [-1, 1].
info on every step: "reward_terms" (the 11 weighted reward terms), "rho" and "beta".
When an episode ends (truncated) it also has "is_success" (the paper's strict criterion: errors averaged over the final
second are within 1 mm and 0.2 deg), "final_pos_err" (m) and "final_rot_err" (rad), which Stable-Baselines3 logs.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pose_math as pm
from env import EnvConfig, PolicyIKEnv
from metrics import episode_metrics, pose_errors

OBS_DIM = 54
ACTION_DIM = 7


class PolicyIKGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, seed=0):
        super().__init__()
        self.env = PolicyIKEnv(cfg or EnvConfig(), seed=seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self._e_p, self._e_R = [], []        # per-step errors of the episode so far, for the success score

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)             # Gymnasium's own generator (unused here, kept for the standard behaviour)
        if seed is not None:
            self.env.seed(seed)              # reseeds the goals and the observation noise
        self._e_p, self._e_R = [], []
        return self.env.reset(), {}

    def step(self, action):
        obs, reward, terminated, truncated, terms = self.env.step(np.asarray(action, dtype=np.float64))
        e = self.env
        p, R = e._tool_pose()
        ep, eR = pose_errors(p, R, e.p_goal, e.R_goal)
        self._e_p.append(ep)
        self._e_R.append(eR)
        info = {"reward_terms": terms, "rho": float(pm.rho(*pm.pose_errors(p, R, e.p_goal, e.R_goal))), "beta": float(e.beta)}
        if terminated or truncated:
            score = episode_metrics(self._e_p, self._e_R)
            info.update(is_success=bool(score["strict"]), final_pos_err=float(score["final_pos_err"]),
                        final_rot_err=float(score["final_rot_err"]))
        return obs, float(reward), terminated, truncated, info

    def close(self):
        pass
