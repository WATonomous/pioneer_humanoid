"""PolicyIK environment (task A4): the left arm in MuJoCo, reaching random goal poses.

One episode = one goal. Each step:
    policy action a (7 numbers, about -1..1)
      -> move the accumulated joint reference:  z = clip(z + eta * beta * a)
      -> hold z as the PD position target and run the physics for one policy step
      -> read the arm, compute the reward, build the next observation
"""
from dataclasses import dataclass

import mujoco
import numpy as np

import pose_math as pm
from arm_model import LeftArm
from reward import PoseReward
from sampling import sample_goal, sample_start


@dataclass
class EnvConfig:
    # TODO(A3): these are placeholders until the PD / action-scale tuning gives real numbers
    eta: float = 0.05            # largest reference change per step (rad) from a full-size action
    policy_hz: float = 30.0      # the reward config uses dt = 1/30
    max_steps: int = 300         # episode time-out (steps)
    obs_noise: float = 0.0       # observation noise, off for now


class PolicyIKEnv:
    def __init__(self, cfg=None, seed=0):
        self.cfg = cfg or EnvConfig()
        self.arm = LeftArm()
        self.model, self.data = self.arm.model, self.arm.data
        self.rng = np.random.default_rng(seed)
        self.reward = PoseReward()

        # physics steps per policy step: the model's timestep is 0.002 s, so 30 Hz is 16.7 steps (not a whole number)
        self.n_substeps = round(1.0 / self.cfg.policy_hz / self.model.opt.timestep)

        # left arm: the first 7 actuators are the joint position targets, in the same order as arm.qadr
        # (the model also has the right arm and the grippers, which this env leaves alone)
        self.n_joints = 7

        self.z = None            # accumulated joint reference (7,)
        self.p_goal = None       # goal tool position (3,)
        self.R_goal = None       # goal tool orientation (3, 3)
        self.t = 0               # steps since reset

    def reset(self):
        """New goal and start pose. Returns the first observation."""
        self.t = 0 # reset steps
        q_goal, self.p_goal, self.R_goal = sample_goal(self.arm, self.rng)
        start_q = sample_start(self.arm, self.rng) # Random angle positions for joints
        self.z = start_q.copy()

        self.data.qpos[self.arm.qadr] = start_q
        self.data.ctrl[:self.n_joints] = self.z
        self.data.qvel[:] = 0

        # runs forward kinematic
        mujoco.mj_forward(self.model, self.data)
        p_t = self.data.site_xpos[self.arm.tcp_id].copy()
        R_t = self.data.site_xmat[self.arm.tcp_id].reshape(3, 3).copy()

        e_pos, e_rot = pm.pose_errors(p_t, R_t, self.p_goal, self.R_goal)
        rho0 = pm.rho(e_pos, e_rot)
        self.reward.reset(rho0)

        return self._observation()

    def step(self, action):
        """Apply one action. Returns (obs, reward, terminated, truncated, info)."""
        raise NotImplementedError

    def _observation(self):
        """The observation vector. Layout still to agree (S2)."""
        raise NotImplementedError
