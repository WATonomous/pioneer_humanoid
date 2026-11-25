"""PolicyIK environment (task A4): the left arm in MuJoCo, reaching random goal poses.

One episode = one goal (or, with cfg.resample_goals, a new goal every goal_every steps). Each step:
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
from arm_sim import EPISODE_STEPS, Sim
from reward import PoseReward
from sampling import sample_goal, sample_start


@dataclass
class EnvConfig:
    # TODO(A3): these are placeholders until the PD / action-scale tuning gives real numbers
    eta: float = 0.05            # largest reference change per step (rad) from a full-size action
    max_steps: int = EPISODE_STEPS   # episode time-out (steps): the shared value (160 = 5.3 s at 30 Hz)
    # Half-width of uniform noise on the joint-position and joint-velocity channels of the observation.
    # 0 for evaluation; the paper trains with 0.01.
    obs_noise: float = 0.0

    # Goal resampling (the paper's training): a new goal every goal_every steps WITHOUT moving the arm, so one
    # episode chains several attempts. Off by default: evaluation uses one goal per episode.
    resample_goals: bool = False
    goal_every: int = 120            # steps between goals (4 s at 30 Hz)

    # Error-dependent action gain beta (paper Eq. 6 and Appendix A; A3 recommends these values).
    # beta is 1 once either error is above its upper threshold u, and beta_min once both are below their lower l.
    beta_min: float = 0.1
    l_pos: float = 0.5e-3                # m
    u_pos: float = 9.31e-3               # m
    l_rot: float = np.radians(0.415)     # rad
    u_rot: float = np.radians(2.328)     # rad


def gain(e, l, u, beta_min):
    """One error e -> a gain from beta_min (e <= l) rising in a straight line to 1 (e >= u)."""
    return beta_min + (1.0 - beta_min) * float(np.clip((e - l) / (u - l), 0.0, 1.0))


class PolicyIKEnv:
    def __init__(self, cfg=None, seed=0):
        self.cfg = cfg or EnvConfig()
        # The arm and its physics come from arm_sim.Sim, the settings the team agreed on (1/600 s physics, 20 substeps
        # per 30 Hz control step, implicit-fast integrator). Don't set those here. No gravity compensation, as in the paper.
        self.sim = Sim(gravity_comp=False)
        self.arm = self.sim.arm
        self.model, self.data = self.sim.model, self.sim.data
        self.seed(seed)
        self.reward = PoseReward()

        # A second copy of the arm only for sampling goals and starts. sample_goal moves the arm it is given to
        # compute the goal pose, which would teleport the simulated arm in the middle of an episode.
        self.goal_arm = LeftArm()

        # joint limits as two (7,) arrays, made once: arm.limits is (7, 2) with columns [lower, upper]
        self.lo = self.arm.limits[:, 0].copy()
        self.hi = self.arm.limits[:, 1].copy()

        self.z = None           # accumulated joint reference (7,)
        self.p_goal = None       # goal tool position (3,)
        self.R_goal = None       # goal tool orientation (3, 3)
        self.t = 0               # steps since reset
        self.best_so_far = None
        self.a_prev = None
        

    def seed(self, seed):
        """(Re)start the random generators: one for goals and starts, and a separate one for observation noise,
        so turning noise on or off never changes which goals and starts you get."""
        self.rng = np.random.default_rng(seed)
        self.noise_rng = np.random.default_rng([seed, 1])

    def reset(self):
        """New goal and start pose. Returns the first observation."""
        self.t = 0 # reset steps
        self.best_so_far = None # reset best
        self.a_prev = np.zeros(7) # no previous action at the start of an episode
        q_goal, self.p_goal, self.R_goal = sample_goal(self.goal_arm, self.rng)
        start_q = sample_start(self.goal_arm, self.rng) # Random angle positions for joints
        self.z = start_q.copy()

        # Puts the left arm at start_q, points the motors at it, zeroes the velocities and runs forward kinematics
        self.sim.reset(start_q)

        p_t, R_t = self._tool_pose()
        e_pos, e_rot = pm.pose_errors(p_t, R_t, self.p_goal, self.R_goal)
        rho0 = pm.rho(e_pos, e_rot)
        self.reward.reset(rho0)

        return self._observation()

    def step(self, action):
        """Apply one action. Returns (obs, reward, terminated, truncated, info)."""
        action = np.clip(np.array(action), -1, 1, out=None)

        p_t, R_t = self._tool_pose()
        e_pos, e_rot = pm.pose_errors(p_t, R_t, self.p_goal, self.R_goal)

        # Compute stuff
        self.rho = pm.rho(e_pos, e_rot)
        c = self.cfg
        e_p, e_R = np.linalg.norm(e_pos), np.linalg.norm(e_rot)        # m and rad
        self.beta = max(gain(e_p, c.l_pos, c.u_pos, c.beta_min), gain(e_R, c.l_rot, c.u_rot, c.beta_min))
        self.z = np.clip(self.z + self.cfg.eta * self.beta * action, self.lo, self.hi)

        # Sets the motor targets to z and runs the 20 physics substeps of one control step
        self.sim.step(self.z)

        # Get new data
        q = self.data.qpos[self.arm.qadr]
        qd = self.data.qvel[self.arm.qadr]
        p_t, R_t = self._tool_pose()

        reward, terms = self.reward.step(p_t, R_t, self.p_goal, self.R_goal, q, qd, action, self.lo, self.hi)

        self.t += 1

        truncated = False
        terminated = False
        if self.t >= self.cfg.max_steps:
            truncated = True

        self.a_prev = action

        # Every goal_every steps, a new goal while the arm stays where it is (not on the last step of the episode)
        if self.cfg.resample_goals and not truncated and self.t % self.cfg.goal_every == 0:
            self._resample_goal(p_t, R_t)

        return self._observation(), reward, terminated, truncated, terms

    def _resample_goal(self, p_t, R_t):
        """Pick a new goal for the arm's current pose (p_t, R_t) and restart the per-goal memory."""
        _, self.p_goal, self.R_goal = sample_goal(self.goal_arm, self.rng)
        self.best_so_far = None        # the running best starts again at the new goal's rho (set in _observation)
        e_pos, e_rot = pm.pose_errors(p_t, R_t, self.p_goal, self.R_goal)
        # a_prev is kept: the arm keeps moving, so the action-rate cost continues from the last action
        self.reward.reset(pm.rho(e_pos, e_rot), a_prev=self.a_prev)

    def _tool_pose(self):
        """Current tool point position (3,) and orientation (3, 3) in the base frame, read from the simulation."""
        # mj_step leaves positions of bodies and sites one physics substep old, so recompute them first
        mujoco.mj_forward(self.model, self.data)
        return self.sim.tcp_pose()

    def _observation(self):
        """The 54-number observation, in the order of Eq. 3 of the PolicyIK paper (float32):
        joints - rest (7), joint velocities (7), goal position (3), keypoint errors (9), coarse error (6),
        fine error (6), best-so-far features (2), command state z - q (7), previous action (7)."""
        blocks = []    # each piece of the observation is appended here, and joined once at the end

        # get position and velocity of all joints
        q = self.data.qpos[self.arm.qadr]
        qd = self.data.qvel[self.arm.qadr]

        # Training noise goes on these two channels only (the paper's Appendix A). The physics, the reward and the
        # command state z - q below all use the true values.
        q_obs, qd_obs = q, qd
        if self.cfg.obs_noise > 0:
            n = self.cfg.obs_noise
            q_obs = q + self.noise_rng.uniform(-n, n, q.shape)
            qd_obs = qd + self.noise_rng.uniform(-n, n, qd.shape)

        blocks.append(q_obs - self.arm.rest) # Current joints - rest joints
        blocks.append(qd_obs) # Adds velocities

        blocks.append(self.p_goal)
        p_t, R_t = self._tool_pose()
        e_pos, e_rot = pm.pose_errors(p_t, R_t, self.p_goal, self.R_goal)
        blocks.append(pm.keypoint_errors(p_t, R_t, self.p_goal, self.R_goal))

        rho = pm.rho(e_pos, e_rot)
        self.best_so_far, features = pm.best_so_far(self.best_so_far, rho)    # call once per observation

        coarse, fine = pm.pose_feedback(p_t, R_t, self.p_goal, self.R_goal)
        blocks.append(coarse)
        blocks.append(fine)
        blocks.append(features)

        blocks.append(self.z - q)
        blocks.append(self.a_prev)

        return np.concatenate(blocks).astype(np.float32)
