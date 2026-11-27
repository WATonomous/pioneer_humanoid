import pose_math as pm
import numpy as np
from dataclasses import dataclass

@dataclass
class RewardConfig:
    dt: float = 1 / 30
    w_pos_linear: float = 0.2;   w_pos_tanh: float = 0.1;   std_pos: float = 0.10
    w_kp_wide: float = 1.0;      std_kp_wide: float = 0.20
    w_kp_sharp: float = 1.0;     std_kp_sharp: float = 0.05;  kp_scale: float = 0.45
    w_improve: float = 26.831;   w_shaping: float = 1.0
    gamma: float = 0.99;         phi_cap: float = 50.0
    w_back: float = 0.2;         back_cap: float = 3.0
    w_dwell: float = 1.0
    # The paper's values were 5e-4 for both. With them a policy that swings every action between +1 and -1 paid only
    # about 0.01 per step (against about +1.3 from the keypoint terms), so nothing discouraged jitter: raised 40x / 10x.
    w_action_rate: float = 2e-2; w_joint_vel: float = 5e-3
    # w_limit was 1.0; 3.0 since v6 failures all ended pinned at joint limits 300-400 mm from the goal (wrong IK branch)
    w_limit: float = 3.0;        limit_margin: float = 0.05;  limit_gate_rho: float = 10.0
    # Not in the paper. Within rho <= near_rho (about 1 cm of the goal) the arm should be still, but joints kept moving at
    # about 0.2-0.3 rad/s there, so the joint-velocity cost is much heavier near the goal (w_joint_vel_near, vs w_joint_vel
    # elsewhere, so that moving fast to get there is still cheap).
    w_joint_vel_near: float = 0.2;  near_rho: float = 10.0
    # Not in the paper. The arm has 7 joints for a 6-number pose, so one joint motion moves the arm without moving the tool
    # (26% of the motion in the last 3 s of a trained policy), and nothing stopped it drifting. A small pull toward the rest
    # posture (joint offsets as a fraction of each joint's range) keeps that motion in check.
    w_posture: float = 0.05
    # Not in the paper. At the end of an episode the trained policy still flipped the sign of its actions on about half the
    # steps (tool moving ~1.2 mm per step, never settling). Within near_rho the action-rate cost is heavier
    # (w_action_rate_near, vs w_action_rate elsewhere) and the action itself is penalized (w_action_near), so doing nothing is
    # the best move once there. And w_time is a small cost per step while not yet within tolerance (rho > 1), so arriving
    # sooner is worth more (episodes only end by time-out, so a cost that never stopped would be a constant and teach nothing).
    w_action_rate_near: float = 0.2;  w_action_near: float = 0.05;  w_time: float = 0.05

# Helper functions below ------------------------------------------------
def U(rho):
    """Smooth step: about 1 when rho is near 0, 0.5 at rho = 1, about 0 when rho is 2 or more."""
    return 0.5 * (1.0 + np.tanh(2.0 * (1.0 - rho)))

# instead of having position and error weights, u just do 3 position weights that also account for error.
def keypoint_error_centered(p_t, R_t, p_g, R_g, kp_scale):
    """Average distance (m) between three points on the gripper at the goal and at the current pose.
    The points are kp_scale * (axis - [1/3, 1/3, 1/3]), so their centre is the tool point."""
    c = kp_scale * (np.eye(3) - 1.0 / 3.0)      # one point per row, in the gripper frame
    pts_t = p_t + c @ R_t.T
    pts_g = p_g + c @ R_g.T
    return float(np.mean(np.linalg.norm(pts_g - pts_t, axis=1)))

def limit_penalty(q, lo, hi, margin):
    """0 when every joint is more than `margin` (a fraction of its range) from its limits, up to 1 at a limit."""
    m = np.minimum(q - lo, hi - q) / (hi - lo)
    return float(np.clip(1.0 - m.min() / margin, 0.0, 1.0))

class PoseReward:
    def __init__(self, cfg=None):
        self.cfg = cfg or RewardConfig()
        self.best = None        # lowest rho so far for the current goal
        self.phi_prev = None    # last step's shaping potential
        self.a_prev = None      # last step's action
        self.fresh = True       # True right after reset: the progress terms pay nothing that step

    def reset(self, rho0, a_prev=None, n_actions=7):
        """Call when a new goal starts. rho0 is the error (in tolerance units) at that moment."""
        self.best = rho0
        self.phi_prev = -min(self.cfg.phi_cap, rho0)
        self.a_prev = np.zeros(n_actions) if a_prev is None else np.asarray(a_prev, float)
        self.fresh = True

    def step(self, p_t, R_t, p_g, R_g, q, qd, a, lo, hi, q_rest=None):
        """Reward for the state reached after applying action `a`. Returns (total, terms).
        q_rest: the rest posture (7,) for the posture term; without it that term is 0."""
        e_pos, e_rot = pm.pose_errors(p_t, R_t, p_g, R_g)
        e_p = np.linalg.norm(e_pos)
        epsilon = keypoint_error_centered(p_t, R_t, p_g, R_g, RewardConfig.kp_scale)
        rho = pm.rho(e_pos, e_rot)
        phi = -min(RewardConfig.phi_cap, rho)

        # All rewards
        linear_position_reward = -e_p
        fine_position_reward = 1 - np.tanh(e_p / RewardConfig.std_pos)
        keypoint_wide_reward = 1 - np.tanh(epsilon / RewardConfig.std_kp_wide)
        keypoint_sharp_reward = 1 - np.tanh(epsilon / RewardConfig.std_kp_sharp)

        # Progress and dwell use the memory from before this step (self.best, self.phi_prev)
        improvement_reward = max(0.0, U(rho) - U(self.best))
        shaping_reward = RewardConfig.gamma * phi - self.phi_prev
        backslide_reward = -min(RewardConfig.back_cap, max(0.0, rho - self.best))
        dwell_reward = float(rho <= 1.0)
        if self.fresh:  # first step after a new goal: these four pay nothing
            improvement_reward = shaping_reward = backslide_reward = dwell_reward = 0.0

        # Costs, negative numbers
        action_rate_reward = -float(np.sum((a - self.a_prev) ** 2))
        joint_vel_reward = -float(np.sum(np.asarray(qd) ** 2))
        limit_reward = -limit_penalty(q, lo, hi, RewardConfig.limit_margin) if rho > RewardConfig.limit_gate_rho else 0.0
        w_vel = RewardConfig.w_joint_vel_near if rho <= RewardConfig.near_rho else RewardConfig.w_joint_vel
        posture_reward = 0.0 if q_rest is None else -float(np.sum(((np.asarray(q) - q_rest) / (hi - lo)) ** 2))

        near = rho <= RewardConfig.near_rho
        w_rate = RewardConfig.w_action_rate_near if near else RewardConfig.w_action_rate
        action_size_reward = -float(np.sum(np.asarray(a) ** 2)) if near else 0.0
        time_reward = -1.0 if rho > 1.0 else 0.0

        weights = {
            "linear position": linear_position_reward * RewardConfig.w_pos_linear,
            "fine position": fine_position_reward * RewardConfig.w_pos_tanh,
            "keypoint wide": keypoint_wide_reward * RewardConfig.w_kp_wide,
            "keypoint sharp": keypoint_sharp_reward * RewardConfig.w_kp_sharp,
            "improvement": improvement_reward * RewardConfig.w_improve,
            "shaping": shaping_reward * RewardConfig.w_shaping,
            "backslide": backslide_reward * RewardConfig.w_back,
            "dwell": dwell_reward * RewardConfig.w_dwell,
            "action rate": action_rate_reward * w_rate,
            "joint velocity": joint_vel_reward * w_vel,
            "joint limits": limit_reward * RewardConfig.w_limit,
            "posture": posture_reward * RewardConfig.w_posture,
            "action size": action_size_reward * RewardConfig.w_action_near,
            "time": time_reward * RewardConfig.w_time,
        }

        # update new values
        self.best = min(rho, self.best)
        self.phi_prev = phi
        self.a_prev = np.array(a, dtype=float) # Makes a copy too
        self.fresh = False # Not first step

        return sum(weights.values()), weights
