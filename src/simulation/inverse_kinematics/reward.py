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
    w_action_rate: float = 5e-4; w_joint_vel: float = 5e-4
    w_limit: float = 1.0;        limit_margin: float = 0.05;  limit_gate_rho: float = 10.0

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

    def step(self, p_t, R_t, p_g, R_g, q, qd, a, lo, hi):
        """Reward for the state reached after applying action `a`. Returns (total, terms)."""
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


        weights = {
            "linear position": linear_position_reward * RewardConfig.w_pos_linear,
            "fine position": fine_position_reward * RewardConfig.w_pos_tanh,
            "keypiont wide": keypoint_wide_reward * RewardConfig.w_kp_wide,
            "keypoint sharp": keypoint_sharp_reward * RewardConfig.w_kp_sharp,
            "improvement": improvement_reward * RewardConfig.w_improve,
            "shaping": shaping_reward * RewardConfig.w_shaping,
            "backslide": backslide_reward * RewardConfig.w_back,
            "dwell": dwell_reward * RewardConfig.w_dwell,
            "action rate": action_rate_reward * RewardConfig.w_action_rate,
            "joint velocity": joint_vel_reward * RewardConfig.w_joint_vel,
            "joint limits": limit_reward * RewardConfig.w_limit
        }

        return sum(weights.values()), weights
