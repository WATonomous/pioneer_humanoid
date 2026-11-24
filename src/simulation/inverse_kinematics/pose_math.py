import numpy as np

SCALES = {"coarse": (0.01, np.radians(20.0)), "fine": (0.0005, np.radians(0.25))}
KEYPOINT_OFFSETS = 0.3 * np.eye(3)  # 3 points, 0.3 m along each gripper axis (one per row, in the gripper frame)
TOL_POS = 1e-3                      # m, position tolerance
TOL_ROT = np.radians(0.2)           # rad, rotation tolerance
BEST_SCALE = 20.0                   # tolerance units, scale of the best-so-far features


def scaled(e_pos, e_rot, scales):
    return np.concatenate([np.clip(e_pos / scales[0], -1.0, 1.0),
                           np.clip(e_rot / scales[1], -1.0, 1.0)])  # 6 numbers


def rot_log(R):
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])  # sin(theta) * axis
    s = np.linalg.norm(w)                      # sin(theta)
    c = 0.5 * (np.trace(R) - 1.0)              # cos(theta)
    theta = np.arctan2(s, c)
    if s < 1e-9:                               # no rotation: the first-order answer is w itself
        return w
    return w * (theta / s)


def pose_errors(p_t, R_t, p_g, R_g):
    """Position error (3,) in m and rotation error vector (3,) in rad, both in the gripper's frame."""
    e_pos = R_t.T @ (p_g - p_t)
    e_rot = rot_log(R_t.T @ R_g)
    return e_pos, e_rot


def pose_feedback(p_t, R_t, p_g, R_g):
    e_pos, e_rot = pose_errors(p_t, R_t, p_g, R_g)

    e_coarse = scaled(e_pos, e_rot, SCALES["coarse"])
    e_fine = scaled(e_pos, e_rot, SCALES["fine"])

    return e_coarse, e_fine


def keypoint_errors(p_t, R_t, p_g, R_g):
    """(9,) goal minus current position of three points on the gripper, in the base frame."""
    pts_t = p_t + KEYPOINT_OFFSETS @ R_t.T  # each row: a point in the base frame
    pts_g = p_g + KEYPOINT_OFFSETS @ R_g.T
    return (pts_g - pts_t).ravel()


def rho(e_pos, e_rot):
    """Error in tolerance units: 1 means exactly at the 1 mm / 0.2 deg tolerance, <= 1 means inside it."""
    return max(np.linalg.norm(e_pos) / TOL_POS, np.linalg.norm(e_rot) / TOL_ROT)


def best_so_far(best_prev, rho_t):
    """Returns (new best rho, (2,) features). Pass best_prev=None on the first step of a new goal."""
    if best_prev is None:
        best_prev = rho_t
    features = np.array([np.clip(best_prev / BEST_SCALE, 0.0, 1.0),
                         np.clip((rho_t - best_prev) / BEST_SCALE, 0.0, 1.0)])
    return min(best_prev, rho_t), features
