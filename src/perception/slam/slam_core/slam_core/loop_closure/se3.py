"""SE(3) helpers for loop closure: error vectors and covariance propagation.

Same conventions as slam_core.types.pose: T_a_b is the pose of b in a, 6x6 covariances are in
GTSAM order (rotation first, then translation) with a right-hand perturbation, T_true = T_est @ Exp(xi).
Only numpy, so everything here runs (and is tested) without GTSAM, OpenCV or a GPU.
"""
import numpy as np

from slam_core.types.pose import inv_T, make_T


# 3x3 matrix such that skew(v) @ w == cross(v, w)
def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# Rotation matrix to rotation vector (axis * angle), the log map of SO(3)
def rotvec(R):
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-8:
        return w
    if np.pi - angle < 1e-6:
        # Within a hair of 180 degrees sin(angle) vanishes, so take the axis from the symmetric part instead
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(A[k, k])
        # Pick the sign that agrees with the (tiny) skew part
        if np.dot(axis, w) < 0:
            axis = -axis
        return axis * angle
    return w * (angle / np.sin(angle))


# Rotation vector to rotation matrix (Rodrigues), the exp map of SO(3)
def rot_from_vec(v):
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    K = skew(v)
    if angle < 1e-8:
        return np.eye(3) + K
    return np.eye(3) + (np.sin(angle) / angle) * K + ((1 - np.cos(angle)) / angle ** 2) * K @ K


# 6x6 adjoint of T in GTSAM order: moves a (rotation, translation) error from one frame to another
def adjoint(T):
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, 3:] = R
    A[3:, :3] = skew(t) @ R
    return A


# How far T_meas is from T_est, as a 6-vector (rotation vector, translation) in T_est's frame.
# This is the small-error version of the SE(3) log, which is all the covariance math needs
def pose_error(T_est, T_meas):
    E = inv_T(T_est) @ T_meas
    return np.concatenate([rotvec(E[:3, :3]), E[:3, 3]])


# Chain two uncertain poses: T_ac = T_ab @ T_bc. Returns T_ac and its covariance.
# The error of T_ab is moved into the frame of c with Ad(T_bc^-1); the two errors are assumed independent
def compose(T_ab, S_ab, T_bc, S_bc):
    A = adjoint(inv_T(T_bc))
    return T_ab @ T_bc, A @ S_ab @ A.T + S_bc


# Invert an uncertain pose. Returns T^-1 and its covariance, Ad(T) S Ad(T)^T
def invert(T, S):
    A = adjoint(T)
    return inv_T(T), A @ S @ A.T


# Squared Mahalanobis distance of error vector r under covariance S
def mahalanobis2(r, S):
    return float(r @ np.linalg.solve(S, r))


# 4x4 pose from planar (x, y, yaw), handy for the simulator and tests
def planar_pose(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return make_T(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), [x, y, 0.0])
