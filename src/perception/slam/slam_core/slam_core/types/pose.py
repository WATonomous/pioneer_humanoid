"""Pose helpers. Poses are 4x4 numpy arrays.

Conventions used across the project:
  T_a_b = pose of frame b in frame a, so p_a = T_a_b @ p_b and T_a_c = T_a_b @ T_b_c.
  Cameras use the OpenCV optical frame (x right, y down, z forward). Only slam_ros
  converts to ROS body frames.
  Quaternions are (qx, qy, qz, qw) like TUM/ROS/scipy. gtsam.Rot3.Quaternion() wants
  (w, x, y, z), so watch out there.
  6x6 covariances follow GTSAM Pose3: rotation first (rx ry rz tx ty tz), right-hand
  perturbation. ROS PoseWithCovariance puts translation first.
"""
import numpy as np


# Build a 4x4 pose from R and t. Missing parts default to identity / zero.
def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


# Invert a pose using R^T and -R^T t. Cheaper and more exact than np.linalg.inv.
def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


# Check that T is a valid rigid transform: 4x4, finite, orthonormal R with det +1,
# bottom row 0 0 0 1.
def is_pose(T, tol=1e-6):
    T = np.asarray(T)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    return (np.allclose(R.T @ R, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1) < tol
            and np.allclose(T[3], [0, 0, 0, 1], atol=tol))


# Rotation angle of R in radians (0 to pi), from its trace.
def rotation_angle(R):
    # clip guards against trace drifting slightly outside [-1, 3]
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


# Quaternion (x, y, z, w) to 3x3 rotation. Normalizes the input first.
def quat_to_rot(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# 3x3 rotation to quaternion (x, y, z, w), always with w >= 0.
def rot_to_quat(R):
    # Shepperd's method: branch on the largest diagonal term for stability
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q if q[3] >= 0 else -q  # q and -q are the same rotation


# TUM row (tx ty tz qx qy qz qw, no timestamp) to 4x4 pose.
def from_tum(vals):
    vals = np.asarray(vals, dtype=float)
    return make_T(quat_to_rot(vals[3:7]), vals[:3])


# 4x4 pose to TUM row (tx ty tz qx qy qz qw).
def to_tum(T):
    return np.concatenate([T[:3, 3], rot_to_quat(T[:3, :3])])


# Diagonal 6x6 covariance from std devs, GTSAM order (rotation first).
# rot in rad, trans in m; each can be a scalar or a 3-vector.
def covariance_from_sigmas(rot, trans):
    sig = np.concatenate([np.broadcast_to(rot, 3), np.broadcast_to(trans, 3)]).astype(float)
    return np.diag(sig ** 2)



