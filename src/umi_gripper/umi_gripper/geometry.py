"""Rigid-transform helpers (numpy only). Quaternions are xyzw, as WebXR sends them; poses are 4x4 matrices."""
from __future__ import annotations

import math

import numpy as np


def quat_to_matrix(xyzw) -> np.ndarray:
    q = np.asarray(xyzw, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError(f"invalid quaternion {xyzw!r}")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(matrix) -> np.ndarray:
    """xyzw with w >= 0."""
    m = np.asarray(matrix, dtype=float)
    trace = np.trace(m)
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q) / np.linalg.norm(q)
    return -q if q[3] < 0 else q


def rotation_vector(matrix) -> np.ndarray:
    """Axis * angle of a rotation matrix (angle in [0, pi])."""
    q = matrix_to_quat(matrix)
    length = np.linalg.norm(q[:3])
    if length < 1e-12:
        return np.zeros(3)
    return q[:3] / length * 2 * math.atan2(length, q[3])


def rotation_exp(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def rpy_deg_to_matrix(rpy_deg) -> np.ndarray:
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.radians(np.asarray(rpy_deg, dtype=float))
    return rotation_exp([0, 0, yaw]) @ rotation_exp([0, pitch, 0]) @ rotation_exp([roll, 0, 0])


def make_pose(rotation=None, position=None) -> np.ndarray:
    pose = np.eye(4)
    if rotation is not None:
        pose[:3, :3] = rotation
    if position is not None:
        pose[:3, 3] = position
    return pose


def invert(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    out = np.eye(4)
    out[:3, :3] = pose[:3, :3].T
    out[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return out


def slerp(q0, q1, fraction: float) -> np.ndarray:
    """Shortest-path spherical interpolation of two xyzw quaternions."""
    q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + fraction * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(min(1.0, dot))
    return (math.sin((1 - fraction) * theta) * q0 + math.sin(fraction * theta) * q1) / math.sin(theta)


def rot6d(rotation) -> np.ndarray:
    """Continuous 6D rotation (Zhou et al. 2019): the first two ROWS of the matrix, as UMI's mat_to_rot6d."""
    r = np.asarray(rotation, dtype=float)
    return np.concatenate([r[0, :], r[1, :]])


def rot6d_to_matrix(value) -> np.ndarray:
    """Inverse of rot6d by Gram-Schmidt, so a network's unnormalised output still gives a proper rotation."""
    v = np.asarray(value, dtype=float)
    a = v[:3] / np.linalg.norm(v[:3])
    b = v[3:] - np.dot(a, v[3:]) * a
    b /= np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)], axis=0)


def pose_to_vector(pose) -> np.ndarray:
    """4x4 -> (9,): position then rot6d."""
    pose = np.asarray(pose, dtype=float)
    return np.concatenate([pose[:3, 3], rot6d(pose[:3, :3])])


def vector_to_pose(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return make_pose(rot6d_to_matrix(vector[3:9]), vector[:3])
