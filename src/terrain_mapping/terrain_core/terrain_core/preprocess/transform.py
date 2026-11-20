"""Move points from one coordinate frame into another."""
import numpy as np


def transform_points(T_a_b, points_b):
    """Re-express points given in frame b in frame a.

    T_a_b:    4x4 pose of frame b in frame a (slam_core naming: p_a = T_a_b @ p_b),
              e.g. T_odom_cam = where the camera is in odom.
    points_b: (N, 3) points in frame b.
    Returns (N, 3) points in frame a.
    """
    R, t = T_a_b[:3, :3], T_a_b[:3, 3]
    return np.einsum("ij,nj->ni", R, points_b) + t
