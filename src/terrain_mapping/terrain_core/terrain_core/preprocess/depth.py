"""Depth image to 3D points (pinhole camera model)."""
import numpy as np


def depth_to_points(depth, K):
    """Turn a depth image into 3D points in the camera optical frame.

    depth: (H, W) float32 depth in metres, 0 = no reading.
    K:     3x3 intrinsics [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    Returns (N, 3) float32 points, one per pixel with depth > 0.
    Camera optical frame: x right, y down, z forward (out of the lens).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))   # column (u) and row (v) of every pixel

    valid = depth > 0                                 # skip pixels with no reading
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.column_stack([x, y, z]).astype(np.float32)
