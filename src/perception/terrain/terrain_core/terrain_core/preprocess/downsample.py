"""Thin a point cloud out evenly: at most one point per small cube (voxel)."""
import numpy as np


def voxel_downsample(points, voxel_size):
    """Keep one point per voxel (a cube voxel_size metres on each side).

    points:     (N, 3) points.
    voxel_size: cube edge length in metres, e.g. 0.02 for 2 cm.
    Returns (M, 3) points, M <= N, at most one per occupied voxel.
    """
    voxel = np.floor(points / voxel_size).astype(np.int64)        # which cube each point is in
    _, first = np.unique(voxel, axis=0, return_index=True)       # first point found in each cube
    return points[np.sort(first)]
