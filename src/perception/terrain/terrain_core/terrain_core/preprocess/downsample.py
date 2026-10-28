"""Thin a point cloud out evenly: at most one point per small cube (voxel)."""
import numpy as np

# Each voxel's (x, y, z) cube number is packed into one 64-bit integer, 21 bits
# each. 21 bits hold 2,097,152 values, so cube numbers from -1,048,576 to
# 1,048,575: with 2 cm voxels that is about +-21 km, far more than a camera sees.
_BITS = 21
_OFFSET = 1 << (_BITS - 1)     # 1,048,576: added so every cube number is >= 0


def voxel_downsample(points, voxel_size):
    """Keep one point per voxel (a cube voxel_size metres on each side).

    points:     (N, 3) points.
    voxel_size: cube edge length in metres, e.g. 0.02 for 2 cm.
    Returns (M, 3) points, M <= N, at most one per occupied voxel.
    """
    voxel = np.floor(points / voxel_size).astype(np.int64) + _OFFSET    # cube numbers, all >= 0
    if voxel.size and (voxel.min() < 0 or voxel.max() >= 1 << _BITS):
        raise ValueError("points too far from the origin for this voxel size")
    # One number per cube: x in the top 21 bits, y in the middle, z in the bottom.
    key = (voxel[:, 0] << (2 * _BITS)) | (voxel[:, 1] << _BITS) | voxel[:, 2]
    _, first = np.unique(key, return_index=True)                       # first point in each cube
    return points[np.sort(first)]
