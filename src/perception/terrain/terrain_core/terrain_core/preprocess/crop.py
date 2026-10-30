"""Keep only the points the camera measures well."""


def in_depth_range(points, z_min=0.3, z_max=3.0):
    """True for points whose depth is between z_min and z_max (metres).

    points: (N, 3) points in the camera optical frame (z = straight ahead).
    Returns an (N,) True/False mask, like body_mask: points[mask] keeps them.
    """
    z = points[:, 2]
    return (z >= z_min) & (z <= z_max)
