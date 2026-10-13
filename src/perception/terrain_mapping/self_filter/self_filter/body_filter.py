"""Remove points that hit the robot's own body.

Plain numpy: no ROS and no URDF library. The caller passes in where each link
is (from yourdfpy offline, from TF on the robot) and a box around each link.
"""
import numpy as np


def inv_T(T):
    """Inverse of a 4x4 transform: rotate back, then move back."""
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points(T, pts):
    """Apply a 4x4 transform to every row of an (N, 3) array."""
    # Same as pts @ R.T + t; einsum avoids a false-alarm warning on Apple Silicon.
    return np.einsum("ij,nj->ni", T[:3, :3], pts) + T[:3, 3]


def inside_box(pts, lo, hi, padding):
    """True for points inside the box [lo, hi] grown by `padding` on every side.

    pts must already be in the box's own frame. lo and hi are the (3,) min and
    max corners.
    """
    return np.all((pts > lo - padding) & (pts < hi + padding), axis=1)


def body_mask(points, link_poses, link_boxes, padding):
    """True for every point inside any link's padded box.

    points:     (N, 3) points, in the same frame as the poses (e.g. "base").
    link_poses: {link name: 4x4 pose of the link in that frame}.
    link_boxes: {link name: (lo, hi)} box corners in the link's own frame.
    """
    mask = np.zeros(len(points), dtype=bool)
    for link, (lo, hi) in link_boxes.items():
        points_in_link = transform_points(inv_T(link_poses[link]), points)
        mask |= inside_box(points_in_link, lo, hi, padding)
    return mask
