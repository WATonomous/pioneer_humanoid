"""The odometry tree of keyframes: relative poses (with uncertainty) between any two keyframes.

Each keyframe has one odometry edge to a previous keyframe (prev_id), so the keyframes form a tree, not always a
chain (VO can track against an older keyframe). Every keyframe caches its pose and covariance from the root, so
the odometry estimate between any two keyframes costs a few matrix products instead of a walk along the path.
The covariance between a and b is taken relative to their lowest common ancestor, which keeps errors shared by
both paths from being counted twice.
"""
import numpy as np

from slam_core.types.pose import inv_T

from .se3 import adjoint, compose, invert


class KeyframeGraph:
    def __init__(self):
        self.kfs = {}
        self.depth = {}     # odometry edges from the root
        self.T_root = {}    # pose of each keyframe in the root keyframe's frame
        self.S_root = {}    # its 6x6 covariance
        self.arc = {}       # distance travelled (sum of odometry step lengths) from the root

    def __len__(self):
        return len(self.kfs)

    # Keyframes must be added after the keyframe they hang off (prev_id), which is how VO creates them
    def add(self, kf):
        if kf.prev_id < 0:
            T, S, d = np.eye(4), np.zeros((6, 6)), 0
        else:
            T, S = compose(self.T_root[kf.prev_id], self.S_root[kf.prev_id], kf.T_prev_kf, kf.covariance)
            d = self.depth[kf.prev_id] + 1
        self.kfs[kf.id], self.depth[kf.id], self.T_root[kf.id], self.S_root[kf.id] = kf, d, T, S
        self.arc[kf.id] = 0.0 if kf.prev_id < 0 else self.arc[kf.prev_id] + float(np.linalg.norm(kf.T_prev_kf[:3, 3]))

    # The id where the paths from a and b meet, or None if they are not connected
    def common_ancestor(self, a, b):
        while a != b:
            if a < 0 or b < 0:
                return None
            if self.depth[a] >= self.depth[b]:
                a = self.kfs[a].prev_id
            else:
                b = self.kfs[b].prev_id
        return a if a >= 0 else None

    # Number of odometry edges between a and b
    def hops(self, a, b):
        lca = self.common_ancestor(a, b)
        if lca is None:
            return 10 ** 9
        return self.depth[a] + self.depth[b] - 2 * self.depth[lca]

    # Distance travelled along the odometry path between a and b, metres (a robot standing still travels none, however
    # many keyframes it makes)
    def path_length(self, a, b):
        lca = self.common_ancestor(a, b)
        if lca is None:
            return float("inf")
        return self.arc[a] + self.arc[b] - 2 * self.arc[lca]

    # Pose of keyframe i in ancestor lca's frame, with covariance. The root-relative covariance of i is the
    # covariance of lca (moved into i's frame) plus the covariance of the path from lca down to i, so that path's
    # covariance is what is left after taking the first part out
    def _below(self, lca, i):
        T = inv_T(self.T_root[lca]) @ self.T_root[i]
        A = adjoint(inv_T(T))
        S = self.S_root[i] - A @ self.S_root[lca] @ A.T
        return T, (S + S.T) / 2

    # Odometry's estimate of T_a_b (pose of keyframe b in keyframe a's frame) and its 6x6 covariance.
    # None if a and b are not connected
    def relative(self, a, b):
        lca = self.common_ancestor(a, b)
        if lca is None:
            return None
        T_la, S_la = self._below(lca, a)
        T_lb, S_lb = self._below(lca, b)
        T_al, S_al = invert(T_la, S_la)
        return compose(T_al, S_al, T_lb, S_lb)
