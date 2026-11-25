"""A small planar (x, y, yaw) pose graph optimizer, only for the simulator and the demo.

The real backend is GTSAM (pose_graph.py, later iSAM2). This stand-in exists so the demo and tests can show
the effect of loop closure on a trajectory without GTSAM installed. Loop edges get a Huber loss, like the GTSAM
prototype, via a few rounds of reweighting. Needs scipy.
"""
import numpy as np

from slam_core.types.pose import inv_T


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# 3x3 square-root information for residuals ordered (tx, ty, yaw), taken from a 6x6 GTSAM-order covariance
def _sqrt_info(cov6):
    idx = [3, 4, 2]
    C = cov6[np.ix_(idx, idx)]
    return np.linalg.cholesky(np.linalg.inv(C)).T


def _planar(T):
    return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])


# x0: (N, 3) initial poses. odom: [(i, j, T_ij 4x4, cov6)], loops: same. Returns the optimized (N, 3) poses
def optimize(x0, odom, loops, huber=1.345, rounds=4):
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    edges = [(i, j, _planar(T), _sqrt_info(C), False) for i, j, T, C in odom]
    edges += [(i, j, _planar(T), _sqrt_info(C), True) for i, j, T, C in loops]
    N, E = len(x0), len(edges)
    w = np.ones(E)

    def resid(v):
        X = v.reshape(N, 3)
        out = np.empty(3 * E + 3)
        for k, (i, j, Z, L, _) in enumerate(edges):
            c, s = np.cos(X[i, 2]), np.sin(X[i, 2])
            d = X[j, :2] - X[i, :2]
            pred = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], _wrap(X[j, 2] - X[i, 2])])
            e = pred - Z
            e[2] = _wrap(e[2])
            out[3 * k:3 * k + 3] = w[k] * (L @ e)
        out[3 * E:] = 1e3 * (X[0] - x0[0])      # pins the first pose
        return out

    S = lil_matrix((3 * E + 3, 3 * N), dtype=int)
    for k, (i, j, *_) in enumerate(edges):
        S[3 * k:3 * k + 3, 3 * i:3 * i + 3] = 1
        S[3 * k:3 * k + 3, 3 * j:3 * j + 3] = 1
    S[3 * E:, 0:3] = 1

    v = x0.ravel().copy()
    for _ in range(rounds if loops else 1):
        v = least_squares(resid, v, jac_sparsity=S, method="trf", max_nfev=30).x
        # Huber reweighting of the loop edges only: big normalized errors get a smaller weight next round
        w[:] = 1.0
        r0 = resid(v)
        for k, (_, _, _, _, is_loop) in enumerate(edges):
            if is_loop:
                r = np.linalg.norm(r0[3 * k:3 * k + 3])
                w[k] = 1.0 if r <= huber else np.sqrt(huber / r)
    return v.reshape(N, 3)
