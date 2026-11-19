"""Do two loops agree with each other? (Pairwise consistency, the idea behind PCM / Kimera-RPGO.)

Two loop edges A (old_a -> new_a) and B (old_b -> new_b) plus the odometry between their endpoints form a cycle:
old_a -> new_a (loop A) -> new_b (odometry) -> old_b (loop B backwards) -> old_a (odometry). If both loops are
right the cycle returns to the start. If one is a false loop, it does not. The cycle's error is compared to its
propagated covariance with a chi-squared test. A false loop can be consistent with odometry-only neighbours by
accident, but it cannot be consistent with the true loops, so the largest mutually consistent group wins.
"""
import numpy as np

from .gate import CHI2_6_999
from .se3 import compose, invert, mahalanobis2, pose_error


# Squared Mahalanobis error of the cycle through loops A and B (0 = perfectly consistent). inf if not connected
def cycle_error2(graph, A, B):
    o1 = graph.relative(A.new_id, B.new_id)
    o2 = graph.relative(B.old_id, A.old_id)
    if o1 is None or o2 is None:
        return float("inf")
    T, S = compose(A.T_old_new, A.covariance, *o1)
    T, S = compose(T, S, *invert(B.T_old_new, B.covariance))
    T, S = compose(T, S, *o2)
    return mahalanobis2(pose_error(np.eye(4), T), S)


def consistent(graph, A, B, chi2=CHI2_6_999):
    return cycle_error2(graph, A, B) <= chi2


# Indices of the largest-weight set of loops that all agree with each other.
# adj[i][j] is True when loops i and j are consistent. Exact (Bron-Kerbosch with pivoting) for small sets,
# greedy by weight for big ones
def best_consistent_set(adj, weights, max_exact=30):
    n = len(weights)
    if n == 0:
        return []
    if n > max_exact:
        keep = []
        for i in sorted(range(n), key=lambda k: -weights[k]):
            if all(adj[i][j] for j in keep):
                keep.append(i)
        return sorted(keep)

    best = [-1.0, []]

    def expand(R, P, X):
        if not P and not X:
            w = sum(weights[i] for i in R)
            if w > best[0]:
                best[0], best[1] = w, sorted(R)
            return
        pivot = max(P | X, key=lambda u: sum(adj[u][v] for v in P if v != u))
        for v in list(P - {v for v in range(n) if adj[pivot][v] and v != pivot}):
            nb = {u for u in range(n) if adj[v][u] and u != v}
            expand(R + [v], P & nb, X & nb)
            P = P - {v}
            X = X | {v}

    expand([], set(range(n)), set())
    return best[1]
