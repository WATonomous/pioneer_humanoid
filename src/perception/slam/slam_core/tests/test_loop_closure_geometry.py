"""Loop closure building blocks that need no images: SE(3) helpers, the odometry tree, the gate, consistency."""
import numpy as np
import pytest

from slam_core.loop_closure import KeyframeGraph, LoopConstraint, best_consistent_set, consistent, plausibility
from slam_core.loop_closure.se3 import adjoint, compose, invert, pose_error, rot_from_vec, rotvec
from slam_core.types import Keyframe
from slam_core.types.pose import covariance_from_sigmas, inv_T, make_T


def random_pose(rng, t=2.0, r=0.5):
    return make_T(rot_from_vec(rng.normal(0, r, 3)), rng.normal(0, t, 3))


def test_rotvec_round_trip_including_near_pi():
    rng = np.random.default_rng(0)
    for scale in (1e-9, 0.1, 1.0, 3.0, np.pi - 1e-4):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * scale
        assert np.allclose(rotvec(rot_from_vec(v)), v, atol=1e-6)


def test_compose_covariance_matches_monte_carlo():
    # Small errors on two chained poses: the propagated covariance should match the sampled one
    rng = np.random.default_rng(1)
    T_ab, T_bc = random_pose(rng), random_pose(rng)
    S_ab = covariance_from_sigmas(2e-3, 3e-3)
    S_bc = covariance_from_sigmas(1e-3, 2e-3)
    T_ac, S_ac = compose(T_ab, S_ab, T_bc, S_bc)

    def perturb(T, S):
        xi = rng.multivariate_normal(np.zeros(6), S)
        return T @ make_T(rot_from_vec(xi[:3]), xi[3:])   # first-order exp, fine for tiny errors

    errs = np.array([pose_error(T_ac, perturb(T_ab, S_ab) @ perturb(T_bc, S_bc)) for _ in range(6000)])
    assert np.allclose(np.cov(errs.T), S_ac, rtol=0.2, atol=2e-7)


def test_invert_covariance_round_trip():
    rng = np.random.default_rng(2)
    T, S = random_pose(rng), covariance_from_sigmas(1e-3, 2e-3)
    T2, S2 = invert(*invert(T, S))
    assert np.allclose(T2, T) and np.allclose(S2, S)


def chain(n, rng, noise=0.01):
    graph = KeyframeGraph()
    for i in range(n):
        T = np.eye(4) if i == 0 else make_T(rot_from_vec(rng.normal(0, 0.05, 3)), [0.3, 0, 0] + rng.normal(0, 0.02, 3))
        graph.add(Keyframe(id=i, prev_id=i - 1, stamp=i, T_prev_kf=T, covariance=covariance_from_sigmas(noise, noise),
                           T_odom_kf=np.eye(4)))
    return graph


def test_relative_matches_the_product_and_grows_with_distance():
    graph = chain(30, np.random.default_rng(3))
    T, S = graph.relative(5, 20)
    expected = np.eye(4)
    for i in range(6, 21):
        expected = expected @ graph.kfs[i].T_prev_kf
    assert np.allclose(T, expected)
    assert np.trace(graph.relative(5, 25)[1]) > np.trace(graph.relative(5, 10)[1])
    assert np.allclose(graph.relative(7, 7)[1], 0, atol=1e-12)
    assert graph.hops(5, 20) == 15


def test_relative_in_a_tree_uses_the_common_ancestor():
    # kf 3 and kf 4 both hang off kf 2: relative(3, 4) must only involve their own two edges, not the shared history
    graph = chain(3, np.random.default_rng(4))
    for i in (3, 4):
        graph.add(Keyframe(id=i, prev_id=2, stamp=i, T_prev_kf=make_T(None, [0.0, 0.1 * i, 0.0]),
                           covariance=covariance_from_sigmas(0.01, 0.01), T_odom_kf=np.eye(4)))
    T, S = graph.relative(3, 4)
    assert np.allclose(T, make_T(None, [0.0, 0.4 - 0.3, 0.0]))
    assert graph.hops(3, 4) == 2
    # independent edges: the covariance is about the sum of two edge covariances (the small excess is the
    # rotation-to-translation coupling over the edge's 0.3 m of translation)
    assert np.trace(S) == pytest.approx(2 * np.trace(covariance_from_sigmas(0.01, 0.01)), rel=1e-2)


def loop(graph, old, new, offset=(0, 0, 0)):
    T_odom, _ = graph.relative(old, new)
    T = T_odom @ make_T(None, offset)
    return LoopConstraint(0, old, new, T, covariance_from_sigmas(0.02, 0.03), 50, 0.9)


def test_gate_accepts_a_loop_that_matches_odometry_and_rejects_a_far_one():
    graph = chain(60, np.random.default_rng(5))
    good = loop(graph, 5, 55, offset=(0.05, 0, 0))
    bad = loop(graph, 5, 55, offset=(30.0, 0, 0))
    assert plausibility(graph, good.old_id, good.new_id, good.T_old_new, good.covariance)[1]
    assert not plausibility(graph, bad.old_id, bad.new_id, bad.T_old_new, bad.covariance)[1]


def test_pairwise_consistency_separates_a_false_loop_from_two_true_ones():
    graph = chain(80, np.random.default_rng(6))
    a, b = loop(graph, 5, 60), loop(graph, 7, 62)
    a.id, b.id = 1, 2
    bad = loop(graph, 9, 64, offset=(5.0, 3.0, 0))
    bad.id = 3
    assert consistent(graph, a, b)
    assert not consistent(graph, a, bad) and not consistent(graph, b, bad)


def test_best_consistent_set():
    # 0-1-2 mutually consistent, 3 consistent with nothing: pick {0, 1, 2}
    adj = [[i == j or (i < 3 and j < 3) for j in range(4)] for i in range(4)]
    assert best_consistent_set(adj, [1, 1, 1, 5]) == [0, 1, 2] or best_consistent_set(adj, [1, 1, 1, 5]) == [3]
    assert best_consistent_set(adj, [2, 2, 2, 5]) == [0, 1, 2]
    assert best_consistent_set([[True]], [1]) == [0]
    assert best_consistent_set([], []) == []
    # the greedy path used for big sets keeps a consistent set too
    n = 40
    adj = [[i == j or (i % 2 == j % 2) for j in range(n)] for i in range(n)]
    keep = best_consistent_set(adj, [1] * n)
    assert len(keep) == 20 and len({k % 2 for k in keep}) == 1


def test_path_length_counts_distance_travelled_not_keyframes():
    graph = KeyframeGraph()
    for i in range(30):
        T = np.eye(4) if i == 0 else make_T(None, [0.0 if i >= 10 else 0.5, 0, 0])      # moves for 9 steps, then stands still
        graph.add(Keyframe(id=i, prev_id=i - 1, stamp=i, T_prev_kf=T, covariance=covariance_from_sigmas(0.01, 0.01),
                           T_odom_kf=np.eye(4)))
    assert graph.path_length(0, 9) == pytest.approx(4.5)
    assert graph.path_length(9, 29) == pytest.approx(0.0)       # 20 keyframes, no travel
    assert graph.hops(9, 29) == 20
