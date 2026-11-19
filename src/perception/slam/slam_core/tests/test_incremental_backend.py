"""The iSAM2 backend (needs gtsam, so these are skipped where it is not installed)."""
import numpy as np
import pytest

pytest.importorskip("gtsam")

from slam_core.backend.incremental import IncrementalBackend  # noqa: E402
from slam_core.loop_closure import LoopClosureDetector, LoopConstraint, LoopParams, LoopUpdate, sim  # noqa: E402
from slam_core.types import Keyframe  # noqa: E402
from slam_core.types.pose import covariance_from_sigmas, make_T  # noqa: E402


def straight_line(n=20):
    be = IncrementalBackend()
    for i in range(n):
        T = np.eye(4) if i == 0 else make_T(None, [1.0, 0, 0])
        be.add_keyframe(Keyframe(id=i, prev_id=i - 1, stamp=i, T_prev_kf=T, covariance=covariance_from_sigmas(0.01, 0.05),
                                 T_odom_kf=np.eye(4)))
    return be


def loop(id, old, new, x):
    return LoopConstraint(id, old, new, make_T(None, [x, 0, 0]), covariance_from_sigmas(0.01, 0.05), 50, 0.9)


def test_odometry_only_follows_the_measurements():
    be = straight_line()
    assert np.allclose(be.pose(19)[:3, 3], [19, 0, 0], atol=1e-6)


def test_a_loop_pulls_the_trajectory_and_retracting_it_puts_it_back():
    be = straight_line()
    L = loop(1, 0, 19, 18.9)        # says keyframe 19 is 18.9 m from keyframe 0, odometry says 19
    assert be.apply(LoopUpdate(added=[L])) == []
    assert be.pose(19)[0, 3] < 18.99
    be.apply(LoopUpdate(retracted=[1]))
    assert be.pose(19)[0, 3] == pytest.approx(19.0, abs=1e-3)


def test_a_loop_the_graph_cannot_live_with_is_rejected_after_optimizing():
    be = straight_line()
    wild = loop(2, 0, 19, 5.0)      # 14 m off: far outside what the odometry covariance allows
    assert be.apply(LoopUpdate(added=[wild])) == [2]
    assert be.pose(19)[0, 3] == pytest.approx(19.0, abs=1e-3)


def test_detector_plus_backend_on_the_simulator():
    sc = sim.make_scenario(seed=0, bias=0.003, rot_sigma=0.02)
    det = LoopClosureDetector(sim.make_verifier(sc), LoopParams(min_inliers=10))
    r = sim.run(sc, det, backend="isam2")
    assert r["n_false"] == 0
    assert r["ate_opt"] < 0.3 * r["ate_raw"]
