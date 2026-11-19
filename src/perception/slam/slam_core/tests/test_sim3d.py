"""The full 6-DoF synthetic world with the real GTSAM iSAM2 backend (skipped where gtsam is not installed)."""
import numpy as np
import pytest

pytest.importorskip("gtsam")

from slam_core.loop_closure import LoopClosureDetector, LoopParams, sim3d  # noqa: E402
from slam_core.types.pose import inv_T  # noqa: E402


@pytest.fixture(scope="module")
def scenario():
    return sim3d.make_scenario3d(seed=0)


def test_the_world_is_really_3d(scenario):
    z = scenario.gt[:, 2]
    assert z.max() - z.min() > 1.0                        # the path climbs
    # camera pitch and roll are not zero somewhere: poses are general SE(3)
    up = [T[:3, :3] @ np.array([0, 0, 1.0]) for T in scenario.gt_T]
    assert min(u[2] for u in up) < 0.999
    # odometry drifts vertically as well as horizontally
    err = np.array([k.T_odom_kf[:3, 3] - T[:3, 3] for k, T in zip(scenario.kfs, scenario.gt_T)])
    assert np.abs(err[:, 2]).max() > 0.3


def test_verifier_recovers_the_true_relative_pose(scenario):
    verify = sim3d.make_verifier3d(scenario, wrong_match=0.0)
    i, j = 40, 44                                        # nearby keyframes, so landmarks overlap
    v = verify(i, j)
    T_true = inv_T(scenario.gt_T[i]) @ scenario.gt_T[j]
    assert np.linalg.norm(v.T_old_new[:3, 3] - T_true[:3, 3]) < 0.1


def run(scenario, max_loop_chi2=None, **kw):
    det = LoopClosureDetector(sim3d.make_verifier3d(scenario), LoopParams(min_inliers=10, **kw))
    return sim3d.run3d(scenario, det, max_loop_chi2)


def test_full_detector_in_3d_has_no_false_loops_and_reduces_error(scenario):
    r = run(scenario)
    assert r["n_false"] == 0
    assert r["n_long"] >= 10
    assert r["ate_opt"] < 0.3 * r["ate_raw"]


def test_appearance_only_is_fooled_in_3d_too(scenario):
    # No detector defenses and the backend's own check switched off: the identical hallways get in
    r = run(scenario, max_loop_chi2=np.inf, gate=False, confirm_count=1, pcm=False)
    assert r["n_false"] >= 1      # (the error still ends below the raw odometry, because the true loops help a lot)


def test_the_backends_own_check_also_catches_the_false_loops(scenario):
    # Same undefended detector, but the iSAM2 backend rejects loops the rest of the graph cannot live with
    r = run(scenario, gate=False, confirm_count=1, pcm=False)
    assert r["n_false"] == 0
