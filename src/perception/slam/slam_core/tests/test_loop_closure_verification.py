"""PnPVerifier: the image-free loop verifier (3D points of the old keyframe + their pixels in the new one -> relative pose)."""
import numpy as np
import pytest

pytest.importorskip("cv2")

from slam_core.loop_closure import PnPVerifier  # noqa: E402
from slam_core.loop_closure.se3 import rot_from_vec  # noqa: E402
from slam_core.types.pose import inv_T, make_T  # noqa: E402

K = np.array([[520.0, 0, 320.0], [0, 520.0, 240.0], [0, 0, 1]])
SIZE = (640, 480)


# Points seen by the new camera at known pixels, expressed in the old camera's frame, plus their (noisy) new pixels
def scene(T_old_new, n=200, outliers=0.1, seed=0):
    rng = np.random.default_rng(seed)
    px = np.column_stack([rng.uniform(20, 620, n), rng.uniform(20, 460, n)])
    z = rng.uniform(1.0, 4.0, n)
    X_new = np.column_stack([(px[:, 0] - K[0, 2]) / K[0, 0] * z, (px[:, 1] - K[1, 2]) / K[1, 1] * z, z])
    X_old = X_new @ T_old_new[:3, :3].T + T_old_new[:3, 3]
    px = px + rng.normal(0, 0.5, px.shape)
    bad = rng.random(n) < outliers
    px[bad] = np.column_stack([rng.uniform(0, 640, bad.sum()), rng.uniform(0, 480, bad.sum())])
    return X_old, px


def test_recovers_the_relative_pose_with_outliers():
    T_true = make_T(rot_from_vec([0.02, 0.15, -0.03]), [0.4, -0.05, 0.2])      # a loop-sized motion, beyond tracking limits
    X, px = scene(T_true)
    v = PnPVerifier(lambda o, n: (X, px), K, SIZE)(0, 1)
    assert v is not None
    E = inv_T(T_true) @ v.T_old_new
    assert np.linalg.norm(E[:3, 3]) < 0.02
    assert v.n_inliers >= 150
    assert np.all(np.diag(v.covariance) > 0)


def test_rejects_matches_that_agree_on_nothing():
    rng = np.random.default_rng(1)
    X = rng.uniform([-2, -2, 1], [2, 2, 4], (200, 3))
    px = np.column_stack([rng.uniform(0, 640, 200), rng.uniform(0, 480, 200)])
    assert PnPVerifier(lambda o, n: (X, px), K, SIZE)(0, 1) is None


def test_no_matches_means_no_loop():
    assert PnPVerifier(lambda o, n: (np.zeros((0, 3)), np.zeros((0, 2))), K, SIZE)(0, 1) is None
