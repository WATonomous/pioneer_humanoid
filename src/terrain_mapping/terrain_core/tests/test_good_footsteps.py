"""Tests for terrain_core.traversability.policy_scan (exact-cell checks)."""
import numpy as np
import pytest

from terrain_core.traversability.good_footsteps import (
    foothold_layer, foothold_obs, make_policy_scan,
)

ORIGIN = (-10.0, -10.0)


def coded_map(res, size_m=20.0):
    n = int(round(size_m / res))
    rows, cols = np.indices((n, n))
    return (rows * 1000 + cols).astype(float)       # value identifies (row, col)


def all_true(elev):
    return {"traversable": np.ones(elev.shape, bool)}


def coded(row, col):
    return -(row * 1000 + col) - 0.5                 # base_z = 0


def coded_scan():
    iy, ix = np.indices((21, 33))
    return (iy * 100 + ix).astype(np.float32)


CELLS = [(0, 0), (32, 0), (0, 20), (32, 20), (5, 7)]   # (ix, iy)


@pytest.mark.parametrize("yaw,rc", [
    (0.0, lambda ix, iy: (190 + iy, 184 + ix)),
    (np.pi / 2, lambda ix, iy: (184 + ix, 210 - iy)),
    (-np.pi / 2, lambda ix, iy: (216 - ix, 190 + iy)),
])
def test_yaw_hits_exact_cells(yaw, rc):
    e = coded_map(0.05)
    out = make_policy_scan(e, 0.05, ORIGIN, (0.0, 0.0), 0.0, yaw, all_true(e))
    assert out["height_scan"].shape == (21, 33) and out["height_scan"].dtype == np.float32
    for ix, iy in CELLS:
        assert out["height_scan"][iy, ix] == coded(*rc(ix, iy))


@pytest.mark.parametrize("res,row,col", [(0.05, 196, 188), (0.04, 245, 235)])
def test_resolutions(res, row, col):
    e = coded_map(res)
    out = make_policy_scan(e, res, ORIGIN, (0.0, 0.0), 0.0, 0.0, all_true(e))
    assert out["height_scan"][6, 4] == coded(row, col)       # ix=4, iy=6


def test_left_is_plus_y():
    e = np.zeros((400, 400))
    e[200:, :] = 1.0                                          # world y >= 0 is higher
    out = make_policy_scan(e, 0.05, ORIGIN, (0.0, 0.0), 0.0, 0.0, all_true(e))
    h = out["height_scan"]
    assert np.all(h[:10, :] == -0.5)                          # iy <= 9: y < 0
    assert np.all(h[11:, :] == -1.5)                          # iy >= 11: y > 0


def test_height_convention_flat():
    e = np.zeros((400, 400))
    out = make_policy_scan(e, 0.05, ORIGIN, (0.0, 0.0), 0.7, 0.0, all_true(e))
    assert np.allclose(out["height_scan"], 0.2) and out["known"].all()


def test_flatten_order():
    e = coded_map(0.05)
    h = make_policy_scan(e, 0.05, ORIGIN, (0.0, 0.0), 0.0, 0.0, all_true(e))["height_scan"]
    assert h.reshape(-1)[7 * 33 + 5] == h[7, 5]


def test_all_nan_and_off_map():
    nan_map = np.full((100, 100), np.nan)
    out = make_policy_scan(nan_map, 0.05, (0.0, 0.0), (2.5, 2.5), 0.0, 0.0,
                           {"traversable": np.ones((100, 100), bool)})
    assert np.isnan(out["height_scan"]).all() and not out["known"].any()
    assert not out["foothold"].any() and (out["foothold_score"] == 0).all()
    e = np.zeros((100, 100))
    far = make_policy_scan(e, 0.05, (0.0, 0.0), (100.0, 100.0), 0.0, 0.0, all_true(e))
    assert np.isnan(far["height_scan"]).all() and not far["foothold"].any()


def test_partial_overlap_known_mask():
    e = np.zeros((400, 400))
    out = make_policy_scan(e, 0.05, ORIGIN, (9.5, 0.0), 0.0, 0.0, all_true(e))
    assert out["known"][:, :21].all() and not out["known"][:, 27:].any()


def test_foothold_all_true_all_false_and_no_crop():
    good, score = foothold_layer(np.ones((20, 20), bool), 0.05)
    assert good.sum() == 16 * 16 and good[2:18, 2:18].all()
    good, score = foothold_layer(np.zeros((20, 20), bool), 0.05)
    assert not good.any() and (score == 0).all()
    e = np.zeros((100, 100))                                  # scan edge must not look like an obstacle
    out = make_policy_scan(e, 0.05, (0.0, 0.0), (2.5, 2.5), 0.0, 0.0, all_true(e))
    assert out["foothold"].all()


def test_foothold_clearance_and_score():
    ok = np.ones((41, 41), bool)
    ok[20, 20] = False
    good, score = foothold_layer(ok, 0.05)
    assert good[20, 23] and not good[20, 22] and not good[22, 22]   # 3.0 ok, 2.83 not
    clean, s2 = foothold_layer(np.ones((41, 41), bool), 0.05)
    assert s2[20, 20] == 1.0 and s2[20, 3] == pytest.approx(0.2 / 0.3, abs=1e-6)
    capped = np.full((41, 41), 0.5)
    capped[20, 20] = np.nan
    _, s3 = foothold_layer(np.ones((41, 41), bool), 0.05, capped)
    assert s3.max() == pytest.approx(0.5) and s3[20, 20] == 0.0


@pytest.mark.parametrize("res", [0.04, 0.05])
def test_foothold_threshold_resolution_safe(res):
    good, _ = foothold_layer(np.ones((30, 30), bool), res)
    k = int(np.ceil(0.15 / res - 1e-6))                       # cells needed from the edge
    assert good[k - 1, 15] and not good[k - 2, 15]


def test_obs_patch_order_length_and_values():
    obs, frac = foothold_obs(coded_scan(), [(0.0, 0.0), (0.0, -0.15)])
    assert obs.shape == (102,) and obs.dtype == np.float32 and (frac == 0).all()
    assert tuple(obs[:2]) == (0.0, 0.0) and obs[51] == 0.0 and obs[52] == pytest.approx(-0.15)
    p0 = obs[2:51].reshape(7, 7)                              # [x offset, y offset]
    p1 = obs[53:].reshape(7, 7)
    for i in range(7):
        for j in range(7):
            assert p0[i, j] == (7 + j) * 100 + (13 + i)       # anchor (16, 10): x slow, y fast
            assert p1[i, j] == (4 + j) * 100 + (13 + i)       # anchor (16, 7)


def test_obs_border_clamp():
    obs, _ = foothold_obs(coded_scan(), [(-0.8, -0.5), (5.0, 5.0)])
    p0 = obs[2:51].reshape(7, 7)
    for i in range(7):
        for j in range(7):
            assert p0[i, j] == max(0, -3 + j) * 100 + max(0, -3 + i)
    assert (obs[53:] == 20 * 100 + 32).all()


def test_obs_nan_fill_and_errors():
    s = np.full((21, 33), 2.0, np.float32)
    s[13, 16] = np.nan                                        # in foot-0 patch only
    obs, frac = foothold_obs(s, [(0.0, 0.0), (0.0, -0.15)])
    assert (obs[2:51] == 2.0).all()
    assert frac[0] == pytest.approx(1 / 49, abs=1e-6) and frac[1] == 0
    s = np.full((21, 33), 3.0, np.float32)
    s[7:14, 13:20] = np.nan                                   # whole foot-0 patch
    obs, frac = foothold_obs(s, [(0.0, 0.0), (0.0, -0.15)])
    assert (obs[2:51] == 3.0).all() and frac[0] == 1.0
    obs, _ = foothold_obs(np.full((21, 33), np.nan), [(0.0, 0.0), (0.0, 0.1)], fallback=9.0)
    assert (obs[2:51] == 9.0).all()
    with pytest.raises(ValueError):
        foothold_obs(np.zeros((33, 21)), [(0.0, 0.0), (0.0, 0.1)])