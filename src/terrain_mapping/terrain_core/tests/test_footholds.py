
import numpy as np
import pytest

from terrain_core.traversability.footholds import compute_footholds


def test_all_traversable_only_border_excluded():
    mask = np.ones((100, 100), dtype=bool)
    out = compute_footholds(mask, 0.05)
    assert out["foothold"][10:-10, 10:-10].all()
    assert not out["foothold"][0, :].any()
    assert not out["foothold"][:, -1].any()


def test_single_blocked_cell_clears_a_foot_sized_hole():
    mask = np.ones((101, 101), dtype=bool)
    mask[50, 50] = False
    good = compute_footholds(mask, 0.05)["foothold"]
    assert not good[50, 50]
    assert not good[50, 52]
    assert good[50, 56]


def test_all_blocked_gives_no_footholds():
    out = compute_footholds(np.zeros((30, 30), dtype=bool), 0.05)
    assert not out["foothold"].any()
    assert (out["foothold_score"] == 0).all()


def test_narrow_strip_has_no_footholds_wide_strip_does():
    narrow = np.zeros((60, 200), dtype=bool)
    narrow[28:32, :] = True
    assert not compute_footholds(narrow, 0.05)["foothold"].any()

    wide = np.zeros((60, 200), dtype=bool)
    wide[26:34, :] = True
    assert compute_footholds(wide, 0.05)["foothold"].any()


def _scene_good_fraction(res):
    n = int(round(4.0 / res))
    b = int(round(0.5 / res))
    mask = np.ones((n, n), dtype=bool)
    c = n // 2
    mask[c - b // 2:c + b // 2, c - b // 2:c + b // 2] = False
    return compute_footholds(mask, res)["foothold"].mean()


def test_resolution_independence():
    assert abs(_scene_good_fraction(0.04) - _scene_good_fraction(0.05)) < 0.03


def test_score_properties():
    rng = np.random.default_rng(0)
    mask = rng.random((80, 80)) > 0.1
    trav = rng.random((80, 80)).astype(np.float32)
    trav[rng.random((80, 80)) > 0.95] = np.nan
    out = compute_footholds(mask, 0.04, traversability=trav)
    s = out["foothold_score"]
    assert out["foothold"].dtype == bool
    assert s.dtype == np.float32
    assert s.shape == mask.shape
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert (s[~mask] == 0).all()
    assert (s <= np.nan_to_num(trav, nan=0.0) + 1e-6).all()
