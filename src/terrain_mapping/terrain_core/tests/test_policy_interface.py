import numpy as np
from terrain_core.traversability.policy_interface import build_height_scan, foothold_observation  # adjust


def _coded_scan():
    ys, xs = np.meshgrid(np.arange(21), np.arange(33), indexing="ij")
    return (ys * 100 + xs).astype(float)                 # value encodes (iy, ix)


def test_observation_length_and_patch_order():
    obs = foothold_observation(_coded_scan(), np.array([[0.0, 0.0], [0.0, 0.0]]))
    assert obs.shape == (102,)
    # foothold at (0, 0) -> anchor ix=16, iy=10; patch element (dx, dy) = (-3, -3) first
    assert obs[2] == 7 * 100 + 13           # iy=7,  ix=13
    assert obs[3] == 8 * 100 + 13           # y changes fastest within the patch
    assert obs[2 + 7] == 7 * 100 + 14       # next x column


def test_clamp_at_border():
    obs = foothold_observation(_coded_scan(), np.array([[0.8, 0.5], [0.0, 0.0]]))
    patch = obs[2:51]
    assert patch.max() == 20 * 100 + 32     # clamped to the last cell


def test_nan_fill_uses_patch_median():
    scan = np.full((21, 33), 1.0)
    scan[10, 16] = np.nan
    obs = foothold_observation(scan, np.array([[0.0, 0.0], [0.0, 0.0]]))
    assert np.isfinite(obs).all() and (obs[2:51] == 1.0).all()


def test_build_scan_flat_and_off_map():
    res = 0.05
    elev = np.full((200, 200), -1.0)
    w2rc = lambda x, y: (int(round(y / res)), int(round(x / res)))
    scan = build_height_scan(elev, w2rc, (5.0, 5.0), 0.0, base_z=0.0)
    assert scan.shape == (21, 33)
    assert np.allclose(scan, 0.0 - (-1.0) - 0.5)
    off = build_height_scan(elev, w2rc, (-50.0, 5.0), 0.0, base_z=0.0)
    assert np.isnan(off).all()


def test_build_scan_yaw_turns_forward_axis():
    res = 0.05
    elev = np.zeros((200, 200))
    elev[:, :] = 0.0
    elev[(5.4 / res).__int__():, :] = 0.2                 # raised ground for world y > ~5.4
    w2rc = lambda x, y: (int(round(y / res)), int(round(x / res)))
    scan = build_height_scan(elev, w2rc, (5.0, 5.0), np.pi / 2, base_z=0.0)
    # yaw 90 deg: base +x (forward) points along world +y, so the raised
    # ground appears at large x_s (right-hand columns of the scan)
    assert scan[10, -1] < scan[10, 0]                     # raised ground -> smaller height value