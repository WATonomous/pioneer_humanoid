"""Tests for terrain_core.traversability.

Clean-surface tests check the geometry. The noisy-map tests matter more: real elevation maps carry 1-2 cm of noise
per cell, and the layers must still call a noisy flat floor and a 5 degree ramp traversable while blocking a 10 degree
ramp and a 10 cm step. They run at 5 cm and 4 cm cells (ETH's map) to check that window sizes in metres hold up.
"""
import numpy as np
import pytest

from terrain_core.traversability import (
    RobotLimits,
    TraversabilityParams,
    compute_roughness,
    compute_slope,
    compute_step,
    compute_traversability,
    estimate_noise_sigma,
    plane_fit,
    planar_tolerance,
)
from terrain_core.traversability._window import window_radius
from terrain_core.traversability.step import step_radius_cells

RES = 0.05  # default test resolution: 5 cm cells
SIZE_M = 5.0  # test terrains are 5 m x 5 m


def grid_size(res):
    return int(round(SIZE_M / res))


def inner(a, res, margin_m=0.5):
    """Drop a border so edge effects of the windows do not count."""
    m = int(round(margin_m / res))
    return a[m:-m, m:-m]


def flat(res=RES, h=0.0):
    n = grid_size(res)
    return np.full((n, n), h)


def ramp(deg, res=RES):
    """Ground rising along +col at the given angle."""
    n = grid_size(res)
    cols = np.arange(n) * res
    return np.tile(np.tan(np.deg2rad(deg)) * cols, (n, 1))


def step_terrain(height, res=RES):
    """Flat floor with a vertical step of `height` metres across the middle (columns >= n//2). Returns (z, edge_col)."""
    z = flat(res)
    edge = z.shape[1] // 2
    z[:, edge:] = height
    return z, edge


def noisy(z, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return z + rng.normal(0.0, sigma, z.shape)


def edge_band(a, res, edge):
    """The two columns either side of a step edge, away from the borders."""
    m = int(round(0.5 / res))
    return a[m:-m, edge - 1:edge + 1]


# --------------------------------------------------------------------------------------------- window sizes in metres

@pytest.mark.parametrize("res,expected", [(0.05, 4), (0.04, 5), (0.025, 8), (0.10, 2)])
def test_window_radius_in_metres(res, expected):
    assert window_radius(0.40, res) == expected


def test_window_never_smaller_than_3x3():
    assert window_radius(0.01, 0.05) == 1
    assert step_radius_cells(0.01, 0.05) == 1


def test_step_radius_in_metres():
    assert step_radius_cells(0.05, 0.05) == 1
    assert step_radius_cells(0.05, 0.04) == 1
    assert step_radius_cells(0.05, 0.02) == 3


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        window_radius(0.0, 0.05)
    with pytest.raises(ValueError):
        step_radius_cells(0.05, 0.0)


@pytest.mark.parametrize("deg", [5, 10, 20])
def test_slope_is_resolution_independent(deg):
    s5 = np.rad2deg(inner(compute_slope(ramp(deg, 0.05), 0.05), 0.05))
    s4 = np.rad2deg(inner(compute_slope(ramp(deg, 0.04), 0.04), 0.04))
    assert np.allclose(s5, deg, atol=0.01)
    assert np.allclose(s4, deg, atol=0.01)


# ----------------------------------------------------------------------------------------------------- clean surfaces

def test_flat_floor_reads_flat():
    out = compute_traversability(flat(h=0.2), RES)
    for key in ("slope", "step", "roughness"):
        assert np.allclose(inner(out[key], RES), 0.0, atol=1e-9)
    assert inner(out["traversable"], RES).all()
    assert np.allclose(inner(out["traversability"], RES), 1.0)


@pytest.mark.parametrize("deg", [5, 10, 15, 25])
def test_ramp_slope_matches_angle(deg):
    slope = compute_slope(ramp(deg), RES)
    assert np.allclose(np.rad2deg(inner(slope, RES)), deg, atol=0.01)


def test_ramp_is_not_a_step():
    z = ramp(15)
    slope, rough = plane_fit(z, RES)
    planar = rough <= 0.01
    step = compute_step(z, RES, slope=slope, planar=planar)
    assert np.nanmax(inner(step, RES)) < 1e-9
    # Without the correction the same ramp reads as a (small) step.
    assert np.nanmax(inner(compute_step(z, RES), RES)) > 0.01


def test_ramp_roughness_is_zero():
    # Round-off from the sum-of-squares form leaves ~1 nm; anything under a micrometre is zero.
    assert np.allclose(inner(compute_roughness(ramp(20), RES), RES), 0.0, atol=1e-6)


def test_vertical_step_height_detected():
    z, edge = step_terrain(0.15)
    slope, rough = plane_fit(z, RES)
    step = compute_step(z, RES, slope=slope, planar=rough <= 0.01)
    assert np.allclose(edge_band(step, RES, edge), 0.15, atol=0.01)
    far = inner(step, RES)[:, :edge - int(round(0.5 / RES)) - 3]
    assert np.allclose(far, 0.0, atol=1e-9)


def test_box_is_not_traversable():
    z = flat()
    z[40:50, 40:50] = 0.25
    out = compute_traversability(z, RES, RobotLimits(max_step_m=0.10))
    assert not out["traversable"][40, 40]          # corner of the box
    assert not out["traversable"][45, 39]          # floor cell touching the box
    assert out["traversable"][15, 15]              # far-away floor
    assert out["step"][45, 39] == pytest.approx(0.25, abs=0.01)


def test_step_below_limit_is_traversable():
    z, edge = step_terrain(0.06)
    out = compute_traversability(z, RES, RobotLimits(max_step_m=0.10))
    band = edge_band(out["traversable"], RES, edge)
    assert band.all()
    assert 0.0 < edge_band(out["traversability"], RES, edge).max() < 1.0


def test_steep_ramp_is_not_traversable():
    out = compute_traversability(ramp(30), RES, RobotLimits(max_slope_deg=20.0))
    assert not inner(out["traversable"], RES).any()
    out = compute_traversability(ramp(10), RES, RobotLimits(max_slope_deg=20.0))
    assert inner(out["traversable"], RES).all()


def test_unknown_cells_never_traversable():
    z = flat()
    z[40:50, 40:50] = np.nan
    out = compute_traversability(z, RES)
    assert np.isnan(out["traversability"][45, 45])
    assert not out["traversable"][45, 45]
    assert np.isnan(out["slope"][45, 45]) and np.isnan(out["step"][45, 45])
    assert out["traversable"][15, 15]


def test_sparse_cells_do_not_crash_or_fabricate():
    z = np.full((60, 60), np.nan)
    z[30, 30] = 0.0           # lone cell, no neighbours
    out = compute_traversability(z, RES)
    assert not out["traversable"].any()


def test_small_hole_in_floor_still_fits():
    z = flat()
    z[40, 40] = np.nan
    out = compute_traversability(z, RES)
    assert np.isnan(out["traversability"][40, 40])
    assert out["traversable"][40, 41]          # neighbour of the hole is fine


def test_variance_gate_marks_uncertain_cells_unknown():
    z = flat()
    var = np.full(z.shape, 0.0001)
    var[40:50, 40:50] = 0.01
    out = compute_traversability(z, RES, RobotLimits(max_variance=0.001), variance=var)
    assert not out["traversable"][45, 45]
    assert np.isnan(out["traversability"][45, 45])
    assert out["traversable"][15, 15]


def test_outputs_have_input_shape_and_expected_dtypes():
    z = ramp(5)[:60, :70]
    out = compute_traversability(z, RES)
    for k in ("slope", "step", "roughness", "traversability", "traversable"):
        assert out[k].shape == z.shape
    assert out["traversable"].dtype == bool


# --------------------------------------------------------------------------------------------------------- noisy maps

@pytest.mark.parametrize("res", [0.05, 0.04])
@pytest.mark.parametrize("sigma,min_frac", [(0.01, 0.99), (0.02, 0.95)])
def test_noisy_flat_floor_is_traversable(res, sigma, min_frac):
    out = compute_traversability(noisy(flat(res), sigma), res)
    assert inner(out["traversable"], res).mean() >= min_frac


@pytest.mark.parametrize("res", [0.05, 0.04])
@pytest.mark.parametrize("sigma,min_frac", [(0.01, 0.98), (0.02, 0.90)])
def test_noisy_5_degree_ramp_passes(res, sigma, min_frac):
    out = compute_traversability(noisy(ramp(5, res), sigma), res)
    assert inner(out["traversable"], res).mean() >= min_frac


@pytest.mark.parametrize("res", [0.05, 0.04])
@pytest.mark.parametrize("sigma", [0.01, 0.02])
def test_noisy_10_degree_ramp_is_blocked(res, sigma):
    out = compute_traversability(noisy(ramp(10, res), sigma), res)
    assert inner(out["traversable"], res).mean() <= 0.05


@pytest.mark.parametrize("res", [0.05, 0.04])
@pytest.mark.parametrize("sigma", [0.01, 0.02])
def test_noisy_10cm_step_is_blocked(res, sigma):
    z, edge = step_terrain(0.10, res)
    out = compute_traversability(noisy(z, sigma), res)
    assert edge_band(out["traversable"], res, edge).mean() <= 0.05
    # floor well away from the edge is unaffected
    far = out["traversable"][int(0.5 / res):-int(0.5 / res), int(0.5 / res):edge - int(0.8 / res)]
    assert far.mean() >= 0.95


@pytest.mark.parametrize("res", [0.05, 0.04])
def test_legal_6cm_step_still_passes_with_small_noise(res):
    z, edge = step_terrain(0.06, res)
    out = compute_traversability(noisy(z, 0.005), res)
    assert edge_band(out["traversable"], res, edge).mean() >= 0.9


def test_noisy_ramp_is_not_read_as_stairs():
    out = compute_traversability(noisy(ramp(5), 0.01), RES)
    # A 5 degree slope over 5 cm is under 0.5 cm of rise; the step layer should sit at the noise level, not near 8 cm.
    assert np.nanpercentile(inner(out["step"], RES), 99) < 0.05


# --------------------------------------------------------------------------------------------------- noise estimation

@pytest.mark.parametrize("sigma", [0.005, 0.01, 0.02])
def test_noise_estimate_recovers_sigma(sigma):
    z = noisy(flat(), sigma)
    _, rough = plane_fit(z, RES)
    cells = (2 * window_radius(0.40, RES) + 1) ** 2
    assert estimate_noise_sigma(rough, cells) == pytest.approx(sigma, rel=0.15)


def test_noise_estimate_survives_obstacles():
    z = noisy(flat(), 0.01)
    z[20:40, 20:40] = 0.30
    z[60:70, 10:80] = 0.15
    _, rough = plane_fit(z, RES)
    cells = (2 * window_radius(0.40, RES) + 1) ** 2
    assert estimate_noise_sigma(rough, cells) == pytest.approx(0.01, rel=0.25)


def test_noise_estimate_of_empty_map_is_zero():
    assert estimate_noise_sigma(np.full((5, 5), np.nan), 25) == 0.0


def test_planar_tolerance_never_below_floor():
    assert planar_tolerance(0.0, floor_m=0.01) == 0.01
    assert planar_tolerance(0.02, floor_m=0.01, noise_factor=1.5) == pytest.approx(0.03)


def test_explicit_noise_sigma_overrides_estimate():
    z = noisy(flat(), 0.01)
    est = compute_traversability(z, RES)["traversable"]
    given = compute_traversability(z, RES, params=TraversabilityParams(noise_sigma_m=0.01))["traversable"]
    assert inner(est, RES).mean() >= 0.99 and inner(given, RES).mean() >= 0.99


def test_plane_window_is_configurable_in_metres():
    z = noisy(ramp(5), 0.02)
    small = compute_traversability(z, RES, params=TraversabilityParams(plane_window_m=0.20))
    large = compute_traversability(z, RES, params=TraversabilityParams(plane_window_m=0.40))
    # The wider window averages out more noise, so the slope estimate is closer to the true 5 degrees.
    err_small = np.nanstd(np.rad2deg(inner(small["slope"], RES)))
    err_large = np.nanstd(np.rad2deg(inner(large["slope"], RES)))
    assert err_large < err_small
