import numpy as np
import pytest

from terrain_core.traversability import (
    RobotLimits,
    compute_roughness,
    compute_slope,
    compute_step,
    plane_fit,
    compute_traversability,
)

RES = 0.05  # 5 cm cells
N = 60


def flat(h=0.0):
    return np.full((N, N), h)


def ramp(deg):
    """Ground rising along +col at the given angle."""
    cols = np.arange(N) * RES
    return np.tile(np.tan(np.deg2rad(deg)) * cols, (N, 1))


def test_flat_floor_reads_flat():
    z = flat(0.2)
    out = compute_traversability(z, RES)
    interior = (slice(2, -2), slice(2, -2))
    assert np.allclose(out["slope"][interior], 0.0, atol=1e-9)
    assert np.allclose(out["step"][interior], 0.0, atol=1e-9)
    assert np.allclose(out["roughness"][interior], 0.0, atol=1e-9)
    assert out["traversable"][interior].all()
    assert np.allclose(out["traversability"][interior], 1.0)


@pytest.mark.parametrize("deg", [5, 10, 15, 25])
def test_ramp_slope_matches_angle(deg):
    slope = compute_slope(ramp(deg), RES)
    assert np.allclose(np.rad2deg(slope[2:-2, 2:-2]), deg, atol=0.01)


def test_ramp_is_not_a_step():
    z = ramp(15)
    slope, rough = plane_fit(z, RES)
    step = compute_step(z, RES, slope=slope, roughness=rough)
    assert np.nanmax(step[2:-2, 2:-2]) < 1e-9
    # Without the correction the same ramp reads as a (small) step.
    assert np.nanmax(compute_step(z, RES)[2:-2, 2:-2]) > 0.01


def test_ramp_roughness_is_zero():
    assert np.allclose(compute_roughness(ramp(20), RES)[2:-2, 2:-2], 0.0, atol=1e-9)


def test_vertical_step_height_detected():
    z = flat()
    z[:, N // 2:] = 0.15
    slope, rough = plane_fit(z, RES)
    step = compute_step(z, RES, slope=slope, roughness=rough)
    edge = step[10:-10, N // 2 - 1:N // 2 + 1]
    assert np.allclose(edge, 0.15, atol=0.01)
    assert np.allclose(step[10:-10, :N // 2 - 3], 0.0, atol=1e-9)


def test_box_is_not_traversable():
    z = flat()
    z[20:30, 20:30] = 0.25
    out = compute_traversability(z, RES, RobotLimits(max_step_m=0.10))
    assert not out["traversable"][20, 20]          # corner of the box
    assert not out["traversable"][25, 19]          # floor cell touching the box
    assert out["traversable"][45, 45]              # far-away floor
    assert out["step"][25, 19] == pytest.approx(0.25, abs=0.01)


def test_step_below_limit_is_traversable():
    z = flat()
    z[:, N // 2:] = 0.06
    out = compute_traversability(z, RES, RobotLimits(max_step_m=0.10))
    assert out["traversable"][30, N // 2]
    assert 0.0 < out["traversability"][30, N // 2] < 1.0


def test_steep_ramp_is_not_traversable():
    out = compute_traversability(ramp(30), RES, RobotLimits(max_slope_deg=20.0))
    assert not out["traversable"][2:-2, 2:-2].any()
    out = compute_traversability(ramp(10), RES, RobotLimits(max_slope_deg=20.0))
    assert out["traversable"][2:-2, 2:-2].all()


def test_roughness_from_noise():
    rng = np.random.default_rng(0)
    z = flat() + rng.normal(0, 0.02, (N, N))
    rough = compute_roughness(z, RES)[2:-2, 2:-2]
    assert 0.005 < np.mean(rough) < 0.03
    out = compute_traversability(z, RES, RobotLimits(max_roughness_m=0.005))
    assert not out["traversable"][2:-2, 2:-2].all()


def test_unknown_cells_never_traversable():
    z = flat()
    z[10:20, 10:20] = np.nan
    out = compute_traversability(z, RES)
    assert np.isnan(out["traversability"][15, 15])
    assert not out["traversable"][15, 15]
    assert np.isnan(out["slope"][15, 15]) and np.isnan(out["step"][15, 15])
    assert out["traversable"][40, 40]


def test_sparse_cells_do_not_crash_or_fabricate():
    z = np.full((N, N), np.nan)
    z[30, 30] = 0.0           # lone cell, no neighbours
    out = compute_traversability(z, RES)
    assert not out["traversable"].any()


def test_small_hole_in_floor_still_fits():
    z = flat()
    z[30, 30] = np.nan
    out = compute_traversability(z, RES)
    assert np.isnan(out["traversability"][30, 30])
    assert out["traversable"][30, 31]          # neighbour of the hole is fine


def test_variance_gate_marks_uncertain_cells_unknown():
    z = flat()
    var = np.full((N, N), 0.0001)
    var[10:20, 10:20] = 0.01
    out = compute_traversability(z, RES, RobotLimits(max_variance=0.001), variance=var)
    assert not out["traversable"][15, 15]
    assert np.isnan(out["traversability"][15, 15])
    assert out["traversable"][40, 40]


def test_outputs_have_input_shape_and_expected_dtypes():
    z = ramp(5)[:40, :50]
    out = compute_traversability(z, RES)
    for k in ("slope", "step", "roughness", "traversability", "traversable"):
        assert out[k].shape == z.shape
    assert out["traversable"].dtype == bool
