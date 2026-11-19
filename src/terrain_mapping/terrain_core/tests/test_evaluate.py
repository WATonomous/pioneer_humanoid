"""Tests for terrain_tools.evaluate and terrain_tools.synthetic_maps (TR-11)."""
import numpy as np
import pytest
import yaml

from check_map import save_map_npz
from terrain_tools.evaluate import DEFAULT_CONFIG, evaluate, main
from terrain_tools.synthetic_maps import GridMap, build_course_map, build_synthetic_suite


@pytest.fixture(scope="module")
def course():
    return yaml.safe_load(DEFAULT_CONFIG.read_text())


def test_synthetic_suite_verdicts(course):
    suite = build_synthetic_suite(course)
    assert set(suite) == {
        "perfect_course",
        "floor_shifted_up",
        "fake_blob_on_floor",
        "hole_under_feet",
    }

    # 1. Perfect map passes every check
    res_perfect = evaluate(suite["perfect_course"], course)
    assert all(m["pass"] for m in res_perfect.values())

    # 2. Floor shifted up (+5 cm) fails height checks on floor, step, ramp, and box,
    # while under-feet coverage passes and ramp slope angle stays ~9 deg
    res_shift = evaluate(suite["floor_shifted_up"], course)
    assert res_shift["under_feet"]["pass"] is True
    assert res_shift["flat_floor"]["pass"] is False
    assert res_shift["step"]["pass"] is False
    assert res_shift["ramp"]["pass"] is False
    assert res_shift["obstacle_box"]["pass"] is False
    assert abs(res_shift["ramp"]["slope_error_deg"]) <= course["regions"]["ramp"]["slope_tolerance_deg"]

    # 3. Fake blob on floor fails only flat_floor (false obstacles)
    res_blob = evaluate(suite["fake_blob_on_floor"], course)
    assert res_blob["flat_floor"]["pass"] is False
    assert res_blob["flat_floor"]["false_obstacle_cells"] > 0
    assert res_blob["under_feet"]["pass"] is True
    assert res_blob["step"]["pass"] is True
    assert res_blob["ramp"]["pass"] is True
    assert res_blob["obstacle_box"]["pass"] is True

    # 4. Hole under feet fails only under_feet coverage
    res_hole = evaluate(suite["hole_under_feet"], course)
    assert res_hole["under_feet"]["pass"] is False
    assert res_hole["under_feet"]["coverage"] == 0.0
    assert res_hole["flat_floor"]["pass"] is True
    assert res_hole["step"]["pass"] is True
    assert res_hole["ramp"]["pass"] is True
    assert res_hole["obstacle_box"]["pass"] is True


def test_check_map_npz_load_and_evaluate(course, tmp_path):
    """Verify GridMap.load() converts grid_map index order, shifts odom z, and centers on map_centre."""
    gm = build_course_map(course)
    cam_h = 1.58
    # Convert from GridMap (row=+y_course, col=+x_course, floor=0) into check_map.py's raw grid_map odom export:
    # row 0 = +x_odom (forward), col 0 = +y_odom (left), and heights in odom (floor ~ -cam_h).
    raw_grid_map = (gm.z - cam_h)[::-1, :].astype(np.float32)

    npz_path = tmp_path / "elevation.npz"
    save_map_npz(
        str(npz_path),
        heights=raw_grid_map,
        resolution=gm.res,
        map_centre=[2.2, 0.0],
        frame="odom",
        camera_z=0.0,
    )

    loaded = GridMap.load(npz_path)
    assert loaded.frame == "odom"
    assert loaded.camera_height == pytest.approx(cam_h, abs=0.01)
    assert np.allclose(loaded.map_centre, [0.0, 2.2])
    assert np.allclose(loaded.origin, [-2.0, -0.8])

    res = evaluate(loaded, course)
    assert all(m["pass"] for m in res.values())
    assert main([str(npz_path), "--strict"]) == 0
