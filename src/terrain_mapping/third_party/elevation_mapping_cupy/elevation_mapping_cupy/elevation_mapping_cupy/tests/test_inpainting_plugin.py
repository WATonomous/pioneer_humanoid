import cupy as cp
import numpy as np
from pathlib import Path

from elevation_mapping_cupy.plugins.inpainting import Inpainting


_DATA_DIR = Path(__file__).resolve().parent / "data"
_LAYER_NAMES = [
    "elevation",
    "variance",
    "is_valid",
    "traversability",
    "time",
    "upper_bound",
    "is_upper_bound",
]


def _make_elevation_map(size: int = 9):
    yy, xx = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing="ij")
    elevation = xx + 2.0 * yy
    valid = np.ones((size, size), dtype=np.float32)
    elevation_map = cp.zeros((7, size, size), dtype=cp.float32)
    elevation_map[0] = cp.asarray(elevation)
    elevation_map[2] = cp.asarray(valid)
    return elevation_map


def _make_snapshot_elevation_map():
    snapshot = np.load(_DATA_DIR / "mole_elevation_snapshot.npz")
    elevation = snapshot["elevation"].astype(np.float32, copy=False)
    valid = np.isfinite(elevation).astype(np.float32)
    elevation_map = cp.zeros((7, elevation.shape[0], elevation.shape[1]), dtype=cp.float32)
    elevation_map[0] = cp.asarray(elevation)
    elevation_map[2] = cp.asarray(valid)
    return elevation, elevation_map


def _run_inpainting(plugin: Inpainting, elevation_map: cp.ndarray) -> cp.ndarray:
    plugin_layers = cp.zeros((0, elevation_map.shape[1], elevation_map.shape[2]), dtype=cp.float32)
    return cp.asnumpy(plugin(elevation_map, _LAYER_NAMES, plugin_layers, []))


def test_inpainting_only_fills_small_holes():
    elevation_map = _make_elevation_map()
    elevation_map[0, 2, 2] = cp.nan
    elevation_map[2, 2, 2] = 0.0
    elevation_map[0, 5:8, 5:8] = cp.nan
    elevation_map[2, 5:8, 5:8] = 0.0

    plugin = Inpainting(max_hole_area=4)
    result = _run_inpainting(plugin, elevation_map)

    assert np.isfinite(result[2, 2])
    assert np.isnan(result[5:8, 5:8]).all()
    assert np.isclose(result[1, 1], 3.0)
    assert np.isclose(result[4, 4], 12.0)


def test_inpainting_flat_terrain_does_not_broadcast_large_invalid_regions():
    size = 9
    elevation_map = cp.zeros((7, size, size), dtype=cp.float32)
    elevation_map[0] = 1.5
    elevation_map[2] = 1.0

    elevation_map[0, 3, 3] = cp.nan
    elevation_map[2, 3, 3] = 0.0
    elevation_map[0, 0:3, 5:8] = cp.nan
    elevation_map[2, 0:3, 5:8] = 0.0

    plugin = Inpainting(max_hole_area=4)
    result = _run_inpainting(plugin, elevation_map)

    assert np.isclose(result[3, 3], 1.5)
    assert np.isnan(result[0:3, 5:8]).all()


def test_inpainting_does_not_fill_border_touching_invalid_cells():
    elevation_map = _make_elevation_map()
    elevation_map[0, 0, 4] = cp.nan
    elevation_map[2, 0, 4] = 0.0

    plugin = Inpainting(max_hole_area=4)
    result = _run_inpainting(plugin, elevation_map)

    assert np.isnan(result[0, 4])


def test_snapshot_aggressive_inpainting_fills_all_holes():
    elevation, elevation_map = _make_snapshot_elevation_map()
    default_plugin = Inpainting(max_hole_area=64, fill_border_holes=False)
    aggressive_plugin = Inpainting(max_hole_area=0, fill_border_holes=True)

    default_result = _run_inpainting(default_plugin, elevation_map)
    aggressive_result = _run_inpainting(aggressive_plugin, elevation_map)

    finite_input = int(np.isfinite(elevation).sum())
    finite_default = int(np.isfinite(default_result).sum())
    finite_aggressive = int(np.isfinite(aggressive_result).sum())

    assert finite_default > finite_input
    assert finite_default < elevation.size
    assert finite_aggressive == elevation.size
    assert np.allclose(aggressive_result[np.isfinite(elevation)], elevation[np.isfinite(elevation)])
