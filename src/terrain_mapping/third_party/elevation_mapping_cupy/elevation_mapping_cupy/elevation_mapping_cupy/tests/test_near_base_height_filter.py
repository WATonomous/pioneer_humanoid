import cupy as cp

from elevation_mapping_cupy.plugins.near_base_height_filter import NearBaseHeightFilter


def test_near_base_height_filter_removes_above_threshold_cells_near_base():
    filt = NearBaseHeightFilter(
        cell_n=21,
        input_layer_name="elevation",
        resolution=0.5,
        radius_m=4.0,
        max_allowed_height_m=0.5,
    )

    elevation_map = cp.zeros((7, 21, 21), dtype=cp.float32)
    elevation_map[2, ...] = 1.0
    elevation_map[0, 10, 10] = 0.6
    elevation_map[0, 10, 14] = 0.6

    filtered = filt(elevation_map, ["elevation"], cp.zeros((0, 21, 21), dtype=cp.float32), [])

    assert cp.isnan(filtered[10, 10])
    assert cp.isnan(filtered[10, 14])


def test_near_base_height_filter_keeps_low_and_far_cells():
    filt = NearBaseHeightFilter(
        cell_n=21,
        input_layer_name="elevation",
        resolution=0.5,
        radius_m=4.0,
        max_allowed_height_m=0.5,
    )

    elevation_map = cp.zeros((7, 21, 21), dtype=cp.float32)
    elevation_map[2, ...] = 1.0
    elevation_map[0, 10, 10] = 0.4
    elevation_map[0, 10, 19] = 0.8

    filtered = filt(elevation_map, ["elevation"], cp.zeros((0, 21, 21), dtype=cp.float32), [])

    assert float(filtered[10, 10]) > 0.3
    assert float(filtered[10, 19]) > 0.7
