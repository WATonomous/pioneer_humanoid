import cupy as cp

from elevation_mapping_cupy.plugins.inpainting import Inpainting
from elevation_mapping_cupy.plugins.positive_spike_filter import PositiveSpikeFilter


def test_positive_spike_filter_removes_small_spikes_and_preserves_wall():
    filt = PositiveSpikeFilter(
        cell_n=21,
        input_layer_name="inpaint",
        median_filter_size=5,
        spike_height_diff_m=0.35,
        support_height_tolerance_m=0.12,
        max_support_neighbor_count=1,
        max_component_cells=2,
    )

    elevation_map = cp.zeros((7, 21, 21), dtype=cp.float32)
    plugin_layers = cp.zeros((1, 21, 21), dtype=cp.float32)
    plugin_layer_names = ["inpaint"]

    wall_height = 0.7
    plugin_layers[0, 6:15, 10:13] = wall_height

    plugin_layers[0, 4, 4] = 1.1
    plugin_layers[0, 16, 5] = 0.95
    plugin_layers[0, 16, 6] = 0.95

    filtered = filt(elevation_map, ["elevation"], plugin_layers, plugin_layer_names)

    assert float(filtered[4, 4]) < 0.2
    assert float(filtered[16, 5]) < 0.2
    assert float(filtered[16, 6]) < 0.2
    assert float(filtered[10, 11]) > 0.6


def test_positive_spike_filter_keeps_negative_trench_drop():
    filt = PositiveSpikeFilter(
        cell_n=21,
        input_layer_name="inpaint",
        median_filter_size=5,
        spike_height_diff_m=0.35,
        support_height_tolerance_m=0.12,
        max_support_neighbor_count=1,
        max_component_cells=2,
    )

    elevation_map = cp.zeros((7, 21, 21), dtype=cp.float32)
    plugin_layers = cp.zeros((1, 21, 21), dtype=cp.float32)
    plugin_layer_names = ["inpaint"]

    plugin_layers[0, ...] = 0.0
    plugin_layers[0, 8:13, 8:13] = -0.8

    filtered = filt(elevation_map, ["elevation"], plugin_layers, plugin_layer_names)
    assert float(filtered[10, 10]) < -0.7


def test_positive_spike_filter_respects_local_height_threshold():
    filt = PositiveSpikeFilter(
        cell_n=21,
        input_layer_name="inpaint",
        median_filter_size=5,
        spike_height_diff_m=0.35,
        support_height_tolerance_m=0.12,
        max_support_neighbor_count=1,
        max_component_cells=2,
    )

    elevation_map = cp.zeros((7, 21, 21), dtype=cp.float32)
    plugin_layers = cp.zeros((1, 21, 21), dtype=cp.float32)
    plugin_layer_names = ["inpaint"]

    plugin_layers[0, 10, 10] = 0.2

    filtered = filt(elevation_map, ["elevation"], plugin_layers, plugin_layer_names)
    assert float(filtered[10, 10]) > 0.19


def test_positive_spike_filter_rejects_isolated_2x2_patch_without_self_support():
    filt = PositiveSpikeFilter(
        cell_n=21,
        input_layer_name="inpaint",
        median_filter_size=5,
        spike_height_diff_m=0.35,
        support_height_tolerance_m=0.12,
        max_support_neighbor_count=1,
        max_component_cells=4,
    )

    elevation_map = cp.zeros((7, 21, 21), dtype=cp.float32)
    plugin_layers = cp.zeros((1, 21, 21), dtype=cp.float32)
    plugin_layer_names = ["inpaint"]

    plugin_layers[0, 9:11, 9:11] = 1.0

    filtered = filt(elevation_map, ["elevation"], plugin_layers, plugin_layer_names)
    assert float(filtered[9, 9]) < 0.2
    assert float(filtered[9, 10]) < 0.2
    assert float(filtered[10, 9]) < 0.2
    assert float(filtered[10, 10]) < 0.2


def test_positive_spike_filter_rejects_exposed_edge_tip_next_to_nans():
    filt = PositiveSpikeFilter(
        cell_n=21,
        input_layer_name="inpaint",
        median_filter_size=3,
        spike_height_diff_m=0.08,
        support_height_tolerance_m=0.02,
        max_support_neighbor_count=1,
        max_component_cells=2,
        edge_invalid_neighbor_count_min=3,
        edge_peak_diff_m=0.12,
        edge_similarity_tolerance_m=0.05,
        edge_max_similar_neighbor_count=1,
    )

    elevation_map = cp.zeros((7, 21, 21), dtype=cp.float32)
    plugin_layers = cp.full((1, 21, 21), cp.nan, dtype=cp.float32)
    plugin_layer_names = ["inpaint"]

    plugin_layers[0, 8:12, 8] = 0.0
    plugin_layers[0, 9:12, 9:11] = 0.45
    plugin_layers[0, 10, 10] = 0.7

    filtered = filt(elevation_map, ["elevation"], plugin_layers, plugin_layer_names)

    assert 0.4 < float(filtered[10, 10]) < 0.55
    assert float(filtered[9, 9]) > 0.4
    assert float(filtered[9, 10]) > 0.4
    assert cp.isnan(filtered[8, 11])


def test_positive_spike_filter_keeps_supported_ridge_along_nan_boundary():
    filt = PositiveSpikeFilter(
        cell_n=21,
        input_layer_name="inpaint",
        median_filter_size=3,
        spike_height_diff_m=0.08,
        support_height_tolerance_m=0.02,
        max_support_neighbor_count=1,
        max_component_cells=2,
        edge_invalid_neighbor_count_min=3,
        edge_peak_diff_m=0.12,
        edge_similarity_tolerance_m=0.05,
        edge_max_similar_neighbor_count=1,
    )

    elevation_map = cp.zeros((7, 21, 21), dtype=cp.float32)
    plugin_layers = cp.full((1, 21, 21), cp.nan, dtype=cp.float32)
    plugin_layer_names = ["inpaint"]

    plugin_layers[0, 9:12, 8] = 0.0
    plugin_layers[0, 9:11, 9:12] = 0.52

    filtered = filt(elevation_map, ["elevation"], plugin_layers, plugin_layer_names)

    assert float(filtered[9, 10]) > 0.45
    assert float(filtered[10, 10]) > 0.45
    assert cp.isnan(filtered[8, 12])


def test_positive_spike_filter_uses_finite_only_local_background_near_sparse_nan_frontier():
    filt = PositiveSpikeFilter(
        cell_n=21,
        input_layer_name="inpaint",
        median_filter_size=3,
        spike_height_diff_m=0.08,
        support_height_tolerance_m=0.02,
        max_support_neighbor_count=1,
        max_component_cells=2,
        edge_invalid_neighbor_count_min=3,
        edge_peak_diff_m=0.12,
        edge_similarity_tolerance_m=0.05,
        edge_max_similar_neighbor_count=1,
    )

    elevation_map = cp.zeros((7, 21, 21), dtype=cp.float32)
    plugin_layers = cp.full((1, 21, 21), 0.8, dtype=cp.float32)
    plugin_layer_names = ["inpaint"]

    plugin_layers[0, 8:13, 8:13] = cp.nan
    plugin_layers[0, 10, 9] = 0.0
    plugin_layers[0, 10, 10] = 0.0
    plugin_layers[0, 10, 11] = 0.25

    filtered = filt(elevation_map, ["elevation"], plugin_layers, plugin_layer_names)

    assert float(filtered[10, 11]) < 0.1
    assert float(filtered[5, 5]) > 0.75


def test_inpainting_can_consume_despiked_plugin_layer():
    elevation_map = cp.zeros((7, 5, 5), dtype=cp.float32)
    elevation_map[2, ...] = 1.0
    elevation_map[2, 2, 2] = 0.0

    plugin_layers = cp.zeros((1, 5, 5), dtype=cp.float32)
    plugin_layers[0, ...] = 0.4
    plugin_layers[0, 2, 2] = cp.nan

    plugin = Inpainting(cell_n=5, input_layer_name="despiked")
    filled = plugin(elevation_map, ["elevation"], plugin_layers, ["despiked"])
    assert float(filled[2, 2]) > 0.3


def test_inpainting_skips_large_hole_when_hole_size_is_limited():
    elevation_map = cp.zeros((7, 7, 7), dtype=cp.float32)
    elevation_map[2, ...] = 1.0
    elevation_map[2, 1:6, 1:6] = 0.0

    plugin_layers = cp.full((1, 7, 7), 0.4, dtype=cp.float32)
    plugin_layers[0, 1:6, 1:6] = cp.nan

    plugin = Inpainting(cell_n=7, input_layer_name="despiked", max_hole_area=4)
    filled = plugin(elevation_map, ["elevation"], plugin_layers, ["despiked"])

    assert cp.isnan(filled[1, 3])
    assert cp.isnan(filled[3, 3])
