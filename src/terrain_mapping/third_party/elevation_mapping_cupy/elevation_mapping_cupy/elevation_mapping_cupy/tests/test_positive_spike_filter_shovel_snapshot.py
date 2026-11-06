import cupy as cp
import cupyx.scipy.ndimage as ndimage

from elevation_mapping_cupy.plugins.plugin_manager import PluginManager, PluginParams


# Captured from the paused DIG3D replay around SHOVEL_1300 in:
# /home/lorenzo/mcap/dig3d_2026-03-26/dig3d_real_run_2026-03-26_21-02-16
# Input layer is the live `near_base_filtered` crop from the paused replay.
# The point of this fixture is to keep the real shovel-neighborhood residual bump
# case in-tree and verify the configured two-stage despike chain clears it.
_DIG3D_SHOVEL_NEAR_BASE_FILTERED = [
    [-0.106454, -0.105685, -0.079661, -0.212375, -0.218414, -0.086213, 0.010798, -0.060414, -0.059289, -0.103631, -0.112124, -0.05758, -0.083119, -0.090957, -0.112136, -0.24986, -0.306922],
    [-0.082693, -0.083633, -0.072124, -0.160965, -0.146849, -0.015773, 0.015772, -0.03653, -0.05221, -0.080262, -0.087122, -0.063964, -0.077231, -0.127877, -0.164966, -0.239629, -0.382257],
    [-0.052809, None, -0.021516, -0.109432, -0.056233, 0.123821, 0.079235, -0.022929, -0.091618, -0.128768, -0.125102, -0.13553, -0.122142, -0.122738, -0.151495, -0.194349, -0.357536],
    [None, None, 0.061757, -0.03017, 0.081713, 0.176614, 0.072558, -0.037393, -0.206509, -0.210881, -0.145878, -0.118147, -0.109268, -0.111122, -0.144733, -0.177084, -0.203425],
    [None, 0.151703, 0.050805, -0.022074, 0.066795, 0.173782, 0.06233, -0.057888, -0.259199, -0.288448, -0.188871, -0.147942, -0.127483, -0.125233, -0.150424, -0.179148, -0.203675],
    [0.211117, 0.17846, 0.025005, -0.042287, 0.070251, 0.144075, 0.077089, -0.067781, -0.190521, -0.311407, -0.307716, -0.219126, -0.197178, -0.181394, -0.187897, -0.212334, -0.221471],
    [0.182953, 0.102395, -0.017858, -0.005488, 0.081912, 0.124752, 0.062562, -0.055928, -0.111967, -0.294161, -0.323928, -0.313504, -0.294019, -0.263872, -0.265382, -0.264924, -0.259117],
    [0.210857, 0.234403, 0.007223, 0.072316, 0.105333, 0.113418, 0.073348, -0.006843, -0.075788, -0.105427, -0.067347, -0.165784, -0.237731, -0.319602, -0.342158, -0.313188, -0.283981],
    [0.274896, 0.262294, 0.103932, 0.117801, 0.102389, 0.095856, 0.101016, 0.066938, -0.044105, -0.063601, -0.045781, -0.138703, -0.214677, -0.300229, -0.282262, -0.374491, -0.317916],
    [None, 0.296808, 0.240826, 0.152636, 0.230127, 0.132109, 0.144157, 0.107457, 0.016317, -0.019914, -0.008179, -0.085746, -0.160717, -0.212806, -0.237671, -0.315363, -0.434866],
    [None, 0.333239, 0.337791, 0.189904, 0.222295, 0.214662, 0.142167, 0.096078, 0.100083, 0.125587, 0.032163, -0.080256, -0.085567, -0.126295, -0.211153, -0.346405, -0.364523],
    [None, None, 0.361097, 0.308092, 0.315232, 0.278821, 0.174725, 0.088646, 0.12519, 0.153612, 0.103663, 0.027897, -0.005071, -0.108958, -0.151502, -0.471054, None],
    [None, None, 0.473916, 0.40097, 0.302285, 0.212646, 0.189816, 0.065589, 0.117446, 0.146662, 0.126095, 0.087161, 0.140688, None, None, None, None],
    [None, None, None, 0.410359, 0.409464, 0.268848, 0.167423, 0.135426, 0.136431, 0.182271, 0.222348, 0.265474, 0.162581, 0.077579, None, None, -0.280323],
    [None, None, None, 0.493615, 0.446911, 0.406165, 0.160394, 0.173892, 0.275772, 0.274095, 0.260192, 0.272611, 0.186416, 0.152871, 0.11557, -0.008845, -0.121657],
    [None, None, None, None, 0.470584, 0.467531, 0.430181, 0.25324, 0.35233, 0.324383, 0.285365, 0.273898, 0.292658, 0.205482, 0.121639, 0.01228, -0.094619],
    [None, None, None, None, 0.519143, 0.471398, 0.396459, 0.262259, 0.337293, 0.333187, 0.300346, 0.277374, 0.289072, 0.239099, 0.101106, -0.082525, -0.179052],
]


def _as_cupy(snapshot):
    return cp.asarray([[cp.nan if value is None else value for value in row] for row in snapshot], dtype=cp.float32)


def _run_configured_despike_chain(snapshot: cp.ndarray) -> cp.ndarray:
    elevation_map = cp.zeros((7, snapshot.shape[0], snapshot.shape[1]), dtype=cp.float32)
    elevation_map[0] = snapshot
    elevation_map[2] = cp.isfinite(snapshot).astype(cp.float32)
    layer_names = [
        "elevation",
        "variance",
        "is_valid",
        "traversability",
        "time",
        "upper_bound",
        "is_upper_bound",
    ]

    manager = PluginManager(cell_n=snapshot.shape[0])
    manager.init(
        [
            PluginParams(name="positive_spike_filter", layer_name="despiked_coarse"),
            PluginParams(name="positive_spike_filter", layer_name="despiked"),
        ],
        [
            {
                "input_layer_name": "elevation",
                "median_filter_size": 5,
                "spike_height_diff_m": 0.35,
                "support_height_tolerance_m": 0.12,
                "max_support_neighbor_count": 1,
                "max_component_cells": 4,
            },
            {
                "input_layer_name": "despiked_coarse",
                "median_filter_size": 3,
                "spike_height_diff_m": 0.08,
                "support_height_tolerance_m": 0.02,
                "max_support_neighbor_count": 1,
                "max_component_cells": 2,
                "edge_invalid_neighbor_count_min": 3,
                "edge_peak_diff_m": 0.12,
                "edge_similarity_tolerance_m": 0.05,
                "edge_max_similar_neighbor_count": 1,
            },
        ],
    )
    manager.update_with_name("despiked", elevation_map, layer_names)
    return manager.get_map_with_name("despiked")


def _count_local_positive_bump_components(
    height_map: cp.ndarray,
    threshold_m: float,
    radius_m: float | None = None,
    resolution_m: float = 0.1,
) -> int:
    finite_mask = cp.isfinite(height_map)
    if not cp.any(finite_mask):
        return 0
    padded = cp.pad(
        cp.where(finite_mask, height_map, cp.nan),
        pad_width=2,
        mode="constant",
        constant_values=cp.nan,
    )
    window_stack = cp.stack(
        [padded[dy : dy + height_map.shape[0], dx : dx + height_map.shape[1]] for dy in range(5) for dx in range(5)],
        axis=0,
    )
    local_median = cp.nanmedian(window_stack, axis=0)
    bump_mask = finite_mask & ((height_map - local_median) > threshold_m)
    if radius_m is not None:
        center = (float(height_map.shape[0]) - 1.0) / 2.0
        coords = (cp.arange(height_map.shape[0], dtype=cp.float32) - center) * resolution_m
        yy, xx = cp.meshgrid(coords, coords, indexing="ij")
        bump_mask &= (xx * xx + yy * yy) <= (radius_m * radius_m)
    _, component_count = ndimage.label(bump_mask, structure=ndimage.generate_binary_structure(2, 2))
    return int(component_count)


def test_positive_spike_filter_chain_preserves_nan_mask_in_dig3d_shovel_snapshot():
    snapshot = _as_cupy(_DIG3D_SHOVEL_NEAR_BASE_FILTERED)
    filtered = _run_configured_despike_chain(snapshot)

    assert bool(cp.array_equal(cp.isfinite(filtered), cp.isfinite(snapshot)))


def test_positive_spike_filter_chain_clears_small_residual_shovel_bumps_in_snapshot():
    snapshot = _as_cupy(_DIG3D_SHOVEL_NEAR_BASE_FILTERED)
    filtered = _run_configured_despike_chain(snapshot)

    assert _count_local_positive_bump_components(filtered, threshold_m=0.15, radius_m=0.8) == 0


def test_positive_spike_filter_chain_preserves_supported_wall():
    snapshot = cp.zeros((21, 21), dtype=cp.float32)

    snapshot[6:15, 10:13] = 0.7
    snapshot[4, 4] = 1.1
    snapshot[16, 5] = 0.95
    snapshot[16, 6] = 0.95

    filtered = _run_configured_despike_chain(snapshot)

    assert float(filtered[10, 11]) > 0.6
    assert float(filtered[4, 4]) < 0.2
    assert float(filtered[16, 5]) < 0.2
    assert float(filtered[16, 6]) < 0.2
