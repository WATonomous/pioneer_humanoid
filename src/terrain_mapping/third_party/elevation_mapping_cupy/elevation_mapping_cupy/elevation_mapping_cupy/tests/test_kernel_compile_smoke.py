from __future__ import annotations

from pathlib import Path

import numpy as np
import cupy as cp

from elevation_mapping_cupy import ElevationMap, Parameter


def test_cupy_cuda_is_available():
    # Fail loudly: this repo targets CUDA + CuPy.
    n = cp.cuda.runtime.getDeviceCount()
    assert n >= 1


def test_kernels_compile_and_one_update_step_runs():
    # .../elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping_cupy/tests/test_kernel_compile_smoke.py
    # parents[2] = ROS package root (contains config/).
    root = Path(__file__).resolve().parents[2]
    p = Parameter(
        use_chainer=False,
        weight_file=str(root / "config" / "core" / "weights.dat"),
        plugin_config_file=str(root / "config" / "core" / "plugin_config.yaml"),
    )

    # Keep map tiny so the smoke test stays fast.
    p.resolution = 0.2
    p.map_length = 4.0
    p.update()

    emap = ElevationMap(p)

    # A few points on a plane in the sensor frame.
    pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, -0.5, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
            [2.0, -0.5, 0.0],
        ],
        dtype=np.float32,
    )
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)

    # Should run without exceptions (kernels + traversability torch filter).
    emap.input_pointcloud(pts, ["x", "y", "z"], R, t, 0.0, 0.0)
    emap.update_variance()
    emap.update_time()


def test_extra_point_channels_match_contiguous_xyz_input():
    root = Path(__file__).resolve().parents[2]

    def make_map():
        p = Parameter(
            use_chainer=False,
            weight_file=str(root / "config" / "core" / "weights.dat"),
            plugin_config_file=str(root / "config" / "core" / "plugin_config.yaml"),
        )
        p.resolution = 0.2
        p.map_length = 4.0
        p.enable_visibility_cleanup = False
        p.update()
        return ElevationMap(p)

    xyz = np.array(
        [[1.0, -0.4, 0.0], [1.0, 0.0, 0.1], [1.0, 0.4, 0.2]],
        dtype=np.float32,
    )
    xyzi = np.column_stack((xyz, np.asarray([100.0, 200.0, 300.0], dtype=np.float32)))
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    xyz_map = make_map()
    xyzi_map = make_map()

    xyz_map.input_pointcloud(xyz, ["x", "y", "z"], rotation, translation, 0.0, 0.0)
    xyzi_map.input_pointcloud(
        xyzi,
        ["x", "y", "z", "intensity"],
        rotation,
        translation,
        0.0,
        0.0,
    )

    cp.testing.assert_allclose(xyzi_map.elevation_map[:3], xyz_map.elevation_map[:3])


def test_dense_cell_fusion_is_order_invariant():
    root = Path(__file__).resolve().parents[2]

    def run(points):
        p = Parameter(
            use_chainer=False,
            weight_file=str(root / "config" / "core" / "weights.dat"),
            plugin_config_file=str(root / "config" / "core" / "plugin_config.yaml"),
        )
        p.resolution = 0.2
        p.map_length = 4.0
        p.enable_visibility_cleanup = False
        p.enable_drift_compensation = False
        p.update()
        elevation_map = ElevationMap(p)
        elevation_map.input_pointcloud(
            points,
            ["x", "y", "z"],
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            0.0,
            0.0,
        )
        return elevation_map.elevation_map[:3].copy()

    z = np.linspace(-0.02, 0.02, 512, dtype=np.float32)
    points = np.column_stack((np.ones_like(z), np.zeros_like(z), z))

    forward = run(points)
    reverse = run(points[::-1].copy())

    cp.testing.assert_allclose(forward, reverse, rtol=1e-6, atol=1e-6)
    cp.testing.assert_array_equal(forward[2], reverse[2])


def test_nan_and_inf_points_are_rejected_in_the_kernel():
    root = Path(__file__).resolve().parents[2]
    p = Parameter(
        use_chainer=False,
        weight_file=str(root / "config" / "core" / "weights.dat"),
        plugin_config_file=str(root / "config" / "core" / "plugin_config.yaml"),
    )
    p.resolution = 0.2
    p.map_length = 4.0
    p.enable_visibility_cleanup = False
    p.update()
    elevation_map = ElevationMap(p)
    points = np.asarray(
        [[1.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, np.inf, 0.0], [1.0, 0.0, -np.inf]],
        dtype=np.float32,
    )

    elevation_map.input_pointcloud(
        points,
        ["x", "y", "z"],
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        0.0,
        0.0,
    )

    assert int(cp.count_nonzero(elevation_map.elevation_map[2]).item()) == 1
