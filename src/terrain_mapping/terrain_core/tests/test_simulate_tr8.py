"""Regression for the camera simulation's ramp-to-landing geometry and review labels."""
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_ramp_exports_match_measured_geometry_and_local_labels(tmp_path):
    script = Path(__file__).resolve().parents[1] / "prototype" / "simulate_tr8.py"
    subprocess.run([sys.executable, str(script), "--noise-mm", "0", "--out", str(tmp_path)], check=True)
    for name, degrees, walkable in [
        ("step_14cm_gentle_ramp", 3, True),
        ("step_14cm_ramp", 5, True),
        ("step_14cm_steep_ramp", 9, False),
    ]:
        grid = np.load(tmp_path / f"{name}.npy")
        labels = np.load(tmp_path / f"{name}_labels.npy")
        x = (np.arange(grid.shape[1]) + 0.5) * 0.04
        grade = np.tan(np.deg2rad(degrees))
        expected_height = np.clip(0.14 + grade * (x - 1.60), 0, 0.14)
        # Each stored point may lie anywhere within its 4 cm cell.
        seen = np.isfinite(grid)
        error = np.abs(grid - expected_height)
        assert np.max(error[seen]) <= grade * 0.02 + 1e-6

        reviewed = np.isfinite(labels)
        incline = reviewed & (expected_height > 0) & (expected_height < 0.14)
        landing = reviewed & (expected_height == 0.14)
        assert incline.any() and landing.any()
        assert np.all(labels[incline] == walkable)
        assert np.all(labels[landing] == 1)
        assert np.array_equal(reviewed, incline | landing)
