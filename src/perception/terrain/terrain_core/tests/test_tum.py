"""TumDataset on a tiny fake sequence (runs in CI), and the alignment check on real TUM data.

The real-data test needs the sequence in terrain/data/ (see the README), or the
TUM_DATA environment variable pointing at it; it is skipped when neither is there.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from terrain_core.preprocess.pipeline import PreprocessConfig, preprocess
from tum import DEFAULT_PATH, TumDataset

TUM_DATA = Path(os.environ.get("TUM_DATA", DEFAULT_PATH))


# ---------- TumDataset on a fake sequence ----------

@pytest.fixture
def fake_sequence(tmp_path):
    """A 2-image TUM-style folder: depth PNGs + depth.txt + groundtruth.txt."""
    (tmp_path / "depth").mkdir()
    for name, metres in [("a.png", 1.0), ("b.png", 2.0)]:
        Image.fromarray(np.full((4, 6), metres * 5000, np.uint16)).save(tmp_path / "depth" / name)
    (tmp_path / "depth.txt").write_text("# depth maps\n10.00 depth/a.png\n10.10 depth/b.png\n")
    (tmp_path / "groundtruth.txt").write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        "10.00 0 0 0 0 0 0 1\n"
        "10.09 1 2 3 0 0 0 1\n")
    return tmp_path


def test_frame_reads_depth_in_metres(fake_sequence):
    data = TumDataset(fake_sequence)
    assert len(data) == 2
    frame = data.frame(1)
    assert frame.stamp == 10.10
    np.testing.assert_allclose(frame.depth, 2.0)        # 10000 in the PNG = 2 m


def test_pose_at_takes_the_closest_line_in_time(fake_sequence):
    data = TumDataset(fake_sequence)
    np.testing.assert_allclose(data.pose_at(10.10)[:3, 3], [1, 2, 3])   # 10.09 is 10 ms away


def test_pose_at_refuses_a_pose_too_far_away_in_time(fake_sequence):
    with pytest.raises(ValueError):
        TumDataset(fake_sequence).pose_at(10.5)          # closest pose is 0.41 s away


# ---------- alignment on real TUM data (local only) ----------

@pytest.mark.skipif(not (TUM_DATA / "depth.txt").exists(), reason="TUM sequence not downloaded")
def test_two_tum_frames_line_up_in_the_world():
    """Frames 0 and 5 see the same desk: with the right poses their clouds overlap.

    With a pose 50 ms late, or frame 0's pose, they overlap much less: so the right
    timestamps and frames matter, and the check can tell.
    """
    from scipy.spatial import cKDTree

    data = TumDataset(TUM_DATA)
    config = PreprocessConfig()
    a = preprocess(data.frame(0), data.pose_at(data.frame(0).stamp), config).points
    b_frame = data.frame(5)
    tree = cKDTree(a)                                    # KD-tree: fast "closest point" lookup

    def share_within_3cm(T_world_cam):
        dist, _ = tree.query(preprocess(b_frame, T_world_cam, config).points)
        return np.mean(dist < 0.03)

    right = share_within_3cm(data.pose_at(b_frame.stamp))
    late = share_within_3cm(data.pose_at(b_frame.stamp + 0.05))
    stale = share_within_3cm(data.pose_at(data.frame(0).stamp))
    # Measured: right 80%, 50 ms late 35%, frame 0's pose 22%
    assert right > 0.7
    assert late < right - 0.2
    assert stale < right - 0.2
