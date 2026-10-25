"""Dataset writers (LeRobot, HDF5, …)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from humanoid_robot_learning.recording.record_utils import get_next_experiment_path_with_gap
from humanoid_robot_learning.recording.sinks.hdf5 import Hdf5Sink
from humanoid_robot_learning.recording.sinks.lerobot import LeRobotSink

SINK_ALIASES = {
    "lerobot": "lerobot",
    "hdf5": "hdf5",
}


def parse_sink_names(raw: str) -> list[str]:
    names: list[str] = []
    for part in raw.split(","):
        key = SINK_ALIASES.get(part.strip().lower())
        if key is None:
            raise ValueError(
                f"Unknown sink '{part}'. Choose from: lerobot, hdf5 (comma-separated)."
            )
        if key not in names:
            names.append(key)
    if not names:
        raise ValueError("At least one sink is required.")
    return names


def create_sinks(
    cfg: dict[str, Any],
    *,
    record_root: Path,
    sink_names: list[str],
) -> tuple[Path, list]:
    """Create writers under the next experiment folder (e.g. datasets/record/001/)."""
    exp_path = get_next_experiment_path_with_gap(record_root)
    sinks = []
    if "lerobot" in sink_names:
        sinks.append(LeRobotSink(cfg, root=exp_path))
    if "hdf5" in sink_names:
        sinks.append(Hdf5Sink(cfg, path=exp_path / "trajectories.h5"))
    return exp_path, sinks
