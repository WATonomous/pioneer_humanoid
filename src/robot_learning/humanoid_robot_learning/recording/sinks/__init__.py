"""Dataset writers (LeRobot, HDF5, …)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    root: Path,
    sink_names: list[str],
) -> list:
    """Create the writers for one take folder `root` (it must not exist yet)."""
    sinks = []
    if "lerobot" in sink_names:
        sinks.append(LeRobotSink(cfg, root=root))
    if "hdf5" in sink_names:
        sinks.append(Hdf5Sink(cfg, path=root / "trajectories.h5"))
    return sinks
