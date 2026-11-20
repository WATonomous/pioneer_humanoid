"""LeRobot dataset writer sink."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from humanoid_robot_learning.recording.frame import build_lerobot_frame
from humanoid_robot_learning.recording.schema import create_dataset, enabled_images
from humanoid_robot_learning.recording.snapshot import ObservationSnapshot


class LeRobotSink:
    def __init__(self, cfg: dict[str, Any], *, root: Path) -> None:
        self._image_keys = list(enabled_images(cfg).keys())
        self._root = root
        self._dataset = create_dataset(cfg, root=root)

    @property
    def name(self) -> str:
        return "lerobot"

    @property
    def output_path(self) -> Path:
        return self._root

    def add_frame(self, snapshot: ObservationSnapshot, task: str) -> None:
        frame = build_lerobot_frame(
            snapshot,
            task=task,
            image_keys=self._image_keys,
        )
        self._dataset.add_frame(frame)

    def clear_episode(self) -> None:
        self._dataset.clear_episode_buffer()

    def save_episode(self) -> None:
        self._dataset.save_episode()

    def finalize(self) -> None:
        self._dataset.finalize()
        # One dataset per take: stop its image-writer threads so they don't pile up.
        self._dataset.stop_image_writer()
