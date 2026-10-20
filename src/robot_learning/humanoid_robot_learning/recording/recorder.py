"""Episode recording session shared by ROS CLI and Isaac sim teleop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from humanoid_robot_learning.recording.episode_keys import EpisodeFlags, EpisodeKeyboard
from humanoid_robot_learning.recording.schema import enabled_images
from humanoid_robot_learning.recording.snapshot import ObservationSnapshot

logger = logging.getLogger(__name__)


class SnapshotSource(Protocol):
    def maybe_spin(self) -> None: ...
    def __call__(self) -> ObservationSnapshot: ...
    def cleanup(self) -> None: ...


class RecordSink(Protocol):
    name: str
    output_path: Path

    def add_frame(self, snapshot: ObservationSnapshot, task: str) -> None: ...
    def clear_episode(self) -> None: ...
    def save_episode(self) -> None: ...
    def finalize(self) -> None: ...


@dataclass
class RecordSettings:
    num_episodes: int = 10
    task_description: str = "humanoid demonstration"
    episode_time_s: float | None = None
    reset_time_s: float = 5.0
    auto_start: bool = True
    use_keyboard: bool = True


class RecordSession:
    """Owns episode flags, optional keyboard, and one or more dataset sinks."""

    def __init__(
        self,
        cfg: dict[str, Any],
        sinks: list[RecordSink],
        settings: RecordSettings,
        *,
        output_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.sinks = sinks
        self.settings = settings
        self.output_dir = output_dir
        self.image_keys = list(enabled_images(cfg).keys())
        self.flags = EpisodeFlags(start=bool(settings.auto_start))
        self._keyboard = EpisodeKeyboard(self.flags)
        self._keyboard_started = False
        self._episode_index = 0
        self._episode_start = time.monotonic()
        self._last_frame_t = 0.0
        self.fps = float(cfg.get("fps", 30))
        self._frame_period = 1.0 / self.fps

    def start_keyboard(self) -> bool:
        if not self.settings.use_keyboard:
            return False
        self._keyboard_started = self._keyboard.start()
        return self._keyboard_started

    def stop_keyboard(self) -> None:
        if self._keyboard_started:
            self._keyboard.stop()
            self._keyboard_started = False

    @property
    def episode_index(self) -> int:
        return self._episode_index

    @property
    def is_complete(self) -> bool:
        return self._episode_index >= self.settings.num_episodes or self.flags.abort

    def begin_episode(self) -> None:
        self.flags.success = False
        self.flags.remove = False
        if self.settings.auto_start:
            self.flags.start = True
        self._episode_start = time.monotonic()

    def should_record_frame(self, now: float | None = None) -> bool:
        if not self.flags.start or self.flags.success or self.flags.abort:
            return False
        t = time.monotonic() if now is None else now
        if t - self._last_frame_t < self._frame_period:
            return False
        self._last_frame_t = t
        return True

    def check_timed_episode(self) -> None:
        if self.settings.episode_time_s is None:
            return
        if time.monotonic() - self._episode_start >= self.settings.episode_time_s:
            self.flags.success = True

    def ingest_snapshot(self, snapshot: ObservationSnapshot) -> bool:
        """Apply discard/save flags and write one frame. Returns True if written."""
        if self.flags.remove:
            self.clear_episode_buffers()
            self.flags.remove = False
            self._episode_start = time.monotonic()
            return False

        for sink in self.sinks:
            sink.add_frame(snapshot, self.settings.task_description)
        return True

    def clear_episode_buffers(self) -> None:
        for sink in self.sinks:
            sink.clear_episode()

    def save_episode_if_ready(self) -> bool:
        if not self.flags.success:
            return False
        for sink in self.sinks:
            sink.save_episode()
        self._episode_index += 1
        logger.info(
            "Saved episode %s / %s",
            self._episode_index,
            self.settings.num_episodes,
        )
        return True

    def maybe_reset_pause(self) -> None:
        if (
            self._episode_index < self.settings.num_episodes
            and self.settings.reset_time_s > 0
            and not self.flags.abort
        ):
            logger.info("Reset window %.1fs", self.settings.reset_time_s)
            time.sleep(self.settings.reset_time_s)

    def finalize(self) -> None:
        if self.flags.abort:
            self.clear_episode_buffers()
        for sink in self.sinks:
            sink.finalize()

    def cleanup(self) -> None:
        self.stop_keyboard()
