"""Recording session for the ROS / dry-run CLI: one LeRobot dataset folder per take.

Keys: S starts a take, N stops and saves it, D deletes the newest saved take of this session
(ignored while a take is recording), ESC quits. Layout:

    <record_root>/<session NNN>/take_001/   one complete LeRobot dataset (+ trajectories.h5)
                                take_002/   ...

`humanoid-merge-takes` combines the takes into one dataset for training or upload.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from humanoid_robot_learning.recording.episode_keys import EpisodeFlags, EpisodeKeyboard
from humanoid_robot_learning.recording.snapshot import ObservationSnapshot

logger = logging.getLogger(__name__)

TAKE_PREFIX = "take_"


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


# Builds the writers for one take folder (it must not exist yet).
SinkFactory = Callable[[Path], list[RecordSink]]


@dataclass
class RecordSettings:
    num_episodes: int = 10  # saved takes to keep; deleted takes don't count
    task_description: str = "humanoid demonstration"
    episode_time_s: float | None = None  # auto-stop and save each take after N seconds
    reset_time_s: float = 5.0  # pause between takes, auto_start only
    auto_start: bool = False  # start takes without S (also used when the keyboard is unavailable)
    use_keyboard: bool = True


def take_dirs(session_dir: Path) -> list[Path]:
    """Take folders of a session, oldest first."""
    if not session_dir.is_dir():
        return []
    return sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith(TAKE_PREFIX))


class RecordSession:
    """Owns the keys, the take in progress, and the takes saved this session."""

    def __init__(
        self,
        settings: RecordSettings,
        *,
        session_dir: Path,
        make_sinks: SinkFactory,
    ) -> None:
        self.settings = settings
        self.session_dir = session_dir
        self._make_sinks = make_sinks
        self.flags = EpisodeFlags()
        self._keyboard = EpisodeKeyboard(self.flags, discard=None, delete_last="d")
        self._keyboard_started = False
        self._saved: list[Path] = []  # this session's saved takes, oldest first
        self._take_dir: Path | None = None
        self._sinks: list[RecordSink] = []
        self._frames = 0
        self._take_start = 0.0
        self._announced_full = False

    # --- keyboard ---------------------------------------------------------

    def start_keyboard(self) -> bool:
        if self.settings.use_keyboard:
            self._keyboard_started = self._keyboard.start()
        if not self._keyboard_started and not self.settings.auto_start:
            logger.warning("No keyboard: takes start automatically (as with --auto_start).")
            self.settings.auto_start = True
        return self._keyboard_started

    def stop_keyboard(self) -> None:
        if self._keyboard_started:
            self._keyboard.stop()
            self._keyboard_started = False

    # --- state ------------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._take_dir is not None

    @property
    def num_saved(self) -> int:
        return len(self._saved)

    @property
    def is_full(self) -> bool:
        return self.num_saved >= self.settings.num_episodes

    @property
    def is_complete(self) -> bool:
        """Stop the loop: ESC, or enough takes and nothing left to undo by key."""
        if self.flags.abort:
            return True
        return self.is_full and not self._keyboard_started

    # --- loop hooks -------------------------------------------------------

    def handle_idle_keys(self) -> None:
        """Between takes: D deletes the newest saved take, S (or auto_start) starts one."""
        self.flags.success = False  # N with no take recording does nothing
        if self.flags.delete_last:
            self.flags.delete_last = False
            self.delete_last_take()
        if self.is_full:
            if self.flags.start and not self._announced_full:
                logger.info("Already have %d takes: press D to delete one, or ESC to finish.", self.num_saved)
            if not self._announced_full and self._keyboard_started:
                logger.info("%d / %d takes saved. D deletes the last one; ESC finishes.",
                            self.num_saved, self.settings.num_episodes)
            self._announced_full = True
            self.flags.start = False
            return
        self._announced_full = False
        if self.flags.start or self.settings.auto_start:
            self.flags.start = False
            self.begin_take()

    def handle_recording_keys(self) -> bool:
        """While recording: D and S are ignored; N (or the timer) saves. Returns True if saved."""
        if self.flags.delete_last:
            self.flags.delete_last = False
            logger.info("D ignored while recording: press N to save first.")
        self.flags.start = False
        if self.settings.episode_time_s is not None and \
                time.monotonic() - self._take_start >= self.settings.episode_time_s:
            self.flags.success = True
        if self.flags.success:
            self.flags.success = False
            self.save_take()
            return True
        return False

    def add_frame(self, snapshot: ObservationSnapshot) -> None:
        for sink in self._sinks:
            sink.add_frame(snapshot, self.settings.task_description)
        self._frames += 1

    # --- takes ------------------------------------------------------------

    def _next_take_dir(self) -> Path:
        # One past the highest take folder (a deleted newest take's number gets reused).
        numbers = [int(p.name[len(TAKE_PREFIX):]) for p in take_dirs(self.session_dir)
                   if p.name[len(TAKE_PREFIX):].isdigit()]
        return self.session_dir / f"{TAKE_PREFIX}{max(numbers, default=0) + 1:03d}"

    def begin_take(self) -> None:
        take_dir = self._next_take_dir()
        self._sinks = self._make_sinks(take_dir)
        self._take_dir = take_dir
        self._frames = 0
        self._take_start = time.monotonic()
        logger.info("[S] Recording %s (take %d of %d).", take_dir.name, self.num_saved + 1,
                    self.settings.num_episodes)

    def save_take(self) -> None:
        assert self._take_dir is not None
        take_dir = self._take_dir
        if self._frames == 0:
            logger.warning("%s has no frames; not saved.", take_dir.name)
            self.discard_take()
            return
        logger.info("[N] Saving %s (%d frames)...", take_dir.name, self._frames)
        for sink in self._sinks:
            sink.save_episode()
            sink.finalize()
        self._saved.append(take_dir)
        self._sinks, self._take_dir = [], None
        logger.info("Saved %s (%d / %d). S: next take, D: delete it.", take_dir.name, self.num_saved,
                    self.settings.num_episodes)
        if self.settings.auto_start and not self.is_full and self.settings.reset_time_s > 0:
            logger.info("Reset window %.1fs", self.settings.reset_time_s)
            time.sleep(self.settings.reset_time_s)

    def discard_take(self) -> None:
        """Drop the take in progress and its folder."""
        if self._take_dir is None:
            return
        for sink in self._sinks:
            sink.clear_episode()
            sink.finalize()
        shutil.rmtree(self._take_dir, ignore_errors=True)
        self._sinks, self._take_dir = [], None

    def delete_last_take(self) -> None:
        if not self._saved:
            logger.info("[D] No saved take to delete in this session.")
            return
        take_dir = self._saved.pop()
        shutil.rmtree(take_dir)
        logger.info("[D] Deleted %s (%d / %d saved).", take_dir.name, self.num_saved,
                    self.settings.num_episodes)

    def close(self) -> None:
        """End the session: a take still recording (ESC, Ctrl+C, crash) is dropped."""
        if self.recording:
            logger.info("Take %s was not saved (session ended while recording).", self._take_dir.name)
            self.discard_take()
        self.stop_keyboard()
        if not self._saved and self.session_dir.is_dir() and not any(self.session_dir.iterdir()):
            self.session_dir.rmdir()
