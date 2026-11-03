"""Blocking record loop for ROS / dry-run CLIs."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_robot_learning.recording.frame import joints_to_snapshot
from humanoid_robot_learning.recording.recorder import RecordSession, RecordSettings, SnapshotSource
from humanoid_robot_learning.recording.record_utils import RateLimiter, get_next_experiment_path_with_gap
from humanoid_robot_learning.recording.schema import enabled_images
from humanoid_robot_learning.recording.sinks import create_sinks

logger = logging.getLogger(__name__)


def dry_snapshot(t: float, image_shapes: dict[str, tuple[int, int]], dim: int):
    vec = np.array([0.2 * math.sin(t + i) for i in range(dim)], dtype=np.float32)
    images = {key: np.zeros((h, w, 3), dtype=np.uint8) for key, (h, w) in image_shapes.items()}
    return joints_to_snapshot(vec, vec.copy(), images)


def run_record_loop(
    cfg: dict[str, Any],
    source: SnapshotSource,
    settings: RecordSettings,
    *,
    record_root: Path,
    sink_names: list[str],
    dry_run: bool = False,
    session: RecordSession | None = None,
) -> Path:
    """Record takes into the next session folder under `record_root`; returns that folder.

    `session` is for tests (drive the flags without a keyboard)."""
    image_shapes = {k: (int(v["height"]), int(v["width"])) for k, v in enabled_images(cfg).items()}
    dim = len(cfg["joint_names"])
    rate = RateLimiter(float(cfg.get("fps", 30)))

    if session is None:
        session_dir = get_next_experiment_path_with_gap(record_root)
        session = RecordSession(
            settings,
            session_dir=session_dir,
            make_sinks=lambda take_dir: create_sinks(cfg, root=take_dir, sink_names=sink_names),
        )
        session.start_keyboard()
    session.session_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Session folder %s (sinks: %s)", session.session_dir, ", ".join(sink_names))
    logger.info("Keys: S start take, N stop + save, D delete last saved take, ESC quit")
    t0 = time.monotonic()

    try:
        while not session.is_complete:
            rate.sleep()
            source.maybe_spin()

            if not session.recording:
                session.handle_idle_keys()
                continue
            if session.handle_recording_keys():
                continue

            try:
                snapshot = source()
                if dry_run:
                    snapshot = dry_snapshot(time.monotonic() - t0, image_shapes, dim)
            except ValueError:
                continue  # no fresh data this tick (e.g. stale gripper feedback)
            session.add_frame(snapshot)
    finally:
        session.close()
        source.cleanup()

    logger.info("%d take(s) saved under %s", session.num_saved, session.session_dir)
    return session.session_dir
