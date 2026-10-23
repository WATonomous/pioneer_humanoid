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
from humanoid_robot_learning.recording.record_utils import RateLimiter
from humanoid_robot_learning.recording.schema import enabled_images
from humanoid_robot_learning.recording.sinks import create_sinks, parse_sink_names
from humanoid_robot_learning.recording.snapshot import ObservationSnapshot

logger = logging.getLogger(__name__)


def dry_snapshot(t: float, image_shapes: dict[str, tuple[int, int]], dim: int) -> ObservationSnapshot:
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
) -> Path:
    image_shapes = {k: (int(v["height"]), int(v["width"])) for k, v in enabled_images(cfg).items()}
    dim = len(cfg["joint_names"])
    fps = float(cfg.get("fps", 30))
    rate = RateLimiter(fps)

    output_dir, sinks = create_sinks(cfg, record_root=record_root, sink_names=sink_names)
    session = RecordSession(cfg, sinks, settings, output_dir=output_dir)
    session.start_keyboard()

    logger.info("Writing to %s (sinks: %s)", output_dir, ", ".join(s.name for s in sinks))
    t0 = time.monotonic()

    try:
        while session.episode_index < settings.num_episodes and not session.flags.abort:
            session.begin_episode()

            while not session.flags.success and not session.flags.abort:
                rate.sleep()
                source.maybe_spin()

                if not session.flags.start:
                    continue

                session.check_timed_episode()

                try:
                    snapshot = source()
                    if dry_run:
                        snapshot = dry_snapshot(time.monotonic() - t0, image_shapes, dim)
                    session.ingest_snapshot(snapshot)
                except ValueError:
                    continue

            if session.flags.abort:
                break

            session.save_episode_if_ready()
            session.maybe_reset_pause()

        session.finalize()
    finally:
        session.cleanup()
        source.cleanup()

    return output_dir
