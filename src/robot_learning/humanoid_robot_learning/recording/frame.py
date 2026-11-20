"""Build dataset frames from ObservationSnapshot."""

from __future__ import annotations

from typing import Any

import numpy as np

from humanoid_robot_learning.recording.snapshot import ObservationSnapshot


def validate_snapshot(
    snapshot: ObservationSnapshot,
    image_keys: list[str],
    *,
    require_images: bool,
) -> None:
    if snapshot.action is None or snapshot.state is None:
        raise ValueError("Missing state or action")
    if require_images:
        for key in image_keys:
            if key not in snapshot.images:
                raise ValueError(f"Missing image '{key}' for frame")


def build_lerobot_frame(
    snapshot: ObservationSnapshot,
    *,
    task: str,
    image_keys: list[str],
) -> dict[str, Any]:
    validate_snapshot(snapshot, image_keys, require_images=bool(image_keys))
    frame: dict[str, Any] = {
        "observation.state": snapshot.state,
        "action": snapshot.action,
        "task": task,
    }
    for key in image_keys:
        frame[f"observation.images.{key}"] = snapshot.images[key]
    return frame


def joints_to_snapshot(
    state_rad: np.ndarray,
    action_rad: np.ndarray,
    images: dict[str, np.ndarray] | None = None,
) -> ObservationSnapshot:
    """Helper for sim / tests: build a snapshot from flat joint vectors."""
    return ObservationSnapshot(
        state=np.asarray(state_rad, dtype=np.float32),
        action=np.asarray(action_rad, dtype=np.float32),
        images=images or {},
    )
