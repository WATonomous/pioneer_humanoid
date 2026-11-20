"""Synthetic raw takes with known ground truth: a smooth hand motion seen through the rig's streams."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from umi_gripper import config as umi_config
from umi_gripper import geometry
from umi_gripper.take import PoseSample, write_take


def synthetic_config(**overrides) -> dict:
    return umi_config.load_config(calibration=None, overrides=overrides or None)


def true_grip_pose(t: float) -> np.ndarray:
    """A hand drawing a 10 cm figure in front of the headset while slowly tilting."""
    position = np.array([0.1 * np.sin(1.3 * t), 1.0 + 0.05 * np.sin(2.1 * t), -0.4 + 0.05 * np.cos(0.9 * t)])
    rotation = geometry.rotation_exp([0.4 * np.sin(1.1 * t), 0.3 * np.cos(0.7 * t), 0.2 * np.sin(1.7 * t)])
    return geometry.make_pose(rotation, position)


def true_closure(t: float) -> float:
    return float(0.5 + 0.5 * np.sin(0.8 * t))


def make_take(path: Path, duration: float = 4.0, *, pose_hz: float = 72.0, camera_hz: float = 25.0,
              width_hz: float = 100.0, clock_offset_s: float = 1234.5, transport_jitter_s: float = 0.004,
              camera_latency_s: float = 0.0, untracked=(), epochs=None, camera: bool = True,
              jump_at: float | None = None, seed: int = 0, meta: dict | None = None) -> Path:
    """Write a raw take. `untracked`: [(t0, t1)] of controller loss. Host clock = true time (+ stream latencies)."""
    rng = np.random.default_rng(seed)
    poses = []
    for i, t in enumerate(np.arange(0.0, duration, 1.0 / pose_hz)):
        tracked = not any(a <= t < b for a, b in untracked)
        sample = PoseSample(host_s=t + transport_jitter_s * rng.random(), client_ms=(t - clock_offset_s) * 1000.0,
                            epoch=0 if epochs is None else epochs(t), tracked=tracked)
        if tracked:
            pose = true_grip_pose(t)
            if jump_at is not None and t >= jump_at:
                pose = pose.copy()
                pose[:3, 3] += [0.3, 0.0, 0.0]
            sample.position = pose[:3, 3]
            sample.quat_xyzw = geometry.matrix_to_quat(pose[:3, :3])
            sample.trigger = true_closure(t)
        poses.append(sample)
    width = [(t, true_closure(t), 1000 + 400 * true_closure(t)) for t in np.arange(0.0, duration, 1.0 / width_hz)]
    camera_s = list(np.arange(0.01, duration, 1.0 / camera_hz) + camera_latency_s) if camera else []
    write_take(path, {"task": "synthetic", "camera": camera, **(meta or {})}, poses, width, camera_s)
    if camera:
        import cv2
        for i, _ in enumerate(camera_s):
            frame = np.full((480, 640, 3), i % 255, dtype=np.uint8)
            cv2.imwrite(str(path / "frames" / f"{i:06d}.jpg"), frame)
    return path
