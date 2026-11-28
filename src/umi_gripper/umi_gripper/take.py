"""Raw take on disk: every stream as it arrived, host-timestamped, nothing resampled.

    <raw_root>/<session>/take_0007/
      meta.json     task, rig (headset, controller, width source) and the config used
      poses.npz     host_s, client_ms, epoch, tracked, position (n,3), quat_xyzw (n,4), trigger  -- the GRIP pose
      width.npz     host_s, closure (0 open .. 1 closed), raw (encoder counts or trigger value)
      camera.npz    host_s (one per frame, frames/000000.jpg ...)
      frames/

The grip pose, not the TCP, is stored, so a later calibration re-converts old takes (postprocess only applies the
config). Writing frames to disk as they come keeps a long take's memory flat.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

POSE_FIELDS = ("host_s", "client_ms", "epoch", "tracked", "position", "quat_xyzw", "trigger")


@dataclass
class RawTake:
    path: Path
    meta: dict
    poses: dict[str, np.ndarray]
    width: dict[str, np.ndarray]
    camera_s: np.ndarray

    def frame_path(self, index: int) -> Path:
        return self.path / "frames" / f"{index:06d}.jpg"

    def load_frame(self, index: int) -> np.ndarray:
        """RGB uint8 (h, w, 3)."""
        import cv2
        image = cv2.imread(str(self.frame_path(index)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.frame_path(index))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dataclass
class PoseSample:
    host_s: float
    client_ms: float
    epoch: int
    tracked: bool
    position: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    quat_xyzw: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 1.]))
    trigger: float = 0.0


class TakeWriter:
    """Buffers poses and width, streams frames to disk; save() finishes the folder, discard() removes it."""

    def __init__(self, path: Path, meta: dict, jpeg_quality: int = 92):
        self.path = Path(path)
        self.frames = self.path / "frames"
        self.frames.mkdir(parents=True, exist_ok=False)
        self.meta = dict(meta)
        self.meta["started_unix_s"] = time.time()
        self.jpeg_quality = int(jpeg_quality)
        self.poses: list[PoseSample] = []
        self.width: list[tuple[float, float, float]] = []
        self.camera_s: list[float] = []
        self.closed = False

    def add_pose(self, sample: PoseSample) -> None:
        self.poses.append(sample)

    def add_width(self, host_s: float, closure: float, raw: float) -> None:
        self.width.append((host_s, closure, raw))

    def add_frame(self, host_s: float, rgb: np.ndarray) -> None:
        import cv2
        index = len(self.camera_s)
        ok = cv2.imwrite(str(self.frames / f"{index:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                         [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise OSError(f"could not write frame {index} to {self.frames}")
        self.camera_s.append(host_s)

    @property
    def duration_s(self) -> float:
        stamps = [p.host_s for p in self.poses]
        return (max(stamps) - min(stamps)) if len(stamps) > 1 else 0.0

    def save(self) -> Path:
        self.meta["saved_unix_s"] = time.time()
        write_take(self.path, self.meta, self.poses, self.width, self.camera_s)
        self.closed = True
        return self.path

    def discard(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        self.closed = True


def write_take(path: Path, meta: dict, poses: list[PoseSample], width: list[tuple[float, float, float]],
               camera_s: list[float]) -> None:
    path = Path(path)
    (path / "frames").mkdir(parents=True, exist_ok=True)
    np.savez(path / "poses.npz",
             host_s=np.array([p.host_s for p in poses], dtype=float),
             client_ms=np.array([p.client_ms for p in poses], dtype=float),
             epoch=np.array([p.epoch for p in poses], dtype=np.int64),
             tracked=np.array([p.tracked for p in poses], dtype=bool),
             position=np.array([p.position for p in poses], dtype=float).reshape(-1, 3),
             quat_xyzw=np.array([p.quat_xyzw for p in poses], dtype=float).reshape(-1, 4),
             trigger=np.array([p.trigger for p in poses], dtype=float))
    w = np.array(width, dtype=float).reshape(-1, 3)
    np.savez(path / "width.npz", host_s=w[:, 0], closure=w[:, 1], raw=w[:, 2])
    np.savez(path / "camera.npz", host_s=np.array(camera_s, dtype=float))
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


def load_take(path: Path) -> RawTake:
    path = Path(path)
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"{path} is not a saved take (no meta.json)")
    with np.load(path / "poses.npz") as data:
        poses = {key: data[key] for key in POSE_FIELDS}
    with np.load(path / "width.npz") as data:
        width = {key: data[key] for key in ("host_s", "closure", "raw")}
    with np.load(path / "camera.npz") as data:
        camera_s = data["host_s"]
    return RawTake(path, json.loads((path / "meta.json").read_text()), poses, width, camera_s)


def find_takes(root: Path) -> list[Path]:
    """Every saved take under `root` (a take, a session or the raw root), in name order."""
    root = Path(root)
    if (root / "meta.json").is_file():
        return [root]
    return sorted(p.parent for p in root.rglob("meta.json"))


def next_take_path(session: Path) -> Path:
    session.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name.split("_")[1]) for p in session.glob("take_*") if p.name.split("_")[1].isdigit()]
    return session / f"take_{(max(taken) + 1) if taken else 0:04d}"
