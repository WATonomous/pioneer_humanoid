"""HDF5 trajectory writer (action, proprio, optional pixels, episode index)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_robot_learning.recording.schema import enabled_images
from humanoid_robot_learning.recording.snapshot import ObservationSnapshot

logger = logging.getLogger(__name__)


class Hdf5Sink:
    """
    Append episodes into a single .h5 file:
      action, proprio, pixels (optional), ep_len, ep_offset
    """

    def __init__(self, cfg: dict[str, Any], *, path: Path) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "h5py is required for HDF5 recording. "
                "Install with: pip install -e 'src/robot_learning[hdf5]'"
            ) from exc

        self._cfg = cfg
        self._path = path
        self._image_keys = list(enabled_images(cfg).keys())
        self._pixel_columns = self._pixel_column_names(self._image_keys)
        self._episode_buffers: dict[str, list] = {
            "action": [],
            "proprio": [],
        }
        for col in self._pixel_columns:
            self._episode_buffers[col] = []

        self._global: dict[str, list] = {
            "action": [],
            "proprio": [],
            "ep_len": [],
            "ep_offset": [],
        }
        for col in self._pixel_columns:
            self._global[col] = []

        self._total_steps = 0

    @staticmethod
    def _pixel_column_names(image_keys: list[str]) -> list[str]:
        if len(image_keys) == 1:
            return ["pixels"]
        return [f"pixels_{key}" for key in image_keys]

    @staticmethod
    def _pixel_column_for_key(image_keys: list[str], key: str) -> str:
        if len(image_keys) == 1:
            return "pixels"
        return f"pixels_{key}"

    @property
    def name(self) -> str:
        return "hdf5"

    @property
    def output_path(self) -> Path:
        return self._path

    def add_frame(self, snapshot: ObservationSnapshot, task: str) -> None:
        del task  # run metadata stored as HDF5 attrs on finalize
        if snapshot.state is None or snapshot.action is None:
            raise ValueError("HDF5 sink requires state and action on every frame")
        self._episode_buffers["proprio"].append(
            np.asarray(snapshot.state, dtype=np.float32)
        )
        self._episode_buffers["action"].append(
            np.asarray(snapshot.action, dtype=np.float32)
        )
        for key in self._image_keys:
            if key not in snapshot.images:
                raise ValueError(f"Missing image '{key}' for HDF5 frame")
            col = self._pixel_column_for_key(self._image_keys, key)
            img = np.asarray(snapshot.images[key], dtype=np.uint8)
            self._episode_buffers[col].append(img)

    def clear_episode(self) -> None:
        for buf in self._episode_buffers.values():
            buf.clear()

    def save_episode(self) -> None:
        n = len(self._episode_buffers["action"])
        if n == 0:
            logger.warning("HDF5 sink: skipping empty episode")
            return

        self._global["ep_offset"].append(self._total_steps)
        self._global["ep_len"].append(n)

        for key in ("action", "proprio"):
            self._global[key].extend(self._episode_buffers[key])
        for col in self._pixel_columns:
            self._global[col].extend(self._episode_buffers[col])

        self._total_steps += n
        self.clear_episode()
        logger.info("HDF5 sink: buffered episode (%s steps, total %s)", n, self._total_steps)

    def finalize(self) -> None:
        import h5py

        if self._total_steps == 0:
            logger.warning("HDF5 sink: no steps recorded; skipping file write")
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self._path, "w") as f:
            f.attrs["robot_id"] = self._cfg.get("robot_id", "unknown")
            f.attrs["fps"] = float(self._cfg.get("fps", 30))
            f.attrs["state_unit"] = self._cfg.get("state_unit", "rad")
            f.attrs["num_episodes"] = len(self._global["ep_len"])

            f.create_dataset(
                "action",
                data=np.stack(self._global["action"], axis=0),
                compression="gzip",
                compression_opts=3,
            )
            f.create_dataset(
                "proprio",
                data=np.stack(self._global["proprio"], axis=0),
                compression="gzip",
                compression_opts=3,
            )
            f.create_dataset(
                "ep_len",
                data=np.asarray(self._global["ep_len"], dtype=np.int32),
            )
            f.create_dataset(
                "ep_offset",
                data=np.asarray(self._global["ep_offset"], dtype=np.int32),
            )

            for col in self._pixel_columns:
                stacked = np.stack(self._global[col], axis=0)
                f.create_dataset(
                    col,
                    data=stacked,
                    compression="gzip",
                    compression_opts=3,
                )

        logger.info("HDF5 sink: wrote %s (%s steps)", self._path, self._total_steps)
