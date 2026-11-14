"""Raw episode recording shared by expert demos and expert corrections.

Raw layout, read by ``humanoid_robot_learning.raw_to_lerobot``::

    <raw>/manifest.json                 fps, joint_names, task, cameras, results[...]
    <raw>/episode_<id>/trajectory.npz   observation.state, action, phase, timestamp, peg/hole poses
    <raw>/episode_<id>/handoff.npz      optional MuJoCo state the episode started from
    <raw>/episode_<id>/<camera>/<frame>.png

Frames stream to ``.attempt_<id>`` and the directory is renamed only after the episode is
verified, so an interrupted or failed attempt never looks like a demonstration.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from .expert import FPS, JOINT_NAMES, TASK

CAMERAS = {"ego": (480, 640), "wrist_left": (480, 640)}  # (height, width) used by every trained policy


def write_json(path, value):
    """Atomic JSON write: readers never observe a half-written manifest or report."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class RawDataset:
    """One raw collection directory; reopening it resumes from its manifest."""

    def __init__(self, root, cameras=CAMERAS, **metadata):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cameras = dict(cameras)
        path = self.root / "manifest.json"
        if path.exists():
            self.manifest = json.loads(path.read_text())
            if self.manifest["cameras"] != {n: list(hw) for n, hw in self.cameras.items()}:
                raise ValueError(f"{root} was recorded with different cameras")
        else:
            self.manifest = {"fps": FPS, "joint_names": JOINT_NAMES, "task": TASK,
                             "cameras": {n: list(hw) for n, hw in self.cameras.items()},
                             "mujoco_version": mujoco.__version__, **metadata, "results": []}

    @property
    def results(self):
        return self.manifest["results"]

    def successes(self):
        return sum(bool(r["success"]) for r in self.results)

    def has(self, key, value):
        return any(r.get(key) == value and r["success"] for r in self.results)

    def add(self, result):
        self.results.append(result)
        write_json(self.root / "manifest.json", self.manifest)
        print(json.dumps(result), flush=True)

    def episode(self, episode_id, renderers):
        return EpisodeWriter(self.root, episode_id, renderers)


class EpisodeWriter:
    """``PegExpert.on_frame`` callback that records one episode; use as a context manager."""

    def __init__(self, root, episode_id, renderers):
        self.staging = root / f".attempt_{episode_id:06d}"
        self.final = root / f"episode_{episode_id:06d}"
        self.renderers = renderers
        self.frames = []
        self.committed = False

    def __enter__(self):
        shutil.rmtree(self.staging, ignore_errors=True)  # interrupted staging only, never a committed episode
        for name in self.renderers:
            (self.staging / name).mkdir(parents=True)
        return self

    def __exit__(self, *exc):
        if not self.committed:
            shutil.rmtree(self.staging, ignore_errors=True)
        return False

    def __call__(self, expert, action):
        from PIL import Image

        data = expert.data
        self.frames.append({"observation.state": expert.state(), "action": np.array(action, copy=True),
                            "phase": str(expert.phase), "timestamp": data.time,
                            "peg_position": expert.peg_pos, "hole_position": expert.hole_pos,
                            "peg_rotation": data.xmat[expert.peg].reshape(3, 3).copy()})
        for name, renderer in self.renderers.items():
            renderer.update_scene(data, camera=name)
            Image.fromarray(renderer.render()).save(self.staging / name / f"{len(self.frames) - 1:06d}.png")

    def commit(self, handoff=None):
        if not self.frames:
            raise ValueError("cannot commit an empty episode")
        np.savez_compressed(self.staging / "trajectory.npz", **{k: np.array([f[k] for f in self.frames]) for k in self.frames[0]})
        if handoff is not None:
            np.savez_compressed(self.staging / "handoff.npz", **handoff)
        self.staging.rename(self.final)
        self.committed = True
        return {"frames": len(self.frames), "phase_frames": dict(Counter(str(f["phase"]) for f in self.frames))}


def snapshot(expert):
    """Everything needed to restore a physical state later (block placement is model data)."""
    data = expert.data
    return {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy(),
            "sim_time": float(data.time), "hole_body_pos": expert.model.body_pos[expert.block].copy()}
