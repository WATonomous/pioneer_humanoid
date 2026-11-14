"""Policy rollouts recorded for RECAP (``sim/evaluate.py --record``): a LeRobot dataset in the Quest teleop's
single-arm format (pioneer_8dof_dataset_schema.yaml, left arm, ego + wrist_left at 640x480, per-frame step instruction
as the task, ``subtask_index``), so rollouts, corrections and demos merge into one training set, plus the sidecar
(recap/sidecar.py) with each episode's origin and every frame's sim state.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from humanoid_robot_learning.recap.sidecar import FrameMeta, SidecarWriter, capture

RECORD_ARMS = ["left"]    # sim evaluation drives the left arm only
# Appended to the policy's prompt (the per-step instruction): pi*0.6's binarised advantage token.
ADVANTAGES = {"none": "", "positive": " Advantage: positive", "negative": " Advantage: negative"}


def advantage_prompt(task: str, advantage: str) -> str:
    """The prompt of an advantage-conditioned policy: `task` plus ADVANTAGES[advantage]."""
    return task + ADVANTAGES[advantage]


def record_schema(root: str | Path, robot: str):
    """(namespace for make_sim_recorder, RecordSchema) of the robot's default schema, left arm, writing to `root`."""
    from humanoid_robot_learning.sim.sim_teleop_record import load_record_schema

    ns = argparse.Namespace(record=True, schema=None, cameras=None, image_size=None, dataset_root=str(root),
                            num_episodes=None, task_description="policy rollout")
    return ns, load_record_schema(argparse.ArgumentParser(prog="--record"), ns, arms=RECORD_ARMS, robot=robot)


class RolloutRecorder:
    """Records rollouts one frame at a time; an episode is written (dataset, then sidecar) by finish().

    `steps`: whether the scene has a multi-step progress() (adds the subtask_index feature)."""

    def __init__(self, root: str | Path, robot: str, fps: int, max_s: float, steps: bool):
        from humanoid_robot_learning.sim.sim_teleop_record import make_sim_recorder

        ns, schema = record_schema(root, robot)
        # (height, width) of each recorded camera: the scene must mount these (SceneEnv's record_cameras).
        self.cameras = {name: (int(spec["height"]), int(spec["width"])) for name, spec in schema.images.items()}
        extra = {"subtask_index": ["subtask_index"]} if steps else None
        # One frame per policy step: the buffer holds the longest episode, plus a frame of margin.
        self.recorder, every = make_sim_recorder(ns, schema, device="cpu", sim_dt=1 / fps, extra_features=extra,
                                                 buffer_capacity_s=max_s + 1)
        if every != 1:
            raise ValueError(f"--fps {fps} differs from the schema's {self.recorder.fps}: rollouts would not merge with demos")
        self.sidecar = SidecarWriter(self.recorder.dataset_root, self.recorder.dataset.meta.total_episodes)
        self.frames: list[FrameMeta] = []

    def add(self, env, obs: dict[str, np.ndarray], action: np.ndarray, task: str) -> None:
        """One frame: `obs` (env.observe()) and the action taken from it; `task`: the plain step instruction."""
        meta = capture(env.model, env.data, env.progress, env.success)
        images = {name: obs[f"observation.images.{name}"] if env.cameras.get(name) == size else env.render(name, size)
                  for name, size in self.cameras.items()}
        extras = {"subtask_index": np.array([meta.progress], np.float32)} if env.progress is not None else None
        self.recorder.push_frame_to_buffer(action, obs["observation.state"], images, extras=extras, task=task)
        self.frames.append(meta)

    def finish(self, **row) -> int:
        """Save the episode (blocking: the sidecar row is written only once the dataset holds it); returns its index."""
        self.recorder.save_episode()
        self.recorder.wait_saved()
        episode = self.sidecar.write(self.frames, source="rollout", **row)
        self.frames = []
        return episode

    def close(self) -> None:
        self.recorder.finalize()
