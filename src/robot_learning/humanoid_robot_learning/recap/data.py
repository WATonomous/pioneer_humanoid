"""RECAP training data: every frame of a dataset with its episode's origin, Monte Carlo return and value inputs.

Episodes come from the sidecar (recap/sidecar.py). Dataset episodes without a sidecar row are teleop demos (the
Quest demo recorder writes no sidecar): source "demo", successful, round 0, and no sim snapshot.

Value inputs, per frame (task-agnostic: the scene and robot only size the snapshot):

    observation.state     the arm joints + gripper, as recorded (every source)
    progress              the scene's step index (sidecar `progress`, else the dataset's `subtask_index`)
    elapsed               sim time / time limit (demos: frame_index / fps)
    privileged            from the sim snapshot: every qpos (objects, drawer, arms) and userdata (latched step, the
                          drawer_stow colour order); zeros for demos, which have no snapshot
    has_snapshot          1 when `privileged` is real

Return (pi*0.6, arXiv 2511.14759 sec. IV): reward -1 per step, 0 at a successful end, a failure ends with -T (T = time
limit in steps). Normalised by T: G_t = -(steps left) / T in (-1, 0] on success, and every failing frame clips to -1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from humanoid_robot_learning.recap import sidecar

# mjtState bits in mj_getState's order: the fields STATE_SPEC packs, in sequence.
_STATE_BITS = sorted({int(m) for m in mujoco.mjtState.__members__.values() if int(m) > 0 and int(m) & (int(m) - 1) == 0})


@dataclass
class Frames:
    """All frames of a dataset, in dataset order (arrays have one row per frame)."""

    index: np.ndarray           # global dataset index
    episode: np.ndarray
    frame: np.ndarray
    task_index: np.ndarray
    source: np.ndarray          # sidecar.SOURCES
    returns: np.ndarray         # normalised Monte Carlo return, [-1, 0]
    inputs: np.ndarray          # value inputs (float32), see the module docstring
    episodes: object            # pandas DataFrame, one row per dataset episode (sidecar.EPISODE_COLUMNS)


class SnapshotDecoder:
    """Reads qpos and userdata out of mj_getState(STATE_SPEC) vectors of one scene + robot."""

    def __init__(self, scene: str, robot: str):
        from humanoid_mujoco_scenes import make_model

        model = make_model(scene, robot=robot)
        offset, self.slices = 0, {}
        for bit in _STATE_BITS:
            if int(sidecar.STATE_SPEC) & bit:
                size = mujoco.mj_stateSize(model, mujoco.mjtState(bit))
                self.slices[bit] = slice(offset, offset + size)
                offset += size
        self.size = mujoco.mj_stateSize(model, sidecar.STATE_SPEC)
        if offset != self.size:
            raise RuntimeError(f"state fields add up to {offset}, mj_stateSize says {self.size}")
        self.dim = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_QPOS) + model.nuserdata

    def __call__(self, states: np.ndarray) -> np.ndarray:
        """(frames, state size) snapshots -> (frames, dim) privileged features."""
        if states.shape[1] != self.size:
            raise ValueError(f"snapshots have {states.shape[1]} values, this scene + robot {self.size}: wrong --scene/--robot")
        qpos = states[:, self.slices[int(mujoco.mjtState.mjSTATE_QPOS)]]
        userdata = states[:, self.slices[int(mujoco.mjtState.mjSTATE_USERDATA)]]
        return np.concatenate([qpos, userdata], axis=1)


def episode_table(root: Path, lengths: dict[int, int]):
    """Sidecar rows of the dataset's episodes; episodes without one become demo rows (successful, round 0)."""
    import pandas as pd

    rows = sidecar.read_episodes(root).set_index("episode_index", drop=False)
    missing = [e for e in sorted(lengths) if e not in rows.index]
    demos = pd.DataFrame([{**sidecar.EPISODE_COLUMNS, "episode_index": e, "source": "demo", "success": True,
                           "num_frames": lengths[e]} for e in missing])
    table = pd.concat([rows, demos], ignore_index=True) if len(demos) else rows.reset_index(drop=True)
    return table.set_index("episode_index", drop=False).sort_index()


def read_dataset_frames(root: Path):
    """The dataset's frames (data/*/*.parquet), sorted by episode then frame."""
    import pandas as pd

    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"{root}: no data/*/*.parquet (not a LeRobot v3 dataset?)")
    frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return frames.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def mc_returns(num_frames: int, success: bool, max_steps: int) -> np.ndarray:
    """Normalised Monte Carlo return of every frame of one episode (module docstring)."""
    if not success:
        return np.full(num_frames, -1.0)
    steps_left = np.arange(num_frames - 1, -1, -1, dtype=np.float64)
    return np.clip(-steps_left / max_steps, -1.0, 0.0)


def _column(frames, name: str) -> np.ndarray:
    return np.stack([np.asarray(v, np.float64).reshape(-1) for v in frames[name]])


def episode_inputs(root: Path, episode: int, rows, decoder: SnapshotDecoder, fps: int, max_s: float) -> np.ndarray:
    """Value inputs of one episode's dataset rows (module docstring)."""
    arm = _column(rows, "observation.state")
    snap = sidecar.frames_path(root, episode)
    if snap.exists():
        side = sidecar.read_frames(root, episode)
        if len(side) != len(rows):
            raise ValueError(f"{root} episode {episode}: {len(rows)} dataset frames but {len(side)} sidecar frames")
        privileged, has = decoder(np.stack(side["state"].to_numpy())), np.ones((len(rows), 1))
        progress, time = side["progress"].to_numpy(np.float64), side["time"].to_numpy(np.float64)
    else:
        privileged, has = np.zeros((len(rows), decoder.dim)), np.zeros((len(rows), 1))
        progress = _column(rows, "subtask_index")[:, 0] if "subtask_index" in rows else np.full(len(rows), -1.0)
        time = rows["frame_index"].to_numpy(np.float64) / fps
    return np.column_stack([arm, progress, time / max_s, privileged, has]).astype(np.float32)


def load_frames(root: str | Path, scene: str, robot: str, fps: int, max_s: float) -> Frames:
    """Every frame of the dataset at `root` with its return and value inputs; `max_s`: the rollouts' time limit."""
    root = Path(root)
    data = read_dataset_frames(root)
    lengths = data.groupby("episode_index").size().to_dict()
    episodes = episode_table(root, lengths)
    decoder, max_steps = SnapshotDecoder(scene, robot), int(round(max_s * fps))
    inputs, returns, sources = [], [], []
    for episode, rows in data.groupby("episode_index", sort=True):
        row = episodes.loc[episode]
        inputs.append(episode_inputs(root, int(episode), rows, decoder, fps, max_s))
        returns.append(mc_returns(len(rows), bool(row["success"]), max_steps))
        sources.append(np.full(len(rows), row["source"], dtype=object))
    return Frames(index=data["index"].to_numpy(), episode=data["episode_index"].to_numpy(),
                  frame=data["frame_index"].to_numpy(), task_index=data["task_index"].to_numpy(),
                  source=np.concatenate(sources), returns=np.concatenate(returns),
                  inputs=np.concatenate(inputs), episodes=episodes)
