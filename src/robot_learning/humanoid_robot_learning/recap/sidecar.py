"""RECAP sidecar files: where each episode of a LeRobot dataset came from, and the full sim state of every frame.

    <dataset>/recap/episodes.parquet                  one row per episode (EPISODE_COLUMNS)
    <dataset>/recap/frames/episode_<NNNNNN>.parquet   one row per frame: episode_index, frame_index, time, progress,
                                                      success, state (mj_getState with STATE_SPEC)

They sit beside the dataset rather than in it, so the dataset keeps the teleop schema and merges with teleop demos.
Frames get one file per episode: parquet cannot append to a file, so one frames.parquet would be rewritten whole on
every episode. Rows are written only for frames the recorder actually saved, so frame_index matches the dataset's.

Every frame is snapshotted. drawer_stow's state is 320 float64 (2.5 KB), 2.6 MB for a 40 s episode at 25 fps before
compression, small next to its two 640x480 camera videos. A correction can then resume exactly at any frame, without
replaying actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np

# qpos, qvel, act, ctrl, applied forces, mocap, userdata (where scenes latch their step counter; drawer_stow also
# keeps its colour order there) and the solver warmstart: restoring it continues the rollout bit for bit.
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
SOURCES = ("rollout", "correction", "demo")
NO_PROGRESS = -1    # progress column of a scene without a multi-step progress()
# episodes.parquet columns and the value a source leaves unset.
EPISODE_COLUMNS = {
    "episode_index": -1,
    "source": "",           # one of SOURCES
    "round": 0,             # RECAP iteration that produced it
    "scene": "",
    "policy": "",           # checkpoint path (rollouts)
    "advantage": "none",    # the prompt's advantage suffix during the rollout
    "seed": -1,             # layout seed (corrections: the parent's)
    "success": False,
    "sim_seconds": 0.0,     # episode length in sim time
    "num_frames": 0,
    "parent_episode": -1,   # corrections: the rollout episode they resume
    "parent_root": "",      # corrections: the dataset holding the parent, when not this one
    "resume_s": math.nan,   # corrections: the parent's time they resume from
    "state_spec": int(STATE_SPEC),
}


@dataclass
class FrameMeta:
    """What the sidecar keeps of one recorded frame, captured at the frame's instant (before its action)."""

    time: float
    progress: int
    success: bool
    state: np.ndarray


def capture(model: mujoco.MjModel, data: mujoco.MjData, progress=None, success=None) -> FrameMeta:
    """Snapshot `data`; `progress` / `success`: the scene's functions (humanoid_mujoco_scenes), or None."""
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    k = progress(model, data)[0] if progress is not None else NO_PROGRESS
    return FrameMeta(float(data.time), int(k), bool(success(model, data)) if success is not None else False, state)


def restore(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray) -> None:
    """Put `data` back at a snapshot taken by capture(); derived quantities are recomputed."""
    if len(state) != mujoco.mj_stateSize(model, STATE_SPEC):
        raise ValueError(f"snapshot has {len(state)} values, this model's state {mujoco.mj_stateSize(model, STATE_SPEC)}: "
                         "a different scene or robot")
    mujoco.mj_setState(model, data, np.asarray(state, np.float64), STATE_SPEC)
    mujoco.mj_forward(model, data)


def recap_dir(root: str | Path) -> Path:
    return Path(root) / "recap"


def frames_path(root: str | Path, episode: int) -> Path:
    return recap_dir(root) / "frames" / f"episode_{episode:06d}.parquet"


class SidecarWriter:
    """Appends episodes to a dataset's sidecar. `first_episode`: the dataset's episode count when recording started
    (the recorder saves episodes in order, so the n-th written here is dataset episode first_episode + n)."""

    def __init__(self, root: str | Path, first_episode: int):
        self.root, self.next_episode = Path(root), int(first_episode)

    def write(self, frames: list[FrameMeta], **row) -> int:
        """Write one episode's frames and its episodes.parquet row (EPISODE_COLUMNS keys); returns its index."""
        import pandas as pd

        unknown = set(row) - set(EPISODE_COLUMNS)
        if unknown or row.get("source") not in SOURCES:
            raise ValueError(f"bad episode row: unknown columns {sorted(unknown)}, source {row.get('source')!r}")
        episode = self.next_episode
        episodes = read_episodes(self.root)
        if episode in set(episodes["episode_index"]):
            raise ValueError(f"{recap_dir(self.root)}: episode {episode} already has a row: sidecar and dataset disagree")
        _write_frames(frames_path(self.root, episode), episode, frames)
        full = {**EPISODE_COLUMNS, **row, "episode_index": episode, "num_frames": len(frames),
                "sim_seconds": frames[-1].time - frames[0].time if frames else 0.0}
        table = pd.concat([episodes, pd.DataFrame([full])], ignore_index=True) if len(episodes) else pd.DataFrame([full])
        _atomic_parquet(table, recap_dir(self.root) / "episodes.parquet")
        self.next_episode += 1
        return episode


def _write_frames(path: Path, episode: int, frames: list[FrameMeta]) -> None:
    import pandas as pd

    table = pd.DataFrame({
        "episode_index": episode,
        "frame_index": np.arange(len(frames)),
        "time": [f.time for f in frames],
        "progress": [f.progress for f in frames],
        "success": [f.success for f in frames],
        "state": [f.state for f in frames],
    })
    _atomic_parquet(table, path)


def _atomic_parquet(table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_episodes(root: str | Path):
    """episodes.parquet as a DataFrame (empty, with EPISODE_COLUMNS, when there is none yet)."""
    import pandas as pd

    path = recap_dir(root) / "episodes.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=list(EPISODE_COLUMNS))


def read_frames(root: str | Path, episode: int):
    """One episode's frame rows; `state` holds float64 arrays ready for restore()."""
    import pandas as pd

    return pd.read_parquet(frames_path(root, episode))


def frame_at(frames, t: float) -> int:
    """Index of the last frame at or before sim time `t` (the first frame if `t` precedes it)."""
    times = frames["time"].to_numpy()
    return int(max(0, np.searchsorted(times, t, side="right") - 1))


def last_advance_time(frames) -> float:
    """Sim time of the frame where the progress counter last went up (the first frame's time if it never did)."""
    progress = frames["progress"].to_numpy()
    rises = np.flatnonzero(np.diff(progress) > 0)
    return float(frames["time"].iloc[rises[-1] + 1 if len(rises) else 0])
