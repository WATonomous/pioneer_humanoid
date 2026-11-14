"""Carry RECAP sidecars (recap/sidecar.py) through a dataset merge (recording/merge_takes.py).

LeRobot's merge concatenates the inputs in order, so input i's episode e becomes episode offset_i + e, offset_i being
the episode count of the inputs before it. The sidecar rows and frame files are renumbered the same way. A correction's
parent (parent_episode in the dataset parent_root, "" = its own) is renumbered when that dataset is among the inputs,
and its parent_root cleared: the parent is then in the merged dataset. Inputs without a sidecar (teleop demos)
contribute no rows; recap/data.py treats their episodes as demos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from humanoid_robot_learning.recap import sidecar


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _renumber(rows, offset: int, own: Path, offsets: dict[Path, int]):
    """Input rows with merged episode numbers (module docstring)."""
    rows = rows.copy()
    rows["episode_index"] = rows["episode_index"].astype(int) + offset
    for i in np.flatnonzero(rows["source"].to_numpy() == "correction"):
        root = str(rows["parent_root"].iloc[i])
        parent_root = own if root == "" else _resolve(root)
        if parent_root in offsets:
            rows.iloc[i, rows.columns.get_loc("parent_episode")] = int(rows["parent_episode"].iloc[i]) + offsets[parent_root]
            rows.iloc[i, rows.columns.get_loc("parent_root")] = ""
        elif root == "":
            raise ValueError(f"{own}: correction with an empty parent_root, but {own} is not an input")
    return rows


def _copy_frames(src: Path, out: Path, old: int, new: int) -> None:
    frames = sidecar.read_frames(src, old)
    frames["episode_index"] = new
    path = sidecar.frames_path(out, new)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(path, index=False)


def merge_sidecars(inputs: list[Path], episode_counts: list[int], out: Path) -> int:
    """Write the merged sidecar of `inputs` (in merge order, with their episode counts) to `out`; returns its rows."""
    import pandas as pd

    starts = np.concatenate([[0], np.cumsum(episode_counts)[:-1]]).astype(int)
    offsets = {_resolve(p): int(o) for p, o in zip(inputs, starts)}
    merged = []
    for src, offset in zip(inputs, starts):
        rows = sidecar.read_episodes(src)
        if not len(rows):
            continue
        for old in rows["episode_index"].astype(int):
            _copy_frames(src, out, old, old + int(offset))
        merged.append(_renumber(rows, int(offset), _resolve(src), offsets))
    if not merged:
        return 0
    table = pd.concat(merged, ignore_index=True)
    sidecar.recap_dir(out).mkdir(parents=True, exist_ok=True)
    table.to_parquet(sidecar.recap_dir(out) / "episodes.parquet", index=False)
    return len(table)
