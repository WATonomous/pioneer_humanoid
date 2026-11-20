"""Cut a synced take into the episodes that enter the dataset, as UMI's dataset plan does.

The take is split at every invalid step (tracking gap, missing or dropped frame) and wherever the pose segment
changes (a headset re-centre, or a jump in the raw poses: see sync.PoseTrack). The resampled steps are checked for
jumps once more as a backstop. Runs shorter than ``min_duration_s`` are dropped; each remaining run is one episode.
A policy therefore never trains across a gap or a jump, and one dropout costs only the moment around it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from umi_gripper import geometry
from umi_gripper.postprocess.sync import Episode


@dataclass
class Split:
    episodes: list[Episode]
    reasons: list[str] = field(default_factory=list)   # why steps or the whole take were dropped
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.episodes)


def speeds(episode: Episode) -> tuple[np.ndarray, np.ndarray]:
    """TCP linear (m/s) and angular (rad/s) speed between consecutive steps."""
    if len(episode) < 2:
        return np.zeros(0), np.zeros(0)
    dt = np.maximum(np.diff(episode.times), 1e-6)
    linear = np.linalg.norm(np.diff(episode.tcp[:, :3, 3], axis=0), axis=1) / dt
    angular = np.array([np.linalg.norm(geometry.rotation_vector(a[:3, :3].T @ b[:3, :3]))
                        for a, b in zip(episode.tcp[:-1], episode.tcp[1:])]) / dt
    return linear, angular


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of every run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def split_episode(episode: Episode, post: dict) -> Split:
    reasons = list(episode.issues)
    n = len(episode)
    stats: dict = {"steps_raw": n}
    if n == 0:
        reasons.append("no step where every stream has data")
        return Split([], reasons, stats)
    valid = episode.valid.copy()
    stats["invalid_steps"] = int((~valid).sum())

    # A cut between steps i and i + 1: the run before ends at i, the run after starts at i + 1.
    cut_after = np.zeros(n, dtype=bool)
    linear, angular = speeds(episode)
    pair_valid = valid[:-1] & valid[1:]
    jumps = pair_valid & ((linear > float(post["max_speed_m_s"])) | (angular > float(post["max_angular_speed_rad_s"])))
    cut_after[:-1] = jumps
    if jumps.any():
        reasons.append(f"{int(jumps.sum())} jumps between resampled steps: split there")
    if episode.segment is not None:
        changed = pair_valid & (episode.segment[1:] != episode.segment[:-1])
        cut_after[:-1] |= changed
        stats["segment_changes"] = int(changed.sum())
    stats["jumps"] = int(jumps.sum())
    if stats["invalid_steps"]:
        reasons.append(f"{stats['invalid_steps']} steps lost tracking or camera: split there")

    min_steps = int(np.ceil(float(post["min_duration_s"]) * float(post["fps"]))) + 1
    episodes, short = [], 0
    for start, stop in runs(valid):
        cuts = np.flatnonzero(cut_after[start:stop - 1]) + start + 1
        for a, b in zip(np.concatenate([[start], cuts]), np.concatenate([cuts, [stop]])):
            if b - a >= min_steps:
                episodes.append(episode.slice(int(a), int(b)))
            else:
                short += 1
    if short:
        reasons.append(f"{short} pieces shorter than {post['min_duration_s']} s dropped")
    stats["episodes"] = len(episodes)
    stats["steps_kept"] = int(sum(len(e) for e in episodes))
    stats["kept_fraction"] = round(stats["steps_kept"] / n, 3)
    if episodes:
        stats["duration_s"] = round(sum(float(e.times[-1] - e.times[0]) for e in episodes), 3)
    return Split(episodes, reasons, stats)
