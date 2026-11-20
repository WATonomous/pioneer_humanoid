"""One raw take -> one episode on a 25 fps grid: TCP pose, jaw closure and the nearest camera frame per step.

Each stream is first moved onto the host clock and back by its latency (config ``latency``): headset times through
clock.headset_to_host, then ``- latency.pose_s``; camera and encoder host stamps ``- camera_s`` / ``- width_s``.
The grid runs where every stream has data. A grid step is invalid when the controller was untracked across it for
longer than ``max_tracking_gap_s`` or no camera frame lies within ``max_camera_gap_s / 2``; quality.py decides what
an invalid step means for the take.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from umi_gripper import config as umi_config
from umi_gripper import geometry
from umi_gripper.clock import headset_to_host
from umi_gripper.take import RawTake


@dataclass
class Episode:
    take: RawTake
    times: np.ndarray            # (n,) host clock, s
    tcp: np.ndarray              # (n, 4, 4) T_world_tcp
    closure: np.ndarray          # (n,) 0 open .. 1 closed
    frame_index: np.ndarray | None   # (n,) camera frame per step, -1 = none
    valid: np.ndarray            # (n,) bool
    issues: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.times)

    def trimmed(self, start: int, stop: int) -> "Episode":
        s = slice(start, stop)
        return Episode(self.take, self.times[s], self.tcp[s], self.closure[s],
                       None if self.frame_index is None else self.frame_index[s], self.valid[s], list(self.issues))


def pose_track(take: RawTake, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, positions, quats) of the TRACKED grip samples, on the host clock minus latency.pose_s."""
    poses = take.poses
    if len(poses["host_s"]) == 0:
        return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 4))
    if len(np.unique(poses["epoch"])) > 1:
        # A reference-space reset (recentre) mid-take moves the world frame: the two halves don't line up.
        raise ValueError("the headset re-centred during the take (reference epoch changed)")
    times = headset_to_host(poses["client_ms"], poses["host_s"]) - float(cfg["latency"]["pose_s"])
    keep = poses["tracked"].astype(bool)
    return times[keep], poses["position"][keep], poses["quat_xyzw"][keep]


def interpolate_poses(sample_t, positions, quats, query_t, max_gap_s):
    """Grip pose at each query time (linear position, slerp rotation) and whether it is valid.

    Valid: both bracketing tracked samples exist and are at most max_gap_s apart (a longer gap = tracking loss).
    """
    n = len(query_t)
    out = np.tile(np.eye(4), (n, 1, 1))
    valid = np.zeros(n, dtype=bool)
    if len(sample_t) < 2:
        return out, valid
    upper = np.searchsorted(sample_t, query_t, side="left")   # first sample at or after each query
    for i, (t, hi) in enumerate(zip(query_t, upper)):
        if hi < len(sample_t) and sample_t[hi] == t:
            lo = hi
        elif hi == 0 or hi == len(sample_t):
            continue   # before the first or after the last tracked sample
        else:
            lo = hi - 1
        if sample_t[hi] - sample_t[lo] > max_gap_s:
            continue
        span = sample_t[hi] - sample_t[lo]
        fraction = 0.0 if span <= 0 else (t - sample_t[lo]) / span
        position = positions[lo] + fraction * (positions[hi] - positions[lo])
        quat = geometry.slerp(quats[lo], quats[hi], fraction)
        out[i] = geometry.make_pose(geometry.quat_to_matrix(quat), position)
        valid[i] = True
    return out, valid


def nearest_frames(frame_t: np.ndarray, query_t: np.ndarray, max_offset_s: float) -> np.ndarray:
    """Index of the camera frame nearest each query time, -1 when the nearest is further than max_offset_s."""
    if len(frame_t) == 0:
        return np.full(len(query_t), -1)
    upper = np.clip(np.searchsorted(frame_t, query_t), 0, len(frame_t) - 1)
    lower = np.clip(upper - 1, 0, len(frame_t) - 1)
    best = np.where(np.abs(frame_t[lower] - query_t) <= np.abs(frame_t[upper] - query_t), lower, upper).astype(int)
    best[np.abs(frame_t[best] - query_t) > max_offset_s] = -1
    return best


def sync_take(take: RawTake, cfg: dict) -> Episode:
    post = cfg["postprocess"]
    fps = float(post["fps"])
    issues: list[str] = []
    try:
        pose_t, positions, quats = pose_track(take, cfg)
    except ValueError as exc:
        pose_t, positions, quats = np.zeros(0), np.zeros((0, 3)), np.zeros((0, 4))
        issues.append(str(exc))
    width_t = take.width["host_s"] - float(cfg["latency"]["width_s"])
    has_camera = bool(take.meta.get("camera", len(take.camera_s) > 0))
    frame_t = take.camera_s - float(cfg["latency"]["camera_s"])

    starts, stops = [], []
    for stamps in (pose_t, width_t) + ((frame_t,) if has_camera else ()):
        if len(stamps) < 2:
            issues.append("a stream has fewer than 2 samples")
            return Episode(take, np.zeros(0), np.zeros((0, 4, 4)), np.zeros(0),
                           np.zeros(0, dtype=int) if has_camera else None, np.zeros(0, dtype=bool), issues)
        starts.append(stamps[0])
        stops.append(stamps[-1])
    start, stop = max(starts), min(stops)
    times = start + np.arange(int(np.floor((stop - start) * fps)) + 1) / fps if stop > start else np.zeros(0)

    grip, pose_ok = interpolate_poses(pose_t, positions, quats, times, float(post["max_tracking_gap_s"]))
    tcp = grip @ umi_config.grip_to_tcp(cfg)
    order = np.argsort(width_t, kind="stable")
    closure = np.interp(times, width_t[order], take.width["closure"][order])
    valid = pose_ok.copy()
    frame_index = None
    if has_camera:
        frame_index = nearest_frames(frame_t, times, float(post["max_camera_gap_s"]) / 2)
        valid &= frame_index >= 0
    return Episode(take, times, tcp, closure, frame_index, valid, issues)


def episode_vectors(episode: Episode) -> tuple[np.ndarray, np.ndarray]:
    """(state, action), each (n, 10): TCP position (3) + rot6d (6) in the episode's FIRST TCP frame, then closure.

    Start-relative, so the headset's world frame (which re-centring moves) never enters the data, and the pose of
    step t relative to step s is still exact: inv(T0^-1 T_s) (T0^-1 T_t) = T_s^-1 T_t (UMI's relative actions).
    action[t] = state[t + 1] (the pose the hand went to next); the last step repeats its own state.
    """
    if len(episode) == 0:
        return np.zeros((0, 10), np.float32), np.zeros((0, 10), np.float32)
    origin = geometry.invert(episode.tcp[0])
    state = np.array([np.concatenate([geometry.pose_to_vector(origin @ pose), [c]])
                      for pose, c in zip(episode.tcp, episode.closure)], dtype=np.float32)
    action = np.concatenate([state[1:], state[-1:]], axis=0)
    return state, action
