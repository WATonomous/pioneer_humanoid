"""One raw take -> one timeline at 25 fps: a camera frame per step, with the TCP pose and jaw read at that frame's time.

Each stream is first moved onto the host clock and back by its latency (config ``latency``): headset times through
clock.headset_to_host, then ``- latency.pose_s``; camera and encoder host stamps ``- camera_s`` / ``- width_s``.

Steps are picked on a 25 fps grid over the span every stream covers. With a camera, each grid step takes its
nearest frame and the step's time becomes that frame's own (corrected) capture time, so image, pose and jaw describe
the same instant (UMI and FastUMI tie poses to frame times the same way). That keeps steps evenly spaced only when
the camera runs at a multiple of 25 fps (camera.fps 25 or 50); sync reports anything else. The pose is interpolated between the two
tracked samples around that time (linear position, slerp rotation).

A step is invalid when:
  - the tracked samples around it are more than ``max_tracking_gap_s`` apart, or in different PoseTrack segments
    (a re-centre or a jump between them; quality.py also cuts wherever the segment changes);
  - no camera frame lies within ``max_frame_offset_s`` of its grid time, or its frame was already used by the
    previous step (a dropped frame).
quality.py splits the timeline at invalid steps.
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
    times: np.ndarray            # (n,) host clock, s: the frame capture times (grid times without a camera)
    tcp: np.ndarray              # (n, 4, 4) T_world_tcp
    closure: np.ndarray          # (n,) 0 open .. 1 closed
    frame_index: np.ndarray | None   # (n,) camera frame per step, -1 = none
    valid: np.ndarray            # (n,) bool
    segment: np.ndarray | None = None   # (n,) PoseTrack segment per step (-1 = no pose); a change = re-centre / jump
    issues: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.times)

    def slice(self, start: int, stop: int) -> "Episode":
        s = slice(start, stop)
        return Episode(self.take, self.times[s], self.tcp[s], self.closure[s],
                       None if self.frame_index is None else self.frame_index[s], self.valid[s],
                       None if self.segment is None else self.segment[s], list(self.issues))


@dataclass
class PoseTrack:
    """The tracked grip samples of a take, on the host clock minus latency.pose_s.

    ``segments`` numbers the stretches the pose is continuous over: it increments wherever the headset re-centred
    (reference epoch changed) or the pose jumped (raw sample-to-sample speed over the postprocess limits: the
    tracker relocalising). Nothing is interpolated across a segment change, and quality.py cuts there.
    """
    times: np.ndarray            # (n,)
    positions: np.ndarray        # (n, 3)
    quats: np.ndarray            # (n, 4) xyzw
    segments: np.ndarray         # (n,)
    jumps: int = 0


def pose_track(take: RawTake, cfg: dict) -> PoseTrack:
    poses = take.poses
    if len(poses["host_s"]) == 0:
        return PoseTrack(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 4)), np.zeros(0, dtype=int))
    times = headset_to_host(poses["client_ms"], poses["host_s"]) - float(cfg["latency"]["pose_s"])
    keep = poses["tracked"].astype(bool)
    times, positions, quats, epochs = times[keep], poses["position"][keep], poses["quat_xyzw"][keep], poses["epoch"][keep]
    breaks = np.zeros(len(times), dtype=bool)
    jumps = 0
    if len(times) > 1:
        post = cfg["postprocess"]
        dt = np.maximum(np.diff(times), 1e-3)
        linear = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
        dots = np.abs(np.sum(quats[1:] * quats[:-1], axis=1)) / (
            np.linalg.norm(quats[1:], axis=1) * np.linalg.norm(quats[:-1], axis=1))
        angular = 2 * np.arccos(np.clip(dots, -1.0, 1.0)) / dt
        jump = (linear > float(post["max_speed_m_s"])) | (angular > float(post["max_angular_speed_rad_s"]))
        jumps = int(jump.sum())
        breaks[1:] = jump | (epochs[1:] != epochs[:-1])
    return PoseTrack(times, positions, quats, np.cumsum(breaks), jumps)


def interpolate_poses(track: PoseTrack, query_t: np.ndarray, max_gap_s: float,
                      return_segments: bool = False):
    """Grip pose (n, 4, 4) at each query time and whether it is valid (+ each step's segment, -1 if invalid).

    Valid: tracked samples exist on both sides, at most max_gap_s apart and in the same segment.
    """
    t = track.times
    n = len(query_t)
    out = np.tile(np.eye(4), (n, 1, 1))
    valid = np.zeros(n, dtype=bool)
    segments = np.full(n, -1, dtype=np.int64)
    if len(t) < 2:
        return (out, valid, segments) if return_segments else (out, valid)
    upper = np.searchsorted(t, query_t, side="left")   # first sample at or after each query
    for i, (q, hi) in enumerate(zip(query_t, upper)):
        if hi < len(t) and t[hi] == q:
            lo = hi
        elif hi == 0 or hi == len(t):
            continue   # before the first or after the last tracked sample
        else:
            lo = hi - 1
        if t[hi] - t[lo] > max_gap_s or track.segments[hi] != track.segments[lo]:
            continue
        span = t[hi] - t[lo]
        fraction = 0.0 if span <= 0 else (q - t[lo]) / span
        position = track.positions[lo] + fraction * (track.positions[hi] - track.positions[lo])
        quat = geometry.slerp(track.quats[lo], track.quats[hi], fraction)
        out[i] = geometry.make_pose(geometry.quat_to_matrix(quat), position)
        valid[i] = True
        segments[i] = track.segments[lo]
    return (out, valid, segments) if return_segments else (out, valid)


def nearest_frames(frame_t: np.ndarray, query_t: np.ndarray, max_offset_s: float) -> np.ndarray:
    """Index of the camera frame nearest each query time, -1 when the nearest is further than max_offset_s."""
    if len(frame_t) == 0:
        return np.full(len(query_t), -1)
    upper = np.clip(np.searchsorted(frame_t, query_t), 0, len(frame_t) - 1)
    lower = np.clip(upper - 1, 0, len(frame_t) - 1)
    best = np.where(np.abs(frame_t[lower] - query_t) <= np.abs(frame_t[upper] - query_t), lower, upper).astype(int)
    best[np.abs(frame_t[best] - query_t) > max_offset_s] = -1
    return best


def _empty(take: RawTake, has_camera: bool, issues: list[str]) -> Episode:
    return Episode(take, np.zeros(0), np.zeros((0, 4, 4)), np.zeros(0),
                   np.zeros(0, dtype=int) if has_camera else None, np.zeros(0, dtype=bool), issues)


def sync_take(take: RawTake, cfg: dict) -> Episode:
    post = cfg["postprocess"]
    fps = float(post["fps"])
    issues: list[str] = []
    has_camera = bool(take.meta.get("camera", len(take.camera_s) > 0))
    try:
        track = pose_track(take, cfg)
    except ValueError as exc:   # e.g. the headset page restarted mid-take: its clock went backwards
        return _empty(take, has_camera, [str(exc)])
    width_t = take.width["host_s"] - float(cfg["latency"]["width_s"])
    frame_t = take.camera_s - float(cfg["latency"]["camera_s"])

    starts, stops = [], []
    for stamps in (track.times, width_t) + ((frame_t,) if has_camera else ()):
        if len(stamps) < 2:
            issues.append("a stream has fewer than 2 samples")
            return _empty(take, has_camera, issues)
        starts.append(stamps[0])
        stops.append(stamps[-1])
    start, stop = max(starts), min(stops)
    grid = start + np.arange(int(np.floor((stop - start) * fps)) + 1) / fps if stop > start else np.zeros(0)

    frame_index = None
    times = grid
    frame_ok = np.ones(len(grid), dtype=bool)
    if has_camera:
        frame_index = nearest_frames(frame_t, grid, float(post["max_frame_offset_s"]))
        repeated = np.concatenate([[False], (frame_index[1:] == frame_index[:-1]) & (frame_index[1:] >= 0)])
        frame_index[repeated] = -1
        frame_ok = frame_index >= 0
        times = np.where(frame_ok, frame_t[np.maximum(frame_index, 0)], grid)
        period = float(np.median(np.diff(frame_t)))
        ratio = period * fps
        if not any(abs(ratio - 1 / k) < 0.05 for k in (1, 2, 3, 4)):
            issues.append(f"camera runs at {1 / period:.1f} fps, not a multiple of {fps:g}: steps fall unevenly "
                          f"between frames; set camera.fps to {fps:g} or {2 * fps:g}")

    grip, pose_ok, segments = interpolate_poses(track, times, float(post["max_tracking_gap_s"]), return_segments=True)
    if track.jumps:
        issues.append(f"{track.jumps} TCP jumps in the raw poses (> {post['max_speed_m_s']} m/s or "
                      f"> {post['max_angular_speed_rad_s']} rad/s, the tracker relocalising): split there")
    tcp = grip @ umi_config.grip_to_tcp(cfg)
    order = np.argsort(width_t, kind="stable")
    closure = np.interp(times, width_t[order], take.width["closure"][order])
    return Episode(take, times, tcp, closure, frame_index, pose_ok & frame_ok, segments, issues)


def gripper_width_m(closure: np.ndarray, cfg: dict) -> np.ndarray:
    """Jaw gap in metres (UMI's gripper representation): linear in the encoder angle, max_width_m fully open."""
    return (1.0 - np.asarray(closure, dtype=float)) * float(cfg["width"]["max_width_m"])


def episode_vectors(episode: Episode) -> tuple[np.ndarray, np.ndarray]:
    """(state, action), each (n, 10): TCP position (3) + rot6d (6) in the episode's FIRST TCP frame, then closure.

    Start-relative, so the headset's world frame never enters these columns, and the pose of step t relative to
    step s is exact: inv(T0^-1 T_s) (T0^-1 T_t) = T_s^-1 T_t, UMI's relative representation.
    action[t] = state[t + 1] (the pose the hand went to next); the last step repeats its own state.
    """
    if len(episode) == 0:
        return np.zeros((0, 10), np.float32), np.zeros((0, 10), np.float32)
    origin = geometry.invert(episode.tcp[0])
    state = np.array([np.concatenate([geometry.pose_to_vector(origin @ pose), [c]])
                      for pose, c in zip(episode.tcp, episode.closure)], dtype=np.float32)
    action = np.concatenate([state[1:], state[-1:]], axis=0)
    return state, action


def episode_extras(episode: Episode, cfg: dict) -> dict[str, np.ndarray]:
    """The absolute columns UMI's replay buffer stores, so any pose representation can be rebuilt at load time.

    eef_pose_world: TCP pose (position + rot6d) in the headset's floor frame, consistent within one episode only.
    demo_start_pose_world: the episode's first TCP pose, repeated on every step (UMI's robot0_demo_start_pose).
    gripper_width_m: the jaw gap in metres.
    """
    n = len(episode)
    if n == 0:
        return {key: np.zeros((0, dim), np.float32) for key, dim in
                (("observation.eef_pose_world", 9), ("observation.demo_start_pose_world", 9),
                 ("observation.gripper_width_m", 1))}
    world = np.array([geometry.pose_to_vector(pose) for pose in episode.tcp], dtype=np.float32)
    return {"observation.eef_pose_world": world,
            "observation.demo_start_pose_world": np.repeat(world[:1], n, axis=0),
            "observation.gripper_width_m": gripper_width_m(episode.closure, cfg).astype(np.float32).reshape(n, 1)}
