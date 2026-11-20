"""Camera mount (tcp_to_camera) and camera-vs-pose latency, from one take of the gripper waved at an ArUco marker.

1. Print an ArUco marker (default DICT_4X4_50 id 0, --marker-size its black square in metres), fix it flat.
2. Record a take with collect.py: 20-30 s of tilting, twisting and moving the gripper while the camera sees the
   marker, some motion brisk (latency needs sharp starts and stops), some slow.
3. python -m umi_gripper.calibration.calibrate_camera datasets/umi_raw/<session>/take_0000

Latency: angular speed doesn't depend on which rigidly attached frame measures it, so the camera's (from the marker)
and the controller's can be cross-correlated before the mount is known. The lag goes into latency.camera_s.
Mount: with the lag removed, hand-eye calibration (OpenCV, Park) solves the camera in the grip frame from the grip
poses and the marker poses in the camera; tcp_to_camera = inv(grip_to_tcp) @ that.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from umi_gripper import config as umi_config
from umi_gripper import geometry
from umi_gripper._paths import PIONEER_HUMANOID, add
from umi_gripper.postprocess.sync import interpolate_poses, pose_track
from umi_gripper.take import load_take

WRIST_PINHOLE_F = 723.0   # perception/wrist_camera README: the rectified 640x480 pinhole, centred, no distortion


def angular_speed(times: np.ndarray, rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(midpoint times, |omega| rad/s) between consecutive rotations."""
    angles = np.array([np.linalg.norm(geometry.rotation_vector(a.T @ b)) for a, b in zip(rotations[:-1], rotations[1:])])
    dt = np.diff(times)
    keep = dt > 1e-6
    return ((times[:-1] + times[1:]) / 2)[keep], angles[keep] / dt[keep]


def estimate_lag(t_ref, s_ref, t_late, s_late, max_lag: float = 0.3, step: float = 0.002) -> tuple[float, float]:
    """Lag L (s) maximising corr(s_late(t), s_ref(t - L)): how much later the second signal's stamps run.

    Returns (L, peak normalised correlation). Both signals are resampled on a common uniform grid.
    """
    t_ref, s_ref, t_late, s_late = map(np.asarray, (t_ref, s_ref, t_late, s_late))
    start, stop = max(t_ref[0], t_late[0]) + max_lag, min(t_ref[-1], t_late[-1]) - max_lag
    if stop - start < 2.0:
        raise ValueError("signals overlap for under 2 s; record a longer take")
    grid = np.arange(start, stop, step)
    late = np.interp(grid, t_late, s_late)
    late = (late - late.mean()) / (late.std() + 1e-12)
    best = (0.0, -np.inf)
    for lag in np.arange(-max_lag, max_lag + step / 2, step):
        ref = np.interp(grid - lag, t_ref, s_ref)
        ref = (ref - ref.mean()) / (ref.std() + 1e-12)
        score = float(np.mean(late * ref))
        if score > best[1]:
            best = (float(lag), score)
    return best


def solve_hand_eye(world_grip: np.ndarray, cam_marker: np.ndarray) -> tuple[np.ndarray, float]:
    """T_grip_cam from grip poses in the world and the (fixed) marker's poses in the camera.

    Returns (T_grip_cam, spread m): how far the implied world marker positions scatter (a consistency check).
    """
    import cv2
    if len(world_grip) < 10:
        raise ValueError(f"only {len(world_grip)} usable views of the marker; need >= 10")
    rotation, translation = cv2.calibrateHandEye(
        [p[:3, :3] for p in world_grip], [p[:3, 3] for p in world_grip],
        [p[:3, :3] for p in cam_marker], [p[:3, 3] for p in cam_marker], method=cv2.CALIB_HAND_EYE_PARK)
    grip_cam = geometry.make_pose(rotation, translation.reshape(3))
    markers = np.array([(g @ grip_cam @ m)[:3, 3] for g, m in zip(world_grip, cam_marker)])
    spread = float(np.sqrt(np.mean(np.sum((markers - markers.mean(axis=0)) ** 2, axis=1))))
    return grip_cam, spread


def camera_matrix(fx: float | None) -> np.ndarray:
    if fx is None:
        try:
            add(PIONEER_HUMANOID)
            from pioneer_humanoid.camera_info import CALIB_DIR, read_camera_info
            _, k, _ = read_camera_info(Path(CALIB_DIR) / "wrist_cam_pinhole_640x480.yaml")
            return np.array(k, dtype=float).reshape(3, 3)
        except (ImportError, OSError, ValueError):
            fx = WRIST_PINHOLE_F
    return np.array([[fx, 0, 319.5], [0, fx, 239.5], [0, 0, 1]])


def detect_marker_poses(take, k: np.ndarray, dictionary: str, marker_id: int, size: float):
    """[(frame index, T_cam_marker)] for frames where the marker is seen."""
    import cv2
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)),
                                       cv2.aruco.DetectorParameters())
    half = size / 2
    corners3d = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    out = []
    for index in range(len(take.camera_s)):
        gray = cv2.cvtColor(take.load_frame(index), cv2.COLOR_RGB2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or marker_id not in ids.reshape(-1):
            continue
        image_points = corners[list(ids.reshape(-1)).index(marker_id)].reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(corners3d, image_points, k, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            out.append((index, geometry.make_pose(cv2.Rodrigues(rvec)[0], tvec.reshape(3))))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("take", type=Path)
    parser.add_argument("--marker-size", type=float, required=True, help="marker side, metres")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--fx", type=float, default=None, help="focal length px (default: the wrist pinhole yaml)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = umi_config.load_config()
    take = load_take(args.take)
    views = detect_marker_poses(take, camera_matrix(args.fx), args.dictionary, args.marker_id, args.marker_size)
    print(f"marker seen in {len(views)}/{len(take.camera_s)} frames")
    if len(views) < 30:
        print("FAILED: too few marker views; keep the marker in the camera's view")
        return 1
    # Lag on the raw camera stamps (no latency applied) against the pose track with its pose latency.
    cfg_no_cam_latency = umi_config.deep_merge(cfg, {"latency": {"camera_s": 0.0}})
    pose_t, positions, quats = pose_track(take, cfg_no_cam_latency)
    pose_rot = np.array([geometry.quat_to_matrix(q) for q in quats])
    cam_t = take.camera_s[[i for i, _ in views]]
    cam_rot = np.array([np.linalg.inv(p[:3, :3]) for _, p in views])   # camera orientation w.r.t. the marker
    lag, score = estimate_lag(*angular_speed(pose_t, pose_rot), *angular_speed(cam_t, cam_rot))
    print(f"camera stamps run {lag * 1000:+.0f} ms relative to poses (correlation {score:.2f})")
    if score < 0.5:
        print("WARNING: weak correlation; wave the gripper more briskly and record again")
    grip, valid = interpolate_poses(pose_t, positions, quats, cam_t - lag, cfg["postprocess"]["max_tracking_gap_s"])
    world_grip = grip[valid]
    cam_marker = np.array([p for _, p in views])[valid]
    grip_cam, spread = solve_hand_eye(world_grip, cam_marker)
    tcp_cam = geometry.invert(umi_config.grip_to_tcp(cfg)) @ grip_cam
    print(f"camera in tcp frame: xyz {np.round(tcp_cam[:3, 3], 4)} m, quat_xyzw "
          f"{np.round(geometry.matrix_to_quat(tcp_cam[:3, :3]), 4)}; marker spread {spread * 1000:.1f} mm")
    default = umi_config.load_config(calibration=None)
    moved = np.linalg.norm(tcp_cam[:3, 3] - umi_config.tcp_to_camera(default)[:3, 3])
    turned = math.degrees(np.linalg.norm(geometry.rotation_vector(
        umi_config.tcp_to_camera(default)[:3, :3].T @ tcp_cam[:3, :3])))
    print(f"vs the robot's wrist camera mount: {moved * 1000:.0f} mm, {turned:.1f} deg "
          "(large = the handheld camera is not where the robot's is)")
    if spread > 0.01:
        print("FAILED: spread > 10 mm; check --marker-size, the TCP calibration and that the marker didn't move")
        return 1
    if not args.dry_run:
        umi_config.update_calibration({
            "tcp_to_camera": {"xyz": tcp_cam[:3, 3].round(5).tolist(),
                              "quat_xyzw": geometry.matrix_to_quat(tcp_cam[:3, :3]).round(6).tolist()},
            # The pose track already had latency.pose_s removed, so the lag is the camera's own latency.
            "latency": {"camera_s": round(lag, 4)}})
        print(f"saved to {umi_config.DEFAULT_CALIBRATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
