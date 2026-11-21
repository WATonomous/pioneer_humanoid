"""VOCore: Frame in, pose + stats + optional keyframe out. Tracking-loss detection and recovery."""
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from slam_core.types.keyframe import Keyframe
from .features import FeatureExtractor
from .matching import Matcher
from .pose_estimation import estimate_pose


def undistort(pts, K, dist):
    # Raw pixels to pizels of an ideal pinhole camera with same K
    if len(pts) == 0 or not np.any(dist):
        return pts.astype(np.float64)
    return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K).reshape(-1, 2)


def depth_at(depth, pts):
    # Depth at each raw pixel
    h, w = depth.shape
    u = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return depth[v, u]


def back_project(pts, z, K):
    # Undistorted pizels and depth to points on a camera frame
    x = (pts[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (pts[:, 1] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


@dataclass
class TrackResult:
    ok: bool                             # pose is trustworthy (passed the checks)
    T_odom_cur: np.ndarray               # 4x4; the last good pose if not ok
    reason: str = ""                     # why tracking failed, if it did
    n_matches: int = 0
    n_inliers: int = 0
    keyframe: Optional[Keyframe] = None  # set when this frame became a new keyframe


# Placeholder VO uncertainty: (0.01 rad)^2 rotation, (1 cm)^2 translation, GTSAM order
# (rx ry rz tx ty tz). TODO FE-17: replace with the covariance from the final Hessian.
COV_PLACEHOLDER = np.diag([1e-4] * 3 + [1e-4] * 3)


class Tracker:
    # Visual odometry: track every frame against the current keyframe.
    def __init__(self, extractor=None, *, kf_min_inliers=100, kf_max_trans=0.15,
                 kf_max_rot_deg=15.0, max_lost=5, pose_kwargs=None):
        self.extractor = extractor or FeatureExtractor()
        self.matcher = Matcher(self.extractor)
        self.kf_min_inliers = kf_min_inliers  # fewer inliers than this -> new keyframe
        self.kf_max_trans = kf_max_trans      # m moved from the keyframe -> new keyframe
        self.kf_max_rot_deg = kf_max_rot_deg  # deg turned from the keyframe -> new keyframe
        self.max_lost = max_lost              # failed frames in a row before restarting
        self.pose_kwargs = pose_kwargs or {}  # passed to estimate_pose (thresholds)
        self.kf = None          # current keyframe: frame, features, 3D points, valid mask
        self.kf_id = -1
        self.T_odom_kf = np.eye(4)
        self.T_odom_cur = np.eye(4)
        self.n_lost = 0
        self.last = None  # (frame, features, T_odom, n_inliers) of the last tracked frame
        self.vel = np.eye(4)
        loose_min_ratio = 0.15
        self.loose_min_ratio = loose_min_ratio

    def _set_keyframe(self, frame, feats, T_odom_frame, T_prev_kf, n_inliers, gap=False):
        # Make `frame` the new tracking reference and return its Keyframe message
        z = depth_at(frame.depth, feats.keypoints)
        valid = z > 0
        points = back_project(undistort(feats.keypoints, frame.K, frame.dist), z, frame.K)

        prev_id = self.kf_id
        self.kf_id += 1
        self.kf = dict(frame=frame, feats=feats, points=points, valid=valid)
        self.T_odom_kf = T_odom_frame

        cov = COV_PLACEHOLDER * (1e4 if gap else 1.0)  # a tracking gap is just a guess
        return Keyframe(
            id=self.kf_id, prev_id=prev_id, stamp=frame.stamp,
            T_prev_kf=T_prev_kf, covariance=cov, T_odom_kf=T_odom_frame,
            tracking_gap=gap, n_inliers=n_inliers,
            keypoints=feats.keypoints[valid].astype(np.float32),
            descriptors=feats.descriptors[valid],
            points_3d=points[valid].astype(np.float32))

    def _track(self, feats, frame, size, min_ratio=None):
        # match against the current keyframe and estimate the pose
        m = self.matcher.match(self.kf["feats"], feats, size)
        keep = self.kf["valid"][m.idx_a]
        X = self.kf["points"][m.idx_a[keep]]
        pts = undistort(feats.keypoints[m.idx_b[keep]], frame.K, frame.dist)
        kwargs = dict(self.pose_kwargs)
        if min_ratio is not None:
            kwargs["min_ratio"] = min_ratio
        return m, estimate_pose(X, pts, frame.K, size, **kwargs)

    def process(self, frame):
        # Track one Frame. Returns a TrackResult (pose, stats, maybe a new Keyframe)
        if not frame.has_depth:
            raise ValueError("Tracker needs depth (RGB-D input)")
        feats = self.extractor.extract(frame.image)
        size = (frame.width, frame.height)

        # first frame: becomes keyframe 0 and defines the odom origin
        if self.kf is None:
            kf = self._set_keyframe(frame, feats, np.eye(4), np.eye(4), len(feats.keypoints))
            self.last = (frame, feats, np.eye(4), len(feats.keypoints))
            return TrackResult(True, self.T_odom_cur.copy(), keyframe=kf)

        m, res = self._track(feats, frame, size)

        # too far from the keyframe: promote the last good frame to keyframe and retry from it
        promoted = None
        if not res.ok and self.last[0] is not self.kf["frame"]:
            lf, lfeats, T_odom_last, n_last = self.last
            T_prev_kf = np.linalg.inv(self.T_odom_kf) @ T_odom_last
            promoted = self._set_keyframe(lf, lfeats, T_odom_last, T_prev_kf, n_last)
            m, res = self._track(feats, frame, size)

        # still failing: accept a rougher pose than coast blind
        if not res.ok:
            m, res = self._track(feats, frame, size, min_ratio=self.loose_min_ratio)

        # still failing: publish nothing; after max_lost in a row, restart from here
        if not res.ok:
            self.T_odom_cur = self.T_odom_cur @ self.vel  # keep moving like the last frame
            self.n_lost += 1
            kf = promoted
            if kf is None and self.n_lost >= self.max_lost:
                T_guess = np.linalg.inv(self.T_odom_kf) @ self.T_odom_cur  # last good pose, as a guess
                kf = self._set_keyframe(frame, feats, self.T_odom_cur.copy(), T_guess, 0, gap=True)
                self.last = (frame, feats, self.T_odom_cur.copy(), 0)
                self.n_lost = 0
            return TrackResult(False, self.T_odom_cur.copy(), res.reason, len(m), len(res.inliers), kf)

        # success: chain into odom. res.T is T_cur_kf, so the camera's pose in the keyframe is its inverse
        was_lost = self.n_lost > 0
        self.n_lost = 0
        T_kf_cur = np.linalg.inv(res.T)
        T_prev = self.T_odom_cur
        self.T_odom_cur = self.T_odom_kf @ T_kf_cur
        if not was_lost:  # only learn the velocity from two real poses in a row
            self.vel = np.linalg.inv(T_prev) @ self.T_odom_cur
        self.last = (frame, feats, self.T_odom_cur.copy(), len(res.inliers))

        # new keyframe if tracking is weakening or we've moved far (skip if we just promoted one)
        kf = promoted
        if kf is None:
            trans = np.linalg.norm(T_kf_cur[:3, 3])
            angle = np.degrees(np.linalg.norm(cv2.Rodrigues(T_kf_cur[:3, :3])[0]))
            if len(res.inliers) < self.kf_min_inliers or trans > self.kf_max_trans or angle > self.kf_max_rot_deg:
                kf = self._set_keyframe(frame, feats, self.T_odom_cur.copy(), T_kf_cur, len(res.inliers))
        return TrackResult(True, self.T_odom_cur.copy(), "", len(m), len(res.inliers), kf)
