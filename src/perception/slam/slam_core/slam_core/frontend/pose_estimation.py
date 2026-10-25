"""PnP RANSAC and refinement."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
# instead of returning 5 loose values, we return one object with named feilds
class PoseResult:
    T: np.ndarray        # 4x4, maps reference-camera points into the current camera
    inliers: np.ndarray  # indices into the input points
    ok: bool             # passed every sanity check
    reason: str = ""     # why it was rejected, if it was
    ratio: float = 0.0   # inliers / points

# 3D points, their matched pixels, intrinsics, and the image size. Everything after * is tuning


def estimate_pose(X, pts, K, image_size, *, reproj_px=2.0, iters=200,
                  min_inliers=30, min_ratio=0.30, min_cells=6,
                  max_trans_m=0.3, max_rot_deg=30.0):

    # if there are not enough points for PnP + RANSAC to use to run then we bail
    if len(X) < 6:
        return PoseResult(np.eye(4), np.empty(0, int), False, "too few points")

    # Solving PnP + RANSAC
    success, rvec, tvec, inl = cv2.solvePnPRansac(
        X.astype(np.float64), pts.astype(np.float64), K, None,
        iterationsCount=iters, reprojectionError=reproj_px, confidence=0.999)

    # if the RANSAC can't find any pose then it bails
    if not success or inl is None:
        return PoseResult(np.eye(4), np.empty(0, int), False, "ransaac failed")

    # returns the inlier indices in an awkward (N, 1) shape, and ravel() flattens them to a plain list.
    # Then rvec and tvec get turned into one 4x4 matrix T
    inl = inl.ravel()
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    ratio = len(inl) / len(X)

    # spread: how many cells of a 4x4 grid over the image contain an inlier
    w, h = image_size
    cells = {(int(4 * x / w), int(4 * y / h)) for x, y in pts[inl]}
    trans = np.linalg.norm(tvec)
    angle = np.degrees(np.linalg.norm(rvec))

    # sanity checks
    reason = ""
    if len(inl) < min_inliers:
        reason = f"{len(inl)} inliers < {min_inliers}"
    elif ratio < min_ratio:
        reason = f"inlier ratio {ratio:.0%} < {min_ratio:.0%}"
    elif len(cells) < min_cells:
        reason = f"inliers in only {len(cells)}/16 cells"
    elif trans > max_trans_m:
        reason = f"jump of {trans:.2f} m"
    elif angle > max_rot_deg:
        reason = f"turn of {angle:.0f} deg"
    return PoseResult(T, inl, reason == "", reason, ratio)
