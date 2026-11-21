"""Geometric verification: turn a candidate (old, new) keyframe pair into a relative pose, or reject it.

A verifier is any callable verify(old_id, new_id) -> Verification | None. The real one (`PnPVerifier`) matches
features between the two keyframes, lifts the old keyframe's matches to 3D with its depth, and runs PnP + RANSAC,
the same recipe as tracking but with loop-sized motion limits and a stricter inlier count. The simulator supplies
its own verifier, so the detector can be tested without images.
"""
import numpy as np

from slam_core.types.pose import inv_T

from .types import Verification


# Loop edge covariance that shrinks with the inlier count, like the odometry edges do:
# sigma = sigma_ref * sqrt(ref_inliers / inliers), clipped so one lucky frame can't get tiny sigmas
def loop_covariance(n_inliers, rot=0.02, trans=0.03, ref_inliers=100):
    k = float(np.clip(np.sqrt(ref_inliers / max(n_inliers, 1)), 0.5, 5.0))
    return np.diag(np.array([rot * k] * 3 + [trans * k] * 3) ** 2)


class PnPVerifier:
    # match_fn(old_id, new_id) -> (X_old (N, 3) points in the old keyframe's camera frame, px_new (N, 2) pixels
    # in the new keyframe that match them). Feature matching and depth lookup live there, so this class has no
    # dependency on XFeat
    def __init__(self, match_fn, K, image_size, min_inliers=30, min_ratio=0.2, min_cells=6):
        self.match_fn, self.K, self.image_size = match_fn, np.asarray(K, dtype=np.float64), image_size
        self.min_inliers, self.min_ratio, self.min_cells = min_inliers, min_ratio, min_cells

    def __call__(self, old_id, new_id):
        from slam_core.frontend.pose_estimation import estimate_pose  # imported here so OpenCV stays optional

        X, px = self.match_fn(old_id, new_id)
        if len(X) == 0:
            return None
        res = estimate_pose(X, px, self.K, self.image_size, min_inliers=self.min_inliers,
                            min_ratio=self.min_ratio, min_cells=self.min_cells,
                            max_trans_m=np.inf, max_rot_deg=180.0)
        if not res.ok:
            return None
        # estimate_pose maps old-camera points into the new camera, so its inverse is the new camera in the old frame
        return Verification(inv_T(res.T), loop_covariance(len(res.inliers)), len(res.inliers))
