import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from slam_core.frontend.features import grid_select
from slam_core.io.realsense import RealSense, RealSenseConfig
from slam_core.types import Keyframe
from slam_core.types.pose import covariance_from_sigmas, inv_T, make_T, rotation_angle, to_tum

# camera settings live in the shared config folder next to slam_core
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "d455.yaml")


# turns a 4x4 pose into the "tx ty tz qx qy qz qw" text used in every output file
def tum_str(T):
    return " ".join(f"{v:.6f}" for v in to_tum(T))


# loads XFeat the same way as x_featone.py / match_two.py, only moving it to the GPU if there is one
def load_xfeat(top_k=1000):
    xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k)
    return xfeat.cuda() if torch.cuda.is_available() else xfeat


# runs XFeat on one image and keeps only the best few keypoints in each grid cell
def detect(xfeat, gray, top_k=1000):
    # XFeat wants an HxWx3 image like cv2.imread gives, so copy the gray IR into 3 channels
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    d = xfeat.detectAndCompute(img, top_k=top_k)[0]
    h, w = gray.shape

    # spreads the keypoints across the image so they don't all pile up on one textured spot
    keep = grid_select(d["keypoints"].cpu().numpy(), d["scores"].cpu().numpy(), h, w)
    keep = torch.from_numpy(keep).to(d["keypoints"].device)
    d = {k: v[keep] for k, v in d.items()}

    # LighterGlue needs the image size to match, same as in match_two.py
    d["image_size"] = (w, h)
    return d


# matches two XFeat outputs with LighterGlue and returns the matched pixels in each image
def match(xfeat, d1, d2):
    res = xfeat.match_lighterglue(d1, d2)
    return np.asarray(res[0], dtype=np.float64), np.asarray(res[1], dtype=np.float64)


# turns pixels into 3D points in the camera frame using the depth image
# returns the points plus a mask of which pixels had usable depth
def backproject(px, depth, K, dist, zmin, zmax):
    # looks up the depth under each pixel, keeping only depths in the trusted range
    h, w = depth.shape
    u = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
    z = depth[v, u]
    valid = (z > zmin) & (z < zmax)
    if not valid.any():
        return np.zeros((0, 3)), valid

    # undistortPoints gives x=(u-cx)/fx, y=(v-cy)/fy, and multiplying by the depth gives the 3D point
    norm = cv2.undistortPoints(px[valid].reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    zv = z[valid].astype(np.float64)
    return np.column_stack([norm * zv[:, None], zv]), valid


# a keyframe plus everything needed to track new frames against it
@dataclass
class Ref:
    kf: Keyframe
    feats: dict                 # XFeat output for this keyframe's image
    depth: np.ndarray           # its depth image, to get 3D for matched pixels
    image: np.ndarray           # its IR image (map.py colors the point cloud with it)
    first_inliers: Optional[int] = None


# finds where the current camera is relative to keyframe ref
# returns (T_ref_cur, inlier pixels, None) on success or (None, None, reason) on failure
def estimate_pose(xfeat, ref, feats, K, dist, p, min_inliers):
    n_feat = len(feats["keypoints"])
    if n_feat < min_inliers:
        return None, None, f"few_features ({n_feat})"

    # matches the keyframe's features to this frame's features
    p_ref, p_cur = match(xfeat, ref.feats, feats)
    if len(p_ref) < min_inliers:
        return None, None, f"few_matches ({n_feat} feat, {len(p_ref)} matches)"

    # lifts the keyframe side of each match to 3D with the keyframe's depth (the current frame's depth isn't used)
    obj, valid = backproject(p_ref, ref.depth, K, dist, p.zmin, p.zmax)
    img = p_cur[valid]
    if len(obj) < min_inliers:
        return None, None, f"few_depth_matches ({len(p_ref)} matches, {len(obj)} with depth)"

    # PnP + RANSAC finds the camera pose that projects those 3D points onto the current pixels,
    # ignoring matches that land more than 3 px off (outliers)
    found, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K, dist, iterationsCount=200, reprojectionError=3.0, confidence=0.999)
    n_inl = 0 if inl is None else len(inl)
    if not found or n_inl < min_inliers:
        return None, None, f"pnp_failed ({len(obj)} matches, {n_inl} inliers)"

    # refines the pose using only the inliers
    idx = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx], K, dist, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)

    # PnP gives the keyframe in the current camera's frame, so flip it to get the camera in the keyframe's frame
    T_cur_ref = make_T(R, tvec.ravel())
    return inv_T(T_cur_ref), img[idx], None


# every tuning knob for VO; main() turns each one into a command line flag
@dataclass
class Params:
    top_k: int = 1000            # XFeat keypoints per image
    zmin: float = 0.3            # m, valid depth range
    zmax: float = 4.0
    min_inliers: int = 20        # below this, tracking is lost
    kf_trans: float = 0.15       # m, new keyframe after moving this far
    kf_rot: float = 12.0         # deg, or turning this much
    kf_inlier_ratio: float = 0.5  # or when inliers fall below this fraction of the keyframe's first tracking
    min_kf_points: int = 50      # a keyframe needs this many features with depth to be useful
    max_jump: float = 0.10       # m, max disagreement with the constant velocity prediction
    recover_kfs: int = 5         # when lost, also try this many recent keyframes
    max_lost: int = 10           # frames lost before re-initializing (leaves a gap)
    # odometry covariance: these sigmas at ref_inliers inliers, scaled by sqrt(ref_inliers / inliers)
    ref_inliers: float = 150
    odom_rot: float = 0.01       # rad
    odom_trans: float = 0.02     # m
    gap_rot: float = 1.0         # rad, across a tracking gap ("somewhere near the last pose")
    gap_trans: float = 1.0       # m
    # loop closure: brute force against old keyframes near the current VO position
    loop_min_time: float = 10.0  # s, only look for loops this long ago (a revisit, not "still here")
    loop_cooldown: int = 10      # keyframes to wait after a loop before looking again
    loop_radius: float = 1.0     # m, candidates within this distance (by VO estimate)
    loop_max_cands: int = 3
    loop_min_inliers: int = 40   # stricter than tracking: false loops are worse than missed ones


# what one process() call returns, which is everything the viewer (and later the ROS node) needs
@dataclass
class Result:
    state: str                   # init | ok | recovered | lost | reinit
    pose_w: np.ndarray
    n_features: int = 0
    n_inliers: int = 0
    reason: str = ""
    px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    inlier_px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    new_kf: Optional[Keyframe] = None   # keyframe created this frame
    loop_to: int = -1                   # id of an old keyframe a loop was found to, else -1


class VO:
    def __init__(self, xfeat, p=Params()):
        self.xfeat, self.p = xfeat, p
        self.refs = []               # every keyframe, index = keyframe id
        self.ref = None              # current reference keyframe
        self.T_ref_cur = np.eye(4)   # last good pose relative to self.ref
        self.vel = np.eye(4)         # last frame to frame motion, T_prev_cur
        self.pose_w = np.eye(4)      # last good pose in the odom frame
        self.lost = 0
        self.loops = []              # (new_id, old_id, stamp, T_old_new, inliers)
        self.last_loop_kf = -10**9
        self.frames = []             # (stamp, kf_id, T_kf_cur) for every tracked frame
        self.K = self.dist = None

    # the Keyframe objects on their own, without the tracking data
    @property
    def keyframes(self):
        return [r.kf for r in self.refs]

    # how much the backend should trust an odometry edge: more inliers means smaller sigmas,
    # and a tracking gap gets huge sigmas since it's only a guess
    def _covariance(self, n_inliers, gap):
        p = self.p
        if gap:
            return covariance_from_sigmas(p.gap_rot, p.gap_trans)
        k = float(np.clip(np.sqrt(p.ref_inliers / max(n_inliers, 1)), 0.5, 10))
        return covariance_from_sigmas(p.odom_rot * k, p.odom_trans * k)

    # makes a new keyframe from frame f, or returns None if too few of its features have depth
    def _make_ref(self, f, feats, prev, T_prev_kf, n_inliers, gap):
        px = feats["keypoints"].cpu().numpy()
        _, valid = backproject(px, f.depth, self.K, self.dist, self.p.zmin, self.p.zmax)
        if valid.sum() < self.p.min_kf_points:
            return None
        kf = Keyframe(id=len(self.refs), prev_id=-1 if prev is None else prev.kf.id, stamp=f.stamp,
                      T_prev_kf=T_prev_kf, covariance=self._covariance(n_inliers, gap),
                      T_odom_kf=self.pose_w, tracking_gap=gap, n_inliers=n_inliers,
                      keypoints=px.astype(np.float32),
                      descriptors=feats["descriptors"].cpu().numpy())
        return Ref(kf, feats, f.depth, f.image)

    # stores a keyframe and makes it the one new frames track against
    def _add(self, r):
        self.refs.append(r)
        self.ref, self.T_ref_cur = r, np.eye(4)

    # runs VO on one frame and returns a Result
    def process(self, f):
        p = self.p
        if self.K is None:
            self.K, self.dist = f.K, f.dist
        feats = detect(self.xfeat, f.image, p.top_k)
        px = feats["keypoints"].cpu().numpy()
        n_feat = len(px)

        # the first frame becomes keyframe 0 at the origin
        if self.ref is None:
            r = self._make_ref(f, feats, None, np.eye(4), 0, False)
            if r is None:
                return Result("init", self.pose_w, n_feat, reason="few features with depth", px=px)
            self._add(r)
            self.frames.append((f.stamp, r.kf.id, np.eye(4)))
            return Result("init", self.pose_w, n_feat, px=px, new_kf=r.kf)

        # predicts where we should be if we keep moving like last frame (constant velocity)
        pred_w = self.ref.kf.T_odom_kf @ self.T_ref_cur @ self.vel
        # after a loss the prediction is stale, so allow a bigger disagreement
        max_jump = p.max_jump * (1 + self.lost)

        # tracks against the current keyframe first, then the most recent ones if that fails
        cands = [self.ref] + [r for r in reversed(self.refs[-(p.recover_kfs + 1):]) if r is not self.ref]
        ref, T, inl_px, reason = None, None, None, ""
        for r in cands:
            T, inl_px, why = estimate_pose(self.xfeat, r, feats, self.K, self.dist, p, p.min_inliers)
            # a pose that jumps too far from the prediction is confident but wrong, so skip it
            if T is not None:
                jump = np.linalg.norm((r.kf.T_odom_kf @ T)[:3, 3] - pred_w[:3, 3])
                if jump < max_jump:
                    ref = r
                    break
                why = f"pose_jump ({jump:.2f} m)"
            # reports why the current keyframe failed, not the later fallbacks
            reason = reason or why
            T = None

        # nothing tracked: hold the last good pose and count the frame as lost
        if ref is None:
            self.lost += 1
            if self.lost <= p.max_lost:
                return Result("lost", self.pose_w, n_feat, reason=reason, px=px)
            # lost for too long, so restart from this frame at the last good pose,
            # with a weak gap edge so the graph stays connected
            r = self._make_ref(f, feats, self.ref, np.eye(4), 0, True)
            if r is None:
                return Result("lost", self.pose_w, n_feat, reason=reason, px=px)
            self._add(r)
            self.lost, self.vel = 0, np.eye(4)
            self.frames.append((f.stamp, r.kf.id, np.eye(4)))
            return Result("reinit", self.pose_w, n_feat, reason=reason, px=px, new_kf=r.kf)

        # tracked: tracking against an older keyframe or coming back from lost counts as recovered,
        # and the velocity is reset since the last motion isn't meaningful anymore
        state = "ok"
        if ref is not self.ref or self.lost:
            state = "recovered"
            self.vel = np.eye(4)
        else:
            self.vel = inv_T(self.T_ref_cur) @ T

        # updates the pose and logs the frame relative to its keyframe (pose_graph.py re-anchors these later)
        self.ref, self.T_ref_cur, self.lost = ref, T, 0
        self.pose_w = ref.kf.T_odom_kf @ T
        self.frames.append((f.stamp, ref.kf.id, T))
        n_inl = len(inl_px)
        res = Result(state, self.pose_w, n_feat, n_inl, px=px, inlier_px=inl_px)

        # makes a new keyframe after moving or turning enough, or when overlap with the keyframe gets low.
        # the inlier rule is relative to the keyframe's first tracking, so a low texture scene
        # doesn't make a keyframe every frame
        if ref.first_inliers is None:
            ref.first_inliers = n_inl
        moved = np.linalg.norm(T[:3, 3]) > p.kf_trans
        turned = np.degrees(rotation_angle(T[:3, :3])) > p.kf_rot
        overlap_low = n_inl < p.kf_inlier_ratio * ref.first_inliers
        if moved or turned or overlap_low:
            r = self._make_ref(f, feats, ref, T, n_inl, False)
            if r is not None:
                self._add(r)
                res.new_kf = r.kf
                res.loop_to = self._detect_loop(r)
        return res

    # brute force loop check: matches the new keyframe against the few old keyframes closest to it
    # by VO position. placeholder for real place recognition, since drift can put the true match
    # outside the radius
    def _detect_loop(self, new):
        p = self.p
        # waits a few keyframes after a loop so the same revisit isn't added again and again
        if new.kf.id - self.last_loop_kf < p.loop_cooldown:
            return -1

        # candidates are keyframes from long enough ago that VO places nearby, closest first
        t = new.kf.T_odom_kf[:3, 3]
        dist = lambda r: np.linalg.norm(r.kf.T_odom_kf[:3, 3] - t)   # noqa: E731
        old = [r for r in self.refs if r.kf.stamp < new.kf.stamp - p.loop_min_time and dist(r) < p.loop_radius]
        old.sort(key=dist)

        # the first candidate that passes PnP with the stricter inlier count becomes a loop edge (T_old_new)
        for r in old[:p.loop_max_cands]:
            T, inl_px, _ = estimate_pose(self.xfeat, r, new.feats, self.K, self.dist, p, p.loop_min_inliers)
            if T is not None:
                self.loops.append((new.kf.id, r.kf.id, new.kf.stamp, T, len(inl_px)))
                self.last_loop_kf = new.kf.id
                return r.kf.id
        return -1

    # writes everything pose_graph.py and map.py need into the out directory
    def write(self, out):
        os.makedirs(out, exist_ok=True)
        kfs = self.keyframes

        # the raw VO trajectory, one pose per tracked frame
        with open(os.path.join(out, "trajectory_vo.txt"), "w") as fh:
            for stamp, kf_id, T in self.frames:
                fh.write(f"{stamp:.6f} {tum_str(kfs[kf_id].T_odom_kf @ T)}\n")

        # the raw VO keyframe poses (T_odom_kf) in id order
        with open(os.path.join(out, "kf_vo.txt"), "w") as fh:
            for kf in kfs:
                fh.write(f"{kf.stamp:.6f} {tum_str(kf.T_odom_kf)}\n")

        # the odometry edges for the pose graph
        with open(os.path.join(out, "keyframes.txt"), "w") as fh:
            fh.write("# kf_id prev_id stamp tx ty tz qx qy qz qw inliers gap sigma_rot sigma_trans"
                     "  (pose = T_prev_kf; prev -1 = first keyframe; sigmas of the diagonal covariance)\n")
            for kf in kfs:
                c = np.sqrt(np.diag(kf.covariance))
                fh.write(f"{kf.id} {kf.prev_id} {kf.stamp:.6f} {tum_str(kf.T_prev_kf)} "
                         f"{kf.n_inliers} {int(kf.tracking_gap)} {c[0]:.6f} {c[3]:.6f}\n")

        # the loop edges for the pose graph
        with open(os.path.join(out, "loops.txt"), "w") as fh:
            fh.write("# new_kf_id old_kf_id stamp tx ty tz qx qy qz qw inliers  (pose = T_old_new)\n")
            for i, j, stamp, T, n in self.loops:
                fh.write(f"{i} {j} {stamp:.6f} {tum_str(T)} {n}\n")

        # each keyframe's depth (as uint16 mm, which compresses well) and IR image, for map.py
        arrays = {"K": self.K}
        for r in self.refs:
            arrays[f"depth_{r.kf.id}"] = np.round(np.clip(r.depth, 0, 65.535) * 1000).astype(np.uint16)
            arrays[f"image_{r.kf.id}"] = r.image
        np.savez_compressed(os.path.join(out, "keyframe_images.npz"), **arrays)

        # every frame relative to its keyframe, so pose_graph.py can move frames along with corrected keyframes
        with open(os.path.join(out, "frames.txt"), "w") as fh:
            fh.write("# stamp kf_id tx ty tz qx qy qz qw  (pose = T_kf_cur, the frame in its keyframe)\n")
            for stamp, kf_id, T in self.frames:
                fh.write(f"{stamp:.6f} {kf_id} {tum_str(T)}\n")


# draws the debug image: red dots for every feature, green dots for PnP inliers, and a status line on top
def draw(f, res, n_kf, n_loops):
    vis = cv2.cvtColor(f.image, cv2.COLOR_GRAY2BGR)
    for u, v in res.px:
        cv2.circle(vis, (int(u), int(v)), 2, (0, 0, 255), -1)
    for u, v in res.inlier_px:
        cv2.circle(vis, (int(u), int(v)), 3, (0, 255, 0), -1)

    # status text is green while tracking and red when lost, with the failure reason underneath
    good = res.state in ("ok", "recovered", "init")
    x, y, z = res.pose_w[:3, 3]
    cv2.putText(vis, f"{res.state.upper()} inl {res.n_inliers}/{res.n_features}  KF {n_kf}  "
                f"loops {n_loops}  x={x:+.2f} y={y:+.2f} z={z:+.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if good else (0, 0, 255), 2)
    if res.reason:
        cv2.putText(vis, res.reason, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG, help="D455 settings yaml")
    ap.add_argument("--play", help="run on a recording (.db3) instead of live")
    ap.add_argument("--record", help="also record the raw streams to this .db3")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--no-viz", action="store_true", help="no OpenCV window")
    ap.add_argument("--foxglove", action="store_true", help="live 3D view in Foxglove (ws://localhost:8765)")
    ap.add_argument("--mcap", help="record the Foxglove view to this .mcap file (open it in Foxglove)")
    # every Params field becomes a flag, e.g. kf_trans -> --kf-trans
    for name, val in vars(Params()).items():
        ap.add_argument("--" + name.replace("_", "-"), type=type(val), default=val)
    args = ap.parse_args()
    p = Params(**{k: getattr(args, k) for k in vars(Params())})

    # opens the camera (or recording) and loads XFeat
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
    cam = RealSense(RealSenseConfig.from_yaml(args.config), play=args.play, record=args.record)
    vo = VO(load_xfeat(p.top_k), p)

    # sets up the Foxglove view only when asked, since it needs the foxglove package
    fox = None
    if args.foxglove or args.mcap:
        from viz import FoxgloveViz
        if args.mcap:
            os.makedirs(os.path.dirname(os.path.abspath(args.mcap)), exist_ok=True)
        fox = FoxgloveViz(server=args.foxglove, mcap=args.mcap)

    n, n_lost, ms = 0, 0, []
    try:
        for f in cam.frames():
            # runs VO on the frame and times it
            t0 = time.perf_counter()
            res = vo.process(f)
            ms.append((time.perf_counter() - t0) * 1000)
            n += 1

            # prints tracking events, only the first frame of each loss so it doesn't spam
            if res.state == "lost":
                n_lost += 1
                if vo.lost == 1:
                    print(f"[{f.stamp:.2f}] lost: {res.reason}")
            elif res.state in ("recovered", "reinit"):
                print(f"[{f.stamp:.2f}] {res.state}")
            if res.loop_to >= 0:
                print(f"[{f.stamp:.2f}] loop: KF {res.new_kf.id} -> KF {res.loop_to}")

            # shows the debug image in Foxglove and/or an OpenCV window, q quits
            if fox or not args.no_viz:
                vis = draw(f, res, len(vo.refs), len(vo.loops))
                if fox:
                    fox.update(f, res, vo, vis)
                if not args.no_viz:
                    cv2.imshow("vo", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        # always writes the outputs and a summary, however the run ended
        cv2.destroyAllWindows()
        if fox:
            fox.close()
        if vo.refs:
            vo.write(args.out)
        gaps = sum(kf.tracking_gap for kf in vo.keyframes)
        end = np.linalg.norm(vo.pose_w[:3, 3])
        print(f"\n{n} frames, {n_lost} lost, {len(vo.refs)} keyframes, {gaps} gaps, "
              f"{len(vo.loops)} loops, {np.mean(ms) if ms else 0:.1f} ms/frame")
        print(f"end position is {end:.3f} m from the start (= loop error if you ended on the start mark)")
        print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
