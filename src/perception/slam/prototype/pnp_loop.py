import cv2
import torch
import numpy as np
from slam_core.frontend.pose_estimation import estimate_pose

# this is gonna be using the TUM dataset, D is the data and K is the camera intrisics matrix
D = "data/rgbd_dataset_freiburg1_desk/"
K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]])
# TUM fr1 RGB: k1, k2, p1, p2, k3 (known)
DIST = np.array([0.2624, -0.9531, -0.0054,  0.0026, 1.1633])
DEPTH_SCALE = 5000.0
# Load XFeat once
device = "cuda" if torch.cuda.is_available() else "cpu"
xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=1000).to(device)

# TUM ships text files in which every image is listed as timestamp filename
# reads the line and skips commens returning ann araray of timestamps and a list of filenames


def load_list(name):
    rows = [l.split() for l in open(D + name) if not l.startswith("#")]
    return np.array([float(r[0]) for r in rows]), [r[1] for r in rows]


# loads timestame and then the file
rgb_t, rgb_f = load_list("rgb.txt")
dep_t, dep_f = load_list("depth.txt")

gaps = np.array([np.min(np.abs(dep_t - t)) for t in rgb_t])
k0 = int(np.argmax(gaps < 0.015))  # first RGB frame with depth within 15 ms
print(f"first good frame {k0}, gap {gaps[k0]*1000:.1f} ms")

# loads RGB frame i and its match depth imageIt finds the closest depth timeframe and coverts it to meters.
# also returns is the timegap between the two so we can see how well they line up


def load(i):
    img = cv2.imread(D + rgb_f[i])
    j = np.argmin(np.abs(dep_t - rgb_t[i]))
    depth = cv2.imread(D + dep_f[j], cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
    return img, depth, abs(dep_t[j] - rgb_t[i])


def undistort(p):
    # undistorted pixel coords where the ideal pinhole would be
    return cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2)


gt = np.loadtxt(D + "groundtruth.txt")


def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ])


def gt_pose(t):
    # nearest groundtruth sample
    row = gt[np.argmin(np.abs(gt[:, 0] - t))]
    # normalize the quanterion
    q = row[4:8] / np.linalg.norm(row[4:8])
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*row[4:8])
    T[:3, 3] = row[1:4]
    return T  # where the camera pose is in the world


def angle_of(Rm):
    return np.degrees(np.linalg.norm(cv2.Rodrigues(Rm)[0]))

# runs XFeat on one image and adds the image size LighterGlue needs


def feats(img):
    d = xfeat.detectAndCompute(img, top_k=1000)[0]
    d["image_size"] = (img.shape[1], img.shape[0])
    return d

# pnp_two as a helper function
# loads both frames, matches features -> undistorts matched points, looks up depth
# bails out early if fewer than 6 points don't have depth since PnP neeeds those.
# Back projects to 3d and runs PnP + RANSAC and compare to ground truth


def run_pair(i, j):
    img0, depth0, gap = load(i)
    img1, _, _, = load(j)
    p0, p1 = xfeat.match_lighterglue(feats(img0), feats(img1))[:2]
    p0u, p1u = undistort(p0), undistort(p1)

    u = np.clip(p0[:, 0].round().astype(int), 0, depth0.shape[1] - 1)
    v = np.clip(p0[:, 1].round().astype(int), 0, depth0.shape[0] - 1)
    z = depth0[v, u]
    ok = z > 0
    if ok.sum() < 6:
        return None

    X = np.stack([(p0u[ok, 0] - K[0, 2]) * z[ok] / K[0, 0],
                  (p0u[ok, 1] - K[1, 2]) * z[ok] / K[1, 1],
                  z[ok]], axis=1)

    res = estimate_pose(X, p1u[ok], K, (img0.shape[1], img0.shape[0]))
    if len(res.inliers) == 0:
        return None  # RANSAC found nothing at all

    R, t = res.T[:3, :3], res.T[:3, 3]
    T_rel = np.linalg.inv(gt_pose(rgb_t[j])) @ gt_pose(rgb_t[i])
    return dict(gap=gap * 1000, ratio=res.ratio, ok=res.ok, reason=res.reason,
                rot_err=angle_of(T_rel[:3, :3].T @ R),
                trans_err=np.linalg.norm(T_rel[:3, 3] - t) * 100,
                motion=np.linalg.norm(T_rel[:3, 3]) * 100)


# The loop
STEP = 5
results = []
for i in range(k0, len(rgb_t) - STEP, STEP):
    r = run_pair(i, i + STEP)
    if r is None:
        print(f"{i}: FAILED")
        continue
    results.append(r)
    flag = "" if r["ok"] else f"  REJECT: {r['reason']}"
    print(f"{i}: gap {r['gap']:4.1f} ms  inliers {r['ratio']:.0%}  "
          f"err {r['rot_err']:.2f} deg {r['trans_err']:.1f} cm  (moved {r['motion']:.1f} cm){flag}")

# Summary
gap_ms = np.array([r["gap"] for r in results])
terr = np.array([r["trans_err"] for r in results])
rerr = np.array([r["rot_err"] for r in results])
ratio = np.array([r["ratio"] for r in results])
motion = np.array([r["motion"] for r in results])
print(f"\n{len(results)} pairs")
print(f"translation error: median {np.median(terr):.1f} cm, mean {terr.mean():.1f} cm")
print(f"rotation error: median {np.median(rerr):.2f} deg, mean {rerr.mean():.2f} deg")
print(f"inlier ratio: median {np.median(ratio):.0%}")
print(f"corr(gap, translation error) = {np.corrcoef(gap_ms, terr)[0, 1]:.2f}")
print(f"corr(gap, inlier ratio) = {np.corrcoef(gap_ms, ratio)[0, 1]:.2f}")
print(f"corr(gap x motion, translation error) = {np.corrcoef(gap_ms * motion, terr)[0, 1]:.2f}")
print(f"corr(inlier ratio, translation error) = {np.corrcoef(ratio, terr)[0, 1]:.2f}")
for th in (0.30, 0.35, 0.40):
    bad = ratio < th
    print(f"ratio < {th:.0%}: rejects {bad.sum()} pairs, their median err {np.median(terr[bad]):.1f} cm, "
          f"kept median err {np.median(terr[~bad]):.1f} cm")
okm = np.array([r["ok"] for r in results])
print(f"\naccepted {okm.sum()}/{len(results)}")
print(f"accepted median err {np.median(terr[okm]):.1f} cm, "
      f"rejected median err {np.median(terr[~okm]):.1f} cm")
