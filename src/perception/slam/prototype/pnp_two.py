import cv2, torch, numpy as np

# this is gonna be using the TUM dataset, D is the data and K is the camera intrisics matrix
D = "data/rgbd_dataset_freiburg1_desk/"
K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]])
# TUM fr1 RGB: k1, k2, p1, p2, k3 (known)
DIST = np.array([0.2624, -0.9531, -0.0054,  0.0026, 1.1633])
DEPTH_SCALE = 5000.0

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

# keyframes only need the depth , frame 5 places the current fram,e which is the pose we want
# if the depth is 0 we render that kinda useless
img0, depth0, gap = load(k0) # keyframe
img1, _, _ = load(k0 + 5) # current frame
print(f"rgb-depth gap {gap*1000:.1f} ms, depth {depth0[depth0 > 0].min():.2f}-{depth0.max():.2f} m")

# same as match_two where we match the features
xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=1000).cuda()
d0 = xfeat.detectAndCompute(img0, top_k = 1000)[0]
d1 = xfeat.detectAndCompute(img1, top_k = 1000)[0]
d0["image_size"] = d1["image_size"] = (img0.shape[1], img0.shape[0])
res = xfeat.match_lighterglue(d0, d1)
p0, p1 = res[0], res[1]
p0u, p1u = undistort(p0), undistort(p1)

# looks at the depth undereach keyframe point
u = np.clip(p0[:, 0].round().astype(int), 0, depth0.shape[1] - 1)
v = np.clip(p0[:, 1].round().astype(int), 0, depth0.shape[0] - 1)
z = depth0[v, u]
ok = z > 0

# back-projection the pinhole model runs backwards. This will be rewritten later
X = np.stack([
    (p0u[ok, 0] - K[0, 2]) * z[ok] / K[0,0],
    (p0u[ok, 1] - K[1, 2]) * z[ok] / K[1,1],
    z[ok]], axis = 1)

pts1 = p1u[ok]
print(len(p0), "matches,", ok.sum(), "with valid depth")

# PnP and RANSAC (yay) -> this is the pose that maps those keyframe 3D points onto their current-frame pixels
# rvec is the rotation and tvec is how it moved
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    X.astype(np.float64), pts1.astype(np.float64), K, None,
    iterationsCount = 200, reprojectionError = 2.0, confidence = 0.999
)

# axis-angle (rotation matrix)
R, _ = cv2.Rodrigues(rvec)
n_in = 0 if inliers is None else len(inliers)
angle = np.degrees(np.linalg.norm(rvec))
print(f"success {success}, inliers {n_in}/{len(X)} ({n_in / len(X):.0%})")
print(f"rotation {angle:.2f} deg, translation {np.linalg.norm(tvec) * 100:.1f} cm")

# ground truth: TUM gives the camera pose in the world per timestamp
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
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*row[4:8])
    T[:3, 3] = row[1:4]
    return T # where the camera pose is in the world

def angle_of(Rm):
    return np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1 , 1)))

# true motion in the same convention PnP returns: keyframes -> current camera
T_rel = np.linalg.inv(gt_pose(rgb_t[k0 + 5])) @ gt_pose(rgb_t[k0])

print(f"ground truth: rotation {angle_of(T_rel[:3, :3]):.2f} deg, "
      f"translation {np.linalg.norm(T_rel[:3, 3]) * 100:.1f} cm")
print(f"error: rotation {angle_of(T_rel[:3, :3].T @ R):.2f} deg, "
      f"translation {np.linalg.norm(T_rel[:3, 3] - tvec.ravel()) * 100:.1f} cm")