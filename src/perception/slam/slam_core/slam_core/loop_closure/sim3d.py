"""The synthetic world of sim.py, but in full 3D (6 degrees of freedom).

The building has a ramp-like height profile, and the robot's camera pitches with the slope and sways (roll), so every
pose is a general SE(3) pose. Odometry drifts in all three rotation axes, which also makes it drift vertically.
Landmarks are 3D points at different heights. The two side hallways are identical copies (rotated 180 degrees about
the vertical axis and at a different height), so they alias just like in the flat world. Verification aligns the
matched 3D landmarks with a 3-point RANSAC + Kabsch. The optimizer is the real incremental GTSAM backend
(needs gtsam), so this exercises the full SE(3) chain: gate, consistency, retraction, iSAM2.

Used by tests/test_sim3d.py and slam_tools/loop_closure_demo3d.py.
"""
from dataclasses import dataclass, field

import numpy as np

from slam_core.types import Keyframe
from slam_core.types.pose import inv_T, make_T

from .se3 import rot_from_vec, rotvec
from .sim import PrototypeDetector, SEE, STEP  # noqa: F401  (re-exported for the demo)
from .types import Verification
from .verification import loop_covariance

H1_Z, H2_Z = 0.4, 1.0       # heights of the two identical hallways
RECT3 = [(0, 0, 0.0), (24, 0, 0.5), (24, 14, 1.5), (0, 14, 0.8), (0, 0, 0.0)]
ROUTE3 = (RECT3 + [(0, 7, H1_Z), (-12, 7, H1_Z), (0, 7, H1_Z), (0, 0, 0.0)] + RECT3[1:]
          + [(24, 0, 0.5), (24, 7, H2_Z), (36, 7, H2_Z), (24, 7, H2_Z), (24, 14, 1.5), (0, 14, 0.8), (0, 0, 0.0)])


# Points every STEP along a 3D polyline, corners rounded by a moving average
def walk3(corners, step=STEP):
    pts = [np.asarray(corners[0], float)]
    for a, b in zip(corners[:-1], corners[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(int(np.linalg.norm(b - a) / step), 1)
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    P = np.array(pts)
    k = 7
    pad = np.vstack([np.repeat(P[:1], k, 0), P, np.repeat(P[-1:], k, 0)])
    return np.array([pad[i:i + 2 * k + 1].mean(axis=0) for i in range(len(P))])


# Camera poses along the path: x forward, yaw from the heading, pitch from the slope, a gentle roll sway
def poses_along(P):
    d = np.gradient(P, axis=0)
    out = []
    for i, (p, di) in enumerate(zip(P, d)):
        yaw, pitch, roll = np.arctan2(di[1], di[0]), -np.arctan2(di[2], np.hypot(di[0], di[1])), 0.06 * np.sin(0.15 * i)
        cz, sz, cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        out.append(make_T(Rz @ Ry @ Rx, p))
    return out


@dataclass
class Scenario3D:
    gt_T: list                          # true 4x4 poses
    kfs: list
    descs: list
    obs: list                           # per keyframe {word: local xyz}
    landmarks: np.ndarray               # (M, 3)
    words: np.ndarray
    n_words: int
    rng: np.random.Generator = field(repr=False, default=None)

    @property
    def gt(self):                       # (N, 3) true positions
        return np.array([T[:3, 3] for T in self.gt_T])


def _landmarks(rng):
    pos = []
    edges = list(zip(RECT3[:-1], RECT3[1:]))
    for a, b in edges:
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b - a)[:2] / np.linalg.norm((b - a)[:2])
        n = np.array([-d[1], d[0]])
        for t in np.linspace(0, 1, max(int(np.linalg.norm(b - a) / 0.6), 2)):
            for side in (-1, 1):
                xy = (a + (b - a) * t)[:2] + side * n * rng.uniform(0.8, 1.8) + rng.normal(0, .2) * d
                pos.append([*xy, (a + (b - a) * t)[2] + rng.uniform(-0.5, 2.0)])
    n_main = len(pos)
    h1 = []
    for x in np.arange(-12.0, 0.0, 0.6):
        for side in (-1, 1):
            h1.append([x + rng.normal(0, .15), 7 + side * rng.uniform(0.8, 1.8), H1_Z + rng.uniform(-0.5, 2.0)])
    pos = np.array(pos + h1)
    h1 = np.array(h1)
    n_words = len(pos)
    # H2: the same landmarks (same words) rotated 180 degrees about the vertical axis through (12, 7), raised
    h2 = np.column_stack([24 - h1[:, 0], 14 - h1[:, 1], h1[:, 2] + (H2_Z - H1_Z)])
    return np.vstack([pos, h2]), np.concatenate([np.arange(n_words), np.arange(n_main, n_words)]), n_words


# Build the scenario: truth, drifting odometry (all three rotation axes), descriptors, observations
def make_scenario3d(seed=0, bias=(0.0004, 0.0006, 0.0012), rot_noise=0.002, trans_noise=0.01, scale=0.01,
                    rot_sigma=0.01, trans_sigma=0.02, dropout=0.1):
    rng = np.random.default_rng(seed)
    gt_T = poses_along(walk3(ROUTE3))
    lm, words, n_words = _landmarks(rng)
    cov = np.diag([rot_sigma] * 3 + [trans_sigma] * 3) ** 2
    kfs, descs, obs = [], [], []
    T_odom = gt_T[0].copy()
    for i, T_gt in enumerate(gt_T):
        if i == 0:
            T_prev = np.eye(4)
        else:
            rel = inv_T(gt_T[i - 1]) @ T_gt
            dR = rot_from_vec(np.array(bias) + rng.normal(0, rot_noise, 3))
            T_prev = make_T(rel[:3, :3] @ dR, rel[:3, 3] * (1 + scale) + rng.normal(0, trans_noise, 3))
            T_odom = T_odom @ T_prev
        kfs.append(Keyframe(id=i, prev_id=i - 1, stamp=0.5 * i, T_prev_kf=T_prev, covariance=cov, T_odom_kf=T_odom.copy()))

        Ti = inv_T(T_gt)
        local = (Ti[:3, :3] @ lm.T + Ti[:3, 3:4]).T
        vis = np.where(np.linalg.norm(local, axis=1) < SEE)[0]
        vis = vis[rng.random(len(vis)) > dropout]
        o = {int(words[k]): local[k] + rng.normal(0, 0.03, 3) for k in vis}
        obs.append(o)
        h = np.zeros(n_words)
        for w in o:
            h[w] += 1
        h[rng.integers(0, n_words, 3)] += 1
        descs.append(h / np.linalg.norm(h))
    return Scenario3D(gt_T, kfs, descs, obs, lm, words, n_words, np.random.default_rng(seed + 1))


# Rigid transform p_old = R p_new + t from 3D matches: 3-point RANSAC, then Kabsch on the inliers
def _kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def _rigid_ransac3(P_new, P_old, rng, thr=0.15, iters=80):
    best, n = None, len(P_new)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        a = P_new[idx]
        if np.linalg.norm(np.cross(a[1] - a[0], a[2] - a[0])) < 0.2:       # nearly collinear: unstable
            continue
        R, t = _kabsch(a, P_old[idx])
        inl = np.linalg.norm((R @ P_new.T).T + t - P_old, axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 4:
        return None, 0
    R, t = _kabsch(P_new[best], P_old[best])
    return make_T(R, t), int(best.sum())


def make_verifier3d(sc, wrong_match=0.08):
    def verify(old_id, new_id):
        common = sorted(set(sc.obs[old_id]) & set(sc.obs[new_id]))
        if len(common) < 5:
            return None
        P_old = np.array([sc.obs[old_id][w] for w in common])
        P_new = np.array([sc.obs[new_id][w] for w in common])
        bad = sc.rng.random(len(common)) < wrong_match
        P_new[bad] = sc.rng.uniform(-SEE, SEE, (int(bad.sum()), 3))
        T, n = _rigid_ransac3(P_new, P_old, sc.rng)
        return None if T is None else Verification(T, loop_covariance(n), n)
    return verify


# Is the loop's relative pose right? (within 0.5 m and 0.15 rad of the true 6-DoF relative pose)
def is_true_loop3d(sc, L, trans=0.5, rot=0.15):
    E = inv_T(inv_T(sc.gt_T[L.old_id]) @ sc.gt_T[L.new_id]) @ L.T_old_new
    return bool(np.linalg.norm(E[:3, 3]) < trans and np.linalg.norm(rotvec(E[:3, :3])) < rot)


def ate3d(poses, gt_T):
    return float(np.sqrt(np.mean([np.sum((T[:3, 3] - G[:3, 3]) ** 2) for T, G in zip(poses, gt_T)])))


# Runs a detector with the real incremental backend, keyframe by keyframe like a live system (needs gtsam)
# max_loop_chi2 is the backend's own post-optimization check on each loop (np.inf turns it off)
def run3d(sc, detector, max_loop_chi2=None):
    from slam_core.backend.incremental import IncrementalBackend

    be = IncrementalBackend() if max_loop_chi2 is None else IncrementalBackend(max_loop_chi2=max_loop_chi2)
    active = {}
    for kf, d in zip(sc.kfs, sc.descs):
        be.add_keyframe(kf)
        up = detector.process(kf, d)
        bad = be.apply(up)
        for L in up.added:
            active[L.id] = L
        for i in up.retracted + bad:
            active.pop(i, None)
    raw = [k.T_odom_kf for k in sc.kfs]
    opt = [be.pose(k.id) for k in sc.kfs]
    truth = [is_true_loop3d(sc, L) for L in active.values()]
    n_long = sum(t and L.new_id - L.old_id >= 100 for t, L in zip(truth, active.values()))
    return dict(loops=list(active.values()), true=truth, raw=raw, opt=opt,
                ate_raw=ate3d(raw, sc.gt_T), ate_opt=ate3d(opt, sc.gt_T),
                n_true=int(sum(truth)), n_false=int(len(truth) - sum(truth)), n_long=int(n_long))
