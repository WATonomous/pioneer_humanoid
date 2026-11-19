"""A synthetic world for testing loop closure without a camera.

A robot walks a rectangular building twice plus two side hallways. The two hallways are IDENTICAL copies (same
landmark words, same local geometry), which is the classic perceptual-aliasing trap: they look the same, verify
geometrically, but are 36 m apart. Odometry drifts (a small constant turn bias, scale error and noise), with a
reported covariance. Each keyframe "sees" landmarks within a radius and gets a bag-of-words histogram as its
place-recognition descriptor. Verification aligns the matched landmarks (with some wrong matches mixed in) using
a small RANSAC, so a false loop is a perfectly consistent but wrong transform, as in real life.

Used by the tests and by slam_tools/loop_closure_demo.py. Not a model of a real camera: it checks the logic
(gating, confirmation, consistency, retraction), not feature quality.
"""
from dataclasses import dataclass, field

import numpy as np

from slam_core.types import Keyframe
from slam_core.types.pose import inv_T

from .se3 import planar_pose
from .types import LoopConstraint, LoopUpdate, Verification
from .verification import loop_covariance

STEP = 0.5          # m between keyframes
SEE = 5.0           # m, landmarks closer than this are seen


# Points every STEP metres along a polyline, with the corners rounded a little
def walk(corners, step=STEP):
    pts = [np.asarray(corners[0], float)]
    for a, b in zip(corners[:-1], corners[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(int(np.linalg.norm(b - a) / step), 1)
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    P = np.array(pts)
    k = 7
    pad = np.vstack([np.repeat(P[:1], k, 0), P, np.repeat(P[-1:], k, 0)])
    Ps = np.array([pad[i:i + 2 * k + 1].mean(axis=0) for i in range(len(P))])
    d = np.gradient(Ps, axis=0)
    return np.column_stack([Ps, np.arctan2(d[:, 1], d[:, 0])])


RECT = [(0, 0), (24, 0), (24, 14), (0, 14), (0, 0)]
# lap 1, hallway on the left, lap 2, then the identical hallway on the right, back around
ROUTE = (RECT + [(0, 7), (-12, 7), (0, 7), (0, 0)] + RECT[1:]
         + [(24, 0), (24, 7), (36, 7), (24, 7), (24, 14), (0, 14), (0, 0)])


@dataclass
class Scenario:
    gt: np.ndarray                      # (N, 3) true x, y, yaw
    kfs: list                           # Keyframe, with the raw odometry as T_odom_kf
    descs: list                         # place recognition descriptor per keyframe
    obs: list                           # per keyframe {word: local xy}
    landmarks: np.ndarray               # (M, 2)
    words: np.ndarray                   # (M,)
    n_words: int
    rng: np.random.Generator = field(repr=False, default=None)


# Landmarks along both sides of every corridor of the building, then H2 as a rotated copy of H1
def _landmarks(rng):
    pos = []
    walls = [((0, 0), (24, 0)), ((24, 0), (24, 14)), ((24, 14), (0, 14)), ((0, 14), (0, 0))]
    for a, b in walls:
        t = np.linspace(0, 1, max(int(np.linalg.norm(np.subtract(b, a)) / 0.6), 2))
        d = np.subtract(b, a) / np.linalg.norm(np.subtract(b, a))
        n = np.array([-d[1], d[0]])
        for ti in t:
            for side in (-1, 1):
                pos.append(np.add(a, (np.subtract(b, a)) * ti) + side * n * rng.uniform(0.8, 1.8) + rng.normal(0, .2, 2) * d)
    h1 = []
    for x in np.arange(-12.0, 0.0, 0.6):
        for side in (-1, 1):
            h1.append([x + rng.normal(0, .15), 7 + side * rng.uniform(0.8, 1.8)])
    n_main = len(pos)
    pos = np.array(pos + h1)
    words = np.arange(len(pos))
    h1 = np.array(h1)
    n_words = len(pos)
    h2 = np.column_stack([24 - h1[:, 0], 14 - h1[:, 1]])           # 180 degree rotation about (12, 7)
    words_h1 = np.arange(n_main, n_main + len(h1))
    return np.vstack([pos, h2]), np.concatenate([words, words_h1]), n_words


# Build the whole scenario: ground truth, noisy odometry, descriptors, observations
def make_scenario(seed=0, route=None, bias=0.0006, yaw_noise=0.002, trans_noise=0.01, scale=0.01,
                  rot_sigma=0.006, trans_sigma=0.02, dropout=0.1, wrong_match=0.08):
    rng = np.random.default_rng(seed)
    gt = walk(ROUTE if route is None else route)
    lm, words, n_words = _landmarks(rng)
    n = len(gt)

    kfs, descs, obs = [], [], []
    T_odom = planar_pose(*gt[0])
    cov = np.diag([rot_sigma] * 3 + [trans_sigma] * 3) ** 2
    for i in range(n):
        T_gt = planar_pose(*gt[i])
        # Odometry measurement from the previous keyframe: truth + scale error + constant turn bias + noise
        if i == 0:
            T_prev = np.eye(4)
        else:
            rel = inv_T(planar_pose(*gt[i - 1])) @ T_gt
            ry = np.arctan2(rel[1, 0], rel[0, 0])
            T_prev = planar_pose(rel[0, 3] * (1 + scale) + rng.normal(0, trans_noise),
                                 rel[1, 3] + rng.normal(0, trans_noise), ry + bias + rng.normal(0, yaw_noise))
            T_odom = T_odom @ T_prev
        kfs.append(Keyframe(id=i, prev_id=i - 1, stamp=0.5 * i, T_prev_kf=T_prev, covariance=cov, T_odom_kf=T_odom.copy()))

        # What this keyframe sees: landmarks within SEE, some dropped, some wrongly matched later
        R = inv_T(T_gt)
        local = (R[:2, :2] @ lm.T + R[:2, 3:4]).T
        vis = np.where(np.linalg.norm(local, axis=1) < SEE)[0]
        vis = vis[rng.random(len(vis)) > dropout]
        o = {int(words[k]): local[k] + rng.normal(0, 0.03, 2) for k in vis}
        obs.append(o)
        h = np.zeros(n_words)
        for w in o:
            h[w] += 1
        spurious = rng.integers(0, n_words, 3)       # a few words that are not really there
        h[spurious] += 1
        descs.append(h / np.linalg.norm(h))
    return Scenario(gt, kfs, descs, obs, lm, words, n_words, np.random.default_rng(seed + 1))


# Local feature descriptors per keyframe (a fixed random unit vector per landmark word plus noise), the kind of
# input Vlad takes, so the whole chain descriptors -> VLAD -> retrieval can be tested on the simulator too
def local_descriptors(sc, dim=64, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(sc.n_words, dim))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return [vec[sorted(o)] + rng.normal(0, noise, (len(o), dim)) for o in sc.obs]


# 2D rigid transform p_old = R p_new + t from matched points, by RANSAC then least squares on the inliers
def _rigid_ransac(P_new, P_old, rng, thr=0.15, iters=60):
    best = None
    n = len(P_new)
    for _ in range(iters):
        a, b = rng.choice(n, 2, replace=False)
        v1, v2 = P_new[b] - P_new[a], P_old[b] - P_old[a]
        if np.linalg.norm(v1) < 0.5:
            continue
        th = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        t = P_old[a] - R @ P_new[a]
        inl = np.linalg.norm((R @ P_new.T).T + t - P_old, axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        return None, 0
    A, B = P_new[best], P_old[best]
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return planar_pose(*(cb - R @ ca), np.arctan2(R[1, 0], R[0, 0])), int(best.sum())


# A verifier for the detector: matches landmarks by word id between two keyframes, mixes in wrong matches, aligns
def make_verifier(sc, wrong_match=0.08):
    def verify(old_id, new_id):
        common = sorted(set(sc.obs[old_id]) & set(sc.obs[new_id]))
        if len(common) < 4:
            return None
        P_old = np.array([sc.obs[old_id][w] for w in common])
        P_new = np.array([sc.obs[new_id][w] for w in common])
        bad = sc.rng.random(len(common)) < wrong_match
        P_new[bad] = sc.rng.uniform(-SEE, SEE, (int(bad.sum()), 2))
        T, n = _rigid_ransac(P_new, P_old, sc.rng)
        if T is None:
            return None
        return Verification(T, loop_covariance(n), n)
    return verify


# The detection the prototype does today (vo.py _detect_loop), reproduced on the simulator for comparison:
# old keyframes within 1 m of the current VO position and 10 s old, the 3 closest, first one to verify wins,
# cooldown of 10 keyframes. Same interface as LoopClosureDetector.process
class PrototypeDetector:
    def __init__(self, verifier, min_inliers=10, radius=1.0, min_time=10.0, max_cands=3, cooldown=10):
        self.verifier, self.min_inliers, self.radius = verifier, min_inliers, radius
        self.min_time, self.max_cands, self.cooldown = min_time, max_cands, cooldown
        self.kfs, self.last, self.n = [], -10 ** 9, 0

    def process(self, kf, desc):
        update = LoopUpdate()
        if kf.id - self.last >= self.cooldown:
            t = kf.T_odom_kf[:3, 3]
            old = [k for k in self.kfs if k.stamp < kf.stamp - self.min_time and np.linalg.norm(k.T_odom_kf[:3, 3] - t) < self.radius]
            old.sort(key=lambda k: np.linalg.norm(k.T_odom_kf[:3, 3] - t))
            for k in old[:self.max_cands]:
                v = self.verifier(k.id, kf.id)
                if v is not None and v.n_inliers >= self.min_inliers:
                    update.added.append(LoopConstraint(self.n, k.id, kf.id, v.T_old_new, loop_covariance(v.n_inliers),
                                                       v.n_inliers, 0.0, kf.stamp))
                    self.n += 1
                    self.last = kf.id
                    break
        self.kfs.append(kf)
        return update


# Ground truth: is the loop's relative pose right? (within 0.5 m and 0.15 rad of the true relative pose).
# An aliased hallway match fails this by tens of metres
def is_true_loop(sc, L, trans=0.5, rot=0.15):
    T_true = inv_T(planar_pose(*sc.gt[L.old_id])) @ planar_pose(*sc.gt[L.new_id])
    E = inv_T(T_true) @ L.T_old_new
    return bool(np.linalg.norm(E[:2, 3]) < trans and abs(np.arctan2(E[1, 0], E[0, 0])) < rot)


# RMSE of position against ground truth, no alignment (all runs start at the true start pose)
def ate(est, gt):
    return float(np.sqrt(np.mean(np.sum((est[:, :2] - gt[:, :2]) ** 2, axis=1))))


# Runs a detector over the whole scenario, applies its updates (adds and retractions), and optimizes.
# backend "planar" is the scipy stand-in (batch, at the end); "isam2" is the real incremental GTSAM backend,
# fed keyframe by keyframe like a live run (needs gtsam)
def run(sc, detector, optimize=True, backend="planar"):
    from .planar_pgo import _planar, optimize as pgo

    active = {}
    x0 = np.array([_planar(k.T_odom_kf) for k in sc.kfs])
    if backend == "isam2":
        from slam_core.backend.incremental import IncrementalBackend
        be = IncrementalBackend()
        for kf, d in zip(sc.kfs, sc.descs):
            be.add_keyframe(kf)
            up = detector.process(kf, d)
            bad = be.apply(up)
            for L in up.added:
                active[L.id] = L
            for i in up.retracted + bad:
                active.pop(i, None)
        opt = np.array([_planar(be.pose(k.id)) for k in sc.kfs])
    else:
        for kf, d in zip(sc.kfs, sc.descs):
            up = detector.process(kf, d)
            for L in up.added:
                active[L.id] = L
            for i in up.retracted:
                active.pop(i, None)
        odom = [(k.prev_id, k.id, k.T_prev_kf, k.covariance) for k in sc.kfs[1:]]
        loops = [(L.old_id, L.new_id, L.T_old_new, L.covariance) for L in active.values()]
        opt = pgo(x0, odom, loops) if optimize else x0

    truth = [is_true_loop(sc, L) for L in active.values()]
    # Long-range loops (revisits 100+ keyframes later, so drift has built up) are the ones that fix the map
    n_long = sum(t and L.new_id - L.old_id >= 100 for t, L in zip(truth, active.values()))
    return dict(loops=list(active.values()), true=truth, x0=x0, opt=opt,
                ate_raw=ate(x0, sc.gt), ate_opt=ate(opt, sc.gt),
                n_true=int(sum(truth)), n_false=int(len(truth) - sum(truth)), n_long=int(n_long))
