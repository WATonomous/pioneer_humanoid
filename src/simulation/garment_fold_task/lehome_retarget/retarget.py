"""LeHome SO-101 demo -> pioneer_bimanual_arm_8dof tool-point paths + IK (MuJoCo model of the 8dof arm, same world
placement as GarmentPioneerEnvCfg). Used offline (reach study) and inside the Isaac runner (online IK)."""
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("PIONEER_SRC", str(HERE.parents[2] / "pioneer_humanoid")))
from pioneer_humanoid.arm_8dof_params import (DEFAULT_JOINT_POS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,  # noqa: E402
                                              TCP_LOCAL)
from pioneer_humanoid.mujoco_bimanual_arm_8dof import arm_spec  # noqa: E402

from so101_fk import fk  # noqa: E402

DATA = Path(os.environ.get("LEHOME_DATA", Path.home() / "Desktop/lehome_ref/data"))
BASE_POS, BASE_QUAT = (0.0, -0.40, 0.95), (0.7071068, 0.0, 0.0, 0.7071068)  # GarmentPioneerEnvCfg robot_base_pos/rot
ARMS = {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS}
OPEN_ABOVE, CLOSE_BELOW = 0.10, 0.05   # LeHome gripper joint (rad), hysteresis: closed ~-0.15; some grasps open it only to ~0.18
TABLE_Z = 0.522             # apartment table top (settled cloth bottom, smoke test)
TCP_GRASP_Z = TABLE_Z + 0.022   # TCP is 25 mm above the fingertips: tips 3 mm into the table so the cloth is between them (the approach hovers)
W_DOWN, W_DOWN_GRASP = 0.02, 1.0   # gripper-down IK weight: soft, and vertical around grasps
END_Z = 0.12                    # the arms stay this high over the table once the fold is done
HOVER = 0.07                    # open gripper keeps this high over the table until a grasp is near
LEHOME_GRASP_Z = 0.499      # their gripper_frame_link height at table grasps
DZ = TCP_GRASP_Z - LEHOME_GRASP_Z


def build(base_pos=BASE_POS):
    spec = arm_spec()
    b = spec.body("base_link")
    b.pos, b.quat = base_pos, BASE_QUAT
    for side in ARMS:
        s = spec.body(f"{side}_link7").add_site()
        s.name, s.pos = f"{side}_tcp", TCP_LOCAL
    m = spec.compile()
    return m, mujoco.MjData(m)


class IK:
    """Damped least squares per arm on the tool point; the gripper pointing down (link7 +z up) is a soft goal, yaw free."""

    def __init__(self, base_pos=BASE_POS):
        self.m, self.d = build(base_pos)
        m = self.m
        self.qadr = {a: np.array([m.jnt_qposadr[m.joint(j).id] for j in js]) for a, js in ARMS.items()}
        self.dadr = {a: np.array([m.jnt_dofadr[m.joint(j).id] for j in js]) for a, js in ARMS.items()}
        self.lim = {a: m.jnt_range[[m.joint(j).id for j in js]] for a, js in ARMS.items()}
        self.site = {a: m.site(f"{a}_tcp").id for a in ARMS}
        self.body = {a: m.body(f"{a}_link7").id for a in ARMS}
        for j, v in DEFAULT_JOINT_POS.items():
            self.d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        self.home = {a: self.d.qpos[self.qadr[a]].copy() for a in ARMS}
        mujoco.mj_kinematics(m, self.d)

    def tcp(self, a):
        return self.d.site_xpos[self.site[a]].copy()

    def solve(self, a, target, iters=40, w_down=0.15, w_home=0.02):
        m, d = self.m, self.d
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        lo, hi = self.lim[a].T
        best = (np.inf, d.qpos[self.qadr[a]].copy(), np.inf)
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e_p = target - d.site_xpos[self.site[a]]
            z = d.xmat[self.body[a]].reshape(3, 3)[:, 2]
            e_r = np.cross(z, [0.0, 0.0, 1.0])
            err = np.linalg.norm(e_p)
            if err + 0.02 * np.linalg.norm(e_r) < best[0]:
                best = (err + 0.02 * np.linalg.norm(e_r), d.qpos[self.qadr[a]].copy(), err)
            mujoco.mj_jacSite(m, d, jp, None, self.site[a])
            mujoco.mj_jacBody(m, d, None, jr, self.body[a])
            J = np.vstack([jp[:, self.dadr[a]], w_down * jr[:, self.dadr[a]]])
            e = np.concatenate([e_p, w_down * e_r])
            n = len(self.dadr[a])
            q = d.qpos[self.qadr[a]]
            dq = np.linalg.solve(J.T @ J + 1e-3 * np.eye(n) + w_home * np.eye(n), J.T @ e + w_home * (self.home[a] - q) * 0.1)
            d.qpos[self.qadr[a]] = np.clip(q + np.clip(dq, -0.15, 0.15), lo, hi)
        d.qpos[self.qadr[a]] = best[1]
        mujoco.mj_kinematics(m, d)
        return best[1], best[2]


def load_episode(ep, kind="top_short"):
    import pandas as pd
    path = DATA / f"{kind}_merged/data/chunk-000/file-000.parquet"
    df = pd.read_parquet(path, filters=[("episode_index", "==", ep)])
    s = np.stack(df["observation.state"].values).astype(np.float64)
    gi = json.load(open(DATA / f"{kind}_merged/meta/garment_info.json"))
    name = f"{kind.title().replace('_', '_')}_Seen_{ep // 25}"
    name = "_".join(w.capitalize() for w in kind.split("_")) + f"_Seen_{ep // 25}"
    pose = gi[name][str(ep % 25)]["object_initial_pose"]
    out = {"garment": name, "pose": pose, "state": s}
    for side, sl in (("left", slice(0, 6)), ("right", slice(6, 12))):
        p, _ = fk(s[:, sl], side)
        out[side] = p + [0.0, 0.0, DZ]
        closed = np.zeros(len(s), bool)
        state = True                         # LeHome's grippers rest closed
        for f, v in enumerate(s[:, sl][:, 5]):
            state = False if v > OPEN_ABOVE else True if v < CLOSE_BELOW else state
            closed[f] = state
        # LeHome's operators also close the gripper in the air after letting go; ours would pinch the cloth still between
        # the fingers and carry it off. Keep it open until the next close at the table.
        tool = fk(s[:, sl], side)[0][:, 2] + DZ
        f = 1
        while f < len(closed):
            if closed[f] and not closed[f - 1] and tool[f] > TCP_GRASP_Z + 0.015:
                e = f
                while e < len(closed) and closed[e]:
                    e += 1
                closed[f:e] = False
                f = e
            f += 1
        # frames to the next close: descend over the last 10 frames before it, hover otherwise while open
        to_close = np.full(len(closed), 99.0)
        nxt = 99.0
        for f in range(len(closed) - 1, -1, -1):
            nxt = 0.0 if (f > 0 and closed[f] and not closed[f - 1]) else nxt + 1
            to_close[f] = nxt
        hover = np.where(closed, 0.0, HOVER * np.clip(to_close / 10.0, 0.0, 1.0))
        out[side][:, 2] = np.maximum(out[side][:, 2], TCP_GRASP_Z + hover)
        rel = [f for f in range(1, len(closed)) if closed[f - 1] and not closed[f]]
        if rel and not closed[rel[-1]:].any():   # after the last release LeHome's arms sink back to their rest height, onto our folded cloth
            out[side][rel[-1]:, 2] = np.maximum(out[side][rel[-1]:, 2], TCP_GRASP_Z + END_Z)
        out[side + "_closed"] = closed
        since = np.full(len(closed), 99.0)
        last = 99.0
        for f in range(len(closed)):
            last = 0.0 if (f > 0 and closed[f] and not closed[f - 1]) else last + 1
            since[f] = last
        out[side + "_wdown"] = np.where((to_close <= 12) | (since <= 4), W_DOWN_GRASP, W_DOWN)
    return out


def _track(ik, a, targets, f0, q0, w_down):
    w_down = np.broadcast_to(w_down, (len(targets),))
    """Warm-started tracking out from frame f0 (seeded at q0) in both directions."""
    T = len(targets)
    qs, errs = [None] * T, np.zeros(T)
    for rng_ in (range(f0, T), range(f0, -1, -1)):
        ik.d.qpos[ik.qadr[a]] = q0
        for f in rng_:
            qs[f], errs[f] = ik.solve(a, targets[f], iters=20, w_down=float(w_down[f]), w_home=0.0)
    qs = np.array(qs)
    jump = np.abs(np.diff(qs, axis=0)).max() if T > 1 else 0.0
    return qs, errs, jump


def plan_track(ik, a, targets, starts=8, seeds=12, rounds=3, seed=0, w_down=0.05):
    """Joint path (T, 7) tracking targets (T, 3) in one IK branch, no joint jumps. Seeds solutions at frame 0 and then
    at the current best path's worst frame, tracks each out to the whole path, keeps the lowest worst error."""
    rng = np.random.default_rng(seed)
    lo, hi = ik.lim[a].T
    best = None

    def consider(f0, q_init):
        nonlocal best
        ik.d.qpos[ik.qadr[a]] = q_init
        q0, e0 = ik.solve(a, targets[f0], iters=200, w_down=float(np.broadcast_to(w_down, (len(targets),))[f0]), w_home=0.0)
        if e0 > 0.003:
            return
        qs, errs, jump = _track(ik, a, targets, f0, q0, w_down)
        score = errs.max() + 0.05 * max(jump - 0.1, 0.0)
        if best is None or score < best[0]:
            best = (score, qs, errs, jump)

    for k in range(starts):
        consider(0, ik.home[a] if k == 0 else rng.uniform(lo, hi))
    for _ in range(rounds):
        if best is not None and best[2].max() < 0.003 and best[3] < 0.15:
            break
        f0 = int(np.argmax(best[2])) if best is not None else 0
        for _ in range(seeds):
            consider(f0, rng.uniform(lo, hi))
    return best[1], best[2], best[3]


# The fold LeHome's operators do on each garment type (checked on their demos): per arm, the keypoint each table grasp
# takes (check_point order), and where that keypoint is laid. kp: the settled garment's keypoints (N, 3).
SEQUENCE = {
    "top_short": {"left": [2, 4], "right": [3, 5]},   # sleeve tip to the middle, then hem corner onto the shoulder
    "top_long": {"left": [2, 4], "right": [3, 5]},    # long sleeve down to the hem's middle, then hem corner onto the shoulder
    "pant_short": {"left": [0], "right": [4]},        # fold in half: 0 onto 1, 4 onto 5
    "pant_long": {"left": [1], "right": [5, 4]},      # 1 onto 0, 5 onto 4, then 4 onto 0
}
OVERSHOOT = 0.02   # m past the goal: the cloth springs back a little after a release


def fold_goal(kind, k, kp):
    xy = kp[:, :2]
    if kind == "top_short":
        mid = xy[[0, 1, 4, 5]].mean(0)
        # sleeves 3 cm past the middle (they spring back ~7 cm) and 4 cm down toward the hem (straight across pulls the
        # shoulders together), hem corners 2.5 cm past the shoulders
        goals = {2: mid + [0.03, -0.04], 3: mid + [-0.03, -0.04], 4: xy[0] + [0.0, 0.025], 5: xy[1] + [0.0, 0.025]}
    elif kind == "top_long":
        hem = xy[[4, 5]].mean(0) + [0.0, 0.03]
        goals = {2: hem + [OVERSHOOT, 0.0], 3: hem - [OVERSHOOT, 0.0], 4: xy[0] + [0.0, 0.025], 5: xy[1] + [0.0, 0.025]}
    else:
        onto = {"pant_short": {0: 1, 4: 5}, "pant_long": {1: 0, 5: 4, 4: 0}}[kind]
        goals = {}
        for src, dst in onto.items():
            d = xy[dst] - xy[src]
            goals[src] = xy[dst] + OVERSHOOT * d / max(np.linalg.norm(d), 1e-6)
    return goals.get(k)


LIFT, LIFT_FRAMES = 0.08, 10   # straight up after a release
LAY_Z, LAY_FRAMES = 0.015, 16  # lay the held cloth down this high over the table, slowly, before letting go (lower presses the fold into the layer below and it keeps sliding; higher springs back)


def snap_grasps(E, keypoints, kind, radius=0.15, blend=10):
    """Object-centric grasps: shift each table grasp (approach .. release) in xy so it lands on the nearest garment
    keypoint of our settled cloth (within radius); the shift blends in over the last `blend` frames of the approach.
    The carry also bends so that the keypoint is released at its fold goal (SEQUENCE / fold_goal)."""
    shifts = []
    for side in ("left", "right"):
        P, cl = E[side], E[side + "_closed"]
        events = np.nonzero(cl[1:] != cl[:-1])[0] + 1
        n_grasp = 0
        for f in events:
            if not cl[f] or P[f][2] > TCP_GRASP_Z + 0.015:
                continue
            n_grasp += 1
            opened = [e for e in events if e < f and not cl[e]]
            released = [e for e in events if e > f and not cl[e]]
            a, b = (opened[-1] if opened else 0), (released[0] if released else len(P))
            d = np.linalg.norm(keypoints[:, :2] - P[f][:2], axis=1)
            seq = SEQUENCE[kind][side]
            k = seq[n_grasp - 1] if n_grasp <= len(seq) else int(np.argmin(d))
            if d[k] > radius:
                continue
            off = keypoints[k, :2] - P[f][:2]
            w = np.clip((np.arange(a, b) - (f - blend - a) - a) / blend, 0.0, 1.0)
            w[np.arange(a, b) >= f] = 1.0
            P[a:b, :2] += w[:, None] * off
            goal = fold_goal(kind, k, keypoints)
            if goal is not None and b < len(P):
                fix = goal - P[b][:2]                       # P[b] already includes off: the grasped keypoint is there
                w2 = np.clip((np.arange(f, b) - f) / max(b - f, 1), 0.0, 1.0)
                P[f:b, :2] += w2[:, None] * fix
                P[b, :2] = P[b - 1, :2]
                lay = np.clip((np.arange(f, b + 1) - (b - LAY_FRAMES)) / LAY_FRAMES, 0.0, 1.0)
                P[f:b + 1, 2] = P[f:b + 1, 2] + lay * (np.minimum(P[f:b + 1, 2], TCP_GRASP_Z + LAY_Z) - P[f:b + 1, 2])
                # after letting go, lift straight up before following the demo again (moving off sideways from the
                # cloth drags the fold back open), then blend back into the demo path
                e = min(b + 1 + LIFT_FRAMES, len(P))
                up = np.linspace(0.0, LIFT, LIFT_FRAMES + 1)[1:e - b]
                P[b + 1:e, :2] = P[b, :2]
                P[b + 1:e, 2] = P[b, 2] + up
                e2 = min(e + LIFT_FRAMES, len(P))
                if e2 > e:
                    w = np.linspace(0.0, 1.0, e2 - e + 1)[1:, None]
                    P[e:e2] = (1 - w) * P[e - 1] + w * P[e:e2]
            shifts.append((side, int(f), k, np.round(off, 3).tolist()))
    return shifts
