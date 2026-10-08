"""Thumb opposition reach (a kinematic Kapandji test) and a search over thumb mounts. No physics.

For a thumb mount, sample the thumb's joint space and each finger's flexion space (spread at 0), take the
fingertip sites, and score how much of each finger's fingertip workspace the thumb tip can meet:

  overlap_i   fraction of finger i's fingertip samples within CONTACT_R of some thumb-tip sample
  reach_i     closest thumb-tip to finger-i-tip distance (m); < CONTACT_R means the tips can touch

The mount search moves the thumb base (x, y in the palm frame) and turns the chain about the palm normal,
like thumb.THUMB_MOUNTS, and keeps the mounts with the best mean overlap. Only positions, not pad
orientation, are checked. A mount is ``feasible`` only if the thumb's base links (the ones that don't move
with its flexion: thumb, thumb_abduction) don't collide with any finger, with the fingers straight,
half-curled and fully curled -- the best unconstrained mounts put the base inside the finger roots.

    python -m hand_synergies.opposition --out out/opposition.csv
"""
from __future__ import annotations

import argparse
import itertools

import mujoco
import numpy as np
from pioneer_humanoid.mujoco_hand import JOINT_NAMES, hand_spec, joint_ranges
from scipy.spatial import cKDTree

from .thumb import STOCK_POS, THUMB_MOUNTS

CONTACT_R = 0.01
FINGERS = ("1", "2", "3", "4")
FINGER_NAMES = ("index", "middle", "ring", "pinky")
THUMB_GRID = (5, 9, 5, 5)   # samples per thumb joint (circumduction, MCP_A, PIP, DIP)
FINGER_GRID = 8             # samples per finger flexion joint (MCP, PIP, DIP)


class Opposition:
    def __init__(self):
        self.model = hand_spec(finger_collisions=True).compile()
        self.data = mujoco.MjData(self.model)
        self.thumb_body = self.model.body("thumb").id
        ranges = joint_ranges(self.model)
        jid = {n: i for i, n in enumerate(JOINT_NAMES)}
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        thumb_axes = [np.linspace(*ranges[jid[n]], k) for n, k in
                      zip(("circumduction", "MCP_A_thumb", "PIP_thumb", "DIP_thumb"), THUMB_GRID)]
        self.thumb_q = np.array(list(itertools.product(*thumb_axes)))
        self.finger_q = {}
        for f in FINGERS:
            axes = [np.linspace(*ranges[jid[f"{j}_{f}"]], FINGER_GRID) for j in ("MCP", "PIP", "DIP")]
            self.finger_q[f] = np.array(list(itertools.product(*axes)))
        self.tip = {f: self.model.site(f"tip_distal_{f}").id for f in FINGERS}
        self.thumb_tip = self.model.site("tip_thumb_distal").id
        self.finger_tips = {f: self._tips([f"MCP_{f}", f"PIP_{f}", f"DIP_{f}"], self.finger_q[f], self.tip[f])
                            for f in FINGERS}  # the fingers don't depend on the thumb mount

    def _tips(self, names, qs, site) -> np.ndarray:
        idx = [self.qadr[JOINT_NAMES.index(n)] for n in names]
        out = np.empty((len(qs), 3))
        for i, q in enumerate(qs):
            self.data.qpos[:] = 0.0
            self.data.qpos[idx] = q
            mujoco.mj_kinematics(self.model, self.data)
            out[i] = self.data.site_xpos[site]
        return out

    def base_clear(self) -> bool:
        """No collision between the thumb's base links and any finger (fingers straight / half / curled)."""
        m, d = self.model, self.data
        base = {m.body("thumb").id, m.body("thumb_abduction").id}
        flex = [n for n in JOINT_NAMES if n[:3] in ("MCP", "PIP", "DIP") and not n.startswith("MCP_A")
                and not n.endswith("thumb")]
        lo, hi = joint_ranges(m)[[JOINT_NAMES.index(n) for n in flex]].T
        closed = np.where(np.abs(hi) > np.abs(lo), hi, lo)
        for frac in (0.0, 0.5, 1.0):
            d.qpos[:] = 0.0
            d.qpos[[self.qadr[JOINT_NAMES.index(n)] for n in flex]] = frac * closed
            mujoco.mj_forward(m, d)
            for c in d.contact[: d.ncon]:
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if c.dist < 0 and (b1 in base) != (b2 in base):
                    return False
        return True

    def score(self, pos, yaw_deg: float) -> dict:
        self.model.body_pos[self.thumb_body] = pos
        half = np.radians(yaw_deg) / 2
        self.model.body_quat[self.thumb_body] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        thumb = cKDTree(self._tips(["circumduction", "MCP_A_thumb", "PIP_thumb", "DIP_thumb"], self.thumb_q,
                                   self.thumb_tip))
        out = {}
        for f, name in zip(FINGERS, FINGER_NAMES):
            d, _ = thumb.query(self.finger_tips[f])
            out[f"overlap_{name}"] = float(np.mean(d < CONTACT_R))
            out[f"reach_{name}"] = float(d.min())
        out["overlap_mean"] = float(np.mean([out[f"overlap_{n}"] for n in FINGER_NAMES]))
        out["feasible"] = int(self.base_clear())
        out["fingers_touched"] = int(sum(out[f"reach_{n}"] < CONTACT_R for n in FINGER_NAMES))
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--candidates", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out/opposition.csv")
    args = p.parse_args()

    opp = Opposition()
    rows = []
    for name, (pos, yaw) in THUMB_MOUNTS.items():
        rows.append(dict(mount=name, x=pos[0], y=pos[1], yaw=yaw, **opp.score(pos, yaw)))
    rng = np.random.default_rng(args.seed)
    for i in range(args.candidates):
        pos = (rng.uniform(-0.05, 0.06), rng.uniform(0.0, 0.11), STOCK_POS[2])
        yaw = rng.uniform(-180.0, 180.0)
        rows.append(dict(mount=f"cand{i}", x=pos[0], y=pos[1], yaw=yaw, **opp.score(pos, yaw)))

    keys = list(rows[0])
    with open(args.out, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    show = ("overlap_index", "overlap_middle", "overlap_ring", "overlap_pinky", "overlap_mean", "fingers_touched",
            "feasible")
    print(f"{'mount':10s} {'x':>7s} {'y':>6s} {'yaw':>6s} " + " ".join(f"{k.replace('overlap_', ''):>8s}" for k in show))
    best = sorted(rows[len(THUMB_MOUNTS):], key=lambda r: -r["overlap_mean"])[:4]
    best += sorted([r for r in rows[len(THUMB_MOUNTS):] if r["feasible"]], key=lambda r: -r["overlap_mean"])[:6]
    for r in rows[:len(THUMB_MOUNTS)] + best:
        print(f"{r['mount']:10s} {r['x']:+.3f} {r['y']:.3f} {r['yaw']:+6.0f} "
              + " ".join(f"{r[k]:8.3f}" if isinstance(r[k], float) else f"{r[k]:8d}" for k in show))


if __name__ == "__main__":
    main()
