"""Where does the peg's tilt come from? Compare the scripted operator's takes with a policy's rollouts.

    python peg_tilt_probe.py --policy <ckpt>/pretrained_model --out tilt.json [--takes 200] [--rollouts 200]

For every episode: the peg's tilt from vertical at the moment it first arrives over the hole while held (within 15 mm
in xy, lifted clear of the block), the largest tilt while it was carried, and whether the episode succeeded. For the
scripted takes also how often the operator's own recoveries fired (crooked peg, regrasp, lift and retry).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

TABLE, BLOCK_TOP = 0.705, 0.705 + 0.06


class Tracker:
    def __init__(self):
        self.arrive = self.carry_max = None

    def __call__(self, sim):
        d = sim.data
        peg, block = d.body("peg"), d.body("hole_block")
        up = peg.xmat.reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(min(1.0, abs(up[2])))))
        bottom = peg.xpos[2] - 0.05 * abs(up[2])
        if bottom > TABLE + 0.02 and tilt < 60:                       # carried (lifted, not lying down)
            self.carry_max = tilt if self.carry_max is None else max(self.carry_max, tilt)
            if self.arrive is None and np.linalg.norm(peg.xpos[:2] - block.xpos[:2]) < 0.015 and bottom > BLOCK_TOP - 0.002:
                self.arrive = tilt


def summary(name, rows):
    a = np.array([r["arrive"] for r in rows if r["arrive"] is not None])
    ok = np.array([r["ok"] for r in rows if r["arrive"] is not None])
    out = {"episodes": len(rows), "success": float(np.mean([r["ok"] for r in rows])), "arrived_over_hole": len(a),
           "arrive_tilt_deg": {"median": float(np.median(a)), "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)), "max": float(a.max())},
           "share_over_3deg": float((a > 3).mean()), "share_over_6deg": float((a > 6).mean()),
           "success_if_tilt_le_3": float(ok[a <= 3].mean()) if (a <= 3).any() else None,
           "success_if_tilt_gt_3": float(ok[a > 3].mean()) if (a > 3).any() else None,
           "success_if_tilt_gt_6": float(ok[a > 6].mean()) if (a > 6).any() else None}
    print(f"[TILT] {name}: " + json.dumps(out), flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--takes", type=int, default=200)
    p.add_argument("--rollouts", type=int, default=200)
    args = p.parse_args()
    result = {}

    if args.takes:
        from synthetic_teleop import run_episode
        rows = []
        for seed in range(args.takes):
            t = Tracker()
            ok, sim_t, notes, _ = run_episode("peg_insert", seed, record_every=1, on_frame=t)
            rows.append({"seed": seed, "ok": ok, "arrive": t.arrive, "carry_max": t.carry_max, "notes": notes})
        result["scripted"] = summary("scripted operator", rows)
        events = {k: sum(any(k in n for n in r["notes"]) for r in rows) for k in ("crooked", "regrasp", "retry", "gave up")}
        result["scripted"]["takes_with_event"] = events
        result["scripted_rows"] = rows
        print(f"[TILT] scripted takes with each recovery: {events}", flush=True)

    if args.policy and args.rollouts:
        import qplan
        prop = qplan.Proposer(args.policy)
        rows = []
        for seed in range(1000, 1000 + args.rollouts):
            t = Tracker()
            qplan.PROBE = t
            ep = qplan.rollout("peg_insert", seed, prop, qplan.pick_first, n=1, z_std=0.0)
            rows.append({"seed": seed, "ok": ep["success"], "arrive": t.arrive, "carry_max": t.carry_max, "end": ep["diag"]})
        qplan.PROBE = None
        result["policy"] = summary("policy", rows)
        result["policy_rows"] = rows

    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
