"""Does a policy read the instruction? duplo, first step only: which brick does it go for.

    MUJOCO_GL=egl python duplo_first_brick.py --policy <pretrained_model> --episodes 100 --seed 1000

Every duplo episode announces "put the <colour> brick on the baseplate" with the colour and the bricks' places
shuffled, so a policy that ignores the instruction picks the right brick one time in three. No precise
placement is needed: the brick counted is the first one the policy moves (15 mm), or, if it moves none within
--max_s, the one its wrist came closest to.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import sim_policy_eval as E  # noqa: E402
from synthetic_teleop import TeleopSim  # noqa: E402

MOVED = 0.015


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_s", type=float, default=15.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    from humanoid_mujoco_scenes.duplo import scene as S

    path = Path(args.policy).resolve()
    policy, pre, post, cfg, cameras = E.load_policy(path, args.device)
    colours, results = list(S.COLOURS), []
    for i in range(args.episodes):
        seed = args.seed + i
        sim = TeleopSim("duplo", seed, {n: E.RENDER_SIZE for n in cameras})
        model, data = sim.model, sim.data
        policy.reset(), pre.reset(), post.reset()
        order = S._order(data)
        asked = order[0]
        bricks = [model.body(f"brick_{c}").id for c in colours]
        wrist = model.jnt_bodyid[model.joint(sim.joints[-1]).id]
        start, nearest, moved = data.xpos[bricks].copy(), np.full(len(colours), np.inf), None
        for _ in range(int(args.max_s * 25)):
            _, _, instruction = sim.progress(model, data)
            E.policy_frame(sim, policy, pre, post, torch.device(args.device), 25, instruction, sizes=cameras)
            nearest = np.minimum(nearest, np.linalg.norm(data.xpos[bricks] - data.xpos[wrist], axis=1))
            shift = np.linalg.norm(data.xpos[bricks] - start, axis=1)
            if shift.max() > MOVED:
                moved = colours[int(shift.argmax())]
                break
        sim.close()
        chosen = moved or colours[int(nearest.argmin())]
        results.append({"seed": seed, "asked": asked, "order": order, "chosen": chosen, "moved": moved is not None})
        right = sum(r["asked"] == r["chosen"] for r in results)
        print(f"[FIRST] seed {seed}: asked {asked:6s} went for {chosen:6s} ({'moved' if moved else 'nearest'})   "
              f"right {right}/{len(results)}", flush=True)
    right = sum(r["asked"] == r["chosen"] for r in results)
    moved = [r for r in results if r["moved"]]
    summary = {"policy": str(path), "episodes": len(results), "right": right, "chance": len(results) / 3,
               "right_among_moved": sum(r["asked"] == r["chosen"] for r in moved), "moved": len(moved), "results": results}
    out = Path(args.out) if args.out else path.parent / "duplo_first_brick.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[FIRST] right brick {right}/{len(results)} (chance {len(results) / 3:.0f}); "
          f"among the {len(moved)} where a brick moved: {summary['right_among_moved']}  -> {out}")


if __name__ == "__main__":
    main()
