"""How well does GAM's general detector ground the peg_insert objects under different prompts?

Runs their extractor (perception block of configs/pioneer_peg.yaml) on frames of recorded episodes and compares each
slot's 2D box centre and 3D centre with the simulator's true peg / hole positions (exact replay of the take's seed).
    python gam_prompt_test.py   (from the GAM repo, venv-gam)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/teleop/pioneer_leader_arm_teleop"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from GAM.config import build_extractor, load_config, perception_settings
from GAM.data.episodes import list_episodes, load_episode

PROMPTS = [["peg", "hole"], ["red peg", "blue block"], ["red block", "blue block"], ["red stick", "blue box with a square hole"]]
EPISODES, FRAMES = [0, 1, 2], [0, 0.25, 0.5, 0.75, 0.99]

cfg = load_config("configs/pioneer_peg.yaml")
d = dict(cfg["data"], clicks={"source": "none"})
pc = perception_settings(cfg)
ext = build_extractor(pc)
K, E = np.array(d["camera_intrinsics"], float), np.array(d["camera_extrinsics"], float)[:3]
takes = [json.loads(line) for line in open(REPO / "datasets/pioneer_v1_left_arm/sim_synthetic/peg_insert/meta/synthetic_takes.jsonl")]
paths = list_episodes(dict(cfg, data=d))


def truth(seed, n):
    """True peg centre and hole-mouth centre (world) for every recorded frame, by exact replay."""
    from grounding_state import replay
    _, g = replay("peg_insert", seed, 1.0)
    from humanoid_mujoco_scenes.peg_insert import scene as P
    hole = np.c_[g[:, 6:8], np.full(len(g), P.TABLE_TOP_Z + P.BLOCK_HEIGHT)]
    return g[:n, :3], hole[:n]


def pix(p):
    c = E @ np.append(p, 1.0)
    return np.array([K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]])


rows = {tuple(p): {"px": [], "m": [], "miss": 0, "n": 0} for p in PROMPTS}
for e in EPISODES:
    ep = load_episode(paths[e], dict(cfg, data=d))
    peg, hole = truth(takes[e]["seed"], ep.T)
    for f in FRAMES:
        t = int(f * (ep.T - 1))
        rgb = ep.rgb(t)
        for p in PROMPTS:
            r = ext(rgb, np.zeros(rgb.shape[:2], np.float32), ep.K[t].astype(float), ep.E[t].astype(float), [], texts=p)
            for k, true in enumerate((peg[t], hole[t])):
                det, row = r.detections[k], rows[tuple(p)]
                row["n"] += 1
                ok = det.score >= float(pc.get("text_score_min", 0.3)) and det.box2d[2] > det.box2d[0]
                if not ok:
                    row["miss"] += 1
                    continue
                centre = np.array([(det.box2d[0] + det.box2d[2]) / 2, (det.box2d[1] + det.box2d[3]) / 2])
                row["px"].append(np.linalg.norm(centre - pix(true)))
                xyz = getattr(det, "xyz", None)
                if xyz is None and getattr(det, "box3d", None) is not None:
                    xyz = np.asarray(det.box3d)[:3]
                if xyz is not None:
                    row["m"].append(np.linalg.norm(np.asarray(xyz, float)[:3] - true))
    ep.close()
    print(f"episode {e} done", flush=True)

for p, row in rows.items():
    px = np.array(row["px"]); m = np.array(row["m"])
    print(f"{' / '.join(p):42s} missed {row['miss']}/{row['n']} | 2D centre error px: median {np.median(px) if len(px) else float('nan'):6.1f}, "
          f"within 25 px {np.mean(px < 25) if len(px) else 0:4.0%} | 3D error m: median {np.median(m) if len(m) else float('nan'):.3f}")
