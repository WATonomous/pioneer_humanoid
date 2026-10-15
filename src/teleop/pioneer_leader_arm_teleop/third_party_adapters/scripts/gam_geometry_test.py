"""3D accuracy of GAM's per-frame features on peg_insert: their FrameFeaturizer with slots [red peg, blue block],
(a) as configured (monocular detector, predicted depth) and (b) with the simulator's exact depth image as a sensor
(detector_depth zero, geometry_depth sensor: their RGB-D recipe). xyz is compared with the true peg centre and the
true hole-block centre (exact replay of each take).
    python gam_geometry_test.py   (from the GAM repo, venv-gam)
"""
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/teleop/pioneer_leader_arm_teleop"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import synthetic_teleop  # noqa: F401  (puts the repo packages on sys.path)
import numpy as np

import GAM.precompute as P
from GAM.config import build_extractor, load_config, perception_settings
from GAM.data.episodes import list_episodes, load_episode

EPISODES, FRAMES = [0, 1, 2], [0, 0.25, 0.5, 0.75, 0.99]
takes = [json.loads(line) for line in open(REPO / "datasets/pioneer_v1_left_arm/sim_synthetic/peg_insert/meta/synthetic_takes.jsonl")]


def replay_qpos(seed):
    from human_operator import HumanOperator, Style
    from synthetic_tasks import TASKS
    from synthetic_teleop import TeleopSim
    sim = TeleopSim("peg_insert", seed, {"ego": (480, 640)})
    qs = []
    sim.frame_sink = lambda a, s, t, e, q: qs.append(q)
    sim.record_every = 4
    rng = np.random.default_rng([seed, 1])
    op = HumanOperator(sim.model, lambda: sim.data, sim.tick, sim.joints, sim.home, sim.control_dt, rng, style=Style.sample(rng, 1.0))
    TASKS["peg_insert"].plan(op, sim.model, sim.data, [])
    return sim, np.array(qs)


def run(depth_mode):
    cfg = load_config("configs/pioneer_peg.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["data"]["slots"] = ["red peg", "blue block"]
    cfg["data"]["clicks"] = {"source": "none"}
    if depth_mode == "sensor":
        cfg["perception"]["detector_depth"], cfg["perception"]["geometry_depth"] = "sensor", "sensor"   # their LIBERO recipe feeds the depth to the detector (sim)
        cfg["perception"]["geometry_source"] = os.environ.get("GEOM_SRC", cfg["perception"]["geometry_source"])
    pc = perception_settings(cfg)
    ext = build_extractor(pc)
    paths = list_episodes(cfg)
    errs = {"red peg": [], "blue block": []}
    for e in EPISODES:
        ep = load_episode(paths[e], cfg)
        sim, qs = replay_qpos(takes[e]["seed"])
        m = sim.model; d = mujoco.MjData(m)
        r = mujoco.Renderer(m, 480, 640); r.enable_depth_rendering()
        def depth_m(t):
            d.qpos[:] = qs[t]; mujoco.mj_forward(m, d); r.update_scene(d, camera="ego"); return r.render().astype(np.float32)
        if depth_mode == "sensor":
            ep.depth_m = depth_m
        f = P.FrameFeaturizer(cfg, ext, P.build_arm_mask_fn(cfg)); f.reset(ep)   # their URDF robot silhouette, as in GAM.precompute
        for fr in FRAMES:
            t = int(fr * (ep.T - 1))
            o = f.frame(ep, t)
            d.qpos[:] = qs[t]; mujoco.mj_forward(m, d)
            true = {"red peg": d.body("peg").xpos.copy(), "blue block": d.body("hole_block").xpos.copy()}
            for k, name in enumerate(errs):
                errs[name].append(np.linalg.norm(np.asarray(o["xyz"][k], float) - true[name]))
            if e == 0:
                np.set_printoptions(precision=3, suppress=True)
                print(f"t={t:3d} block det xyz {np.asarray(o['xyz'][1], float)} true {true['blue block']} | boxgeo {np.asarray(o['boxgeo'][1], float)} | box2d {np.asarray(o['box2d'][1]).round(0)}", flush=True)
        r.close(); ep.close()
    for name, v in errs.items():
        v = np.array(v)
        print(f"[{depth_mode:6s}] {name:10s}: 3D error median {np.median(v):.3f} m, max {v.max():.3f} m, within 2 cm {np.mean(v < 0.02):.0%}", flush=True)


for mode in ("sensor",):
    run(mode)
