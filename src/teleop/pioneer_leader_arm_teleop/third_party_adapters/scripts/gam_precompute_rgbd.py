"""GAM.precompute with depth: each recorded take is replayed from its seed (exact, verified against the recorded
states in grounding_state.py) to render the ego camera's depth image for every frame, which GAM reads as a depth
sensor (perception.detector_depth / geometry_depth: sensor). Everything else is GAM.precompute unchanged.

    python gam_precompute_rgbd.py --config configs/pioneer_peg_rgbd.yaml --ep-end 50   (from the GAM repo, venv-gam)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/teleop/pioneer_leader_arm_teleop"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import synthetic_teleop  # noqa: E402,F401  (puts the repo packages on sys.path)

import GAM.precompute as P  # noqa: E402

TAKES = {t["episode_index"]: t for t in map(json.loads, open(REPO / "datasets/pioneer_v1_left_arm/sim_synthetic/peg_insert/meta/synthetic_takes.jsonl"))}


def replay_qpos(scene, seed, pace):
    from human_operator import HumanOperator, Style
    from synthetic_tasks import TASKS
    from synthetic_teleop import TeleopSim
    sim = TeleopSim(scene, seed, {"ego": (480, 640)})
    qs = []
    sim.frame_sink = lambda a, s, t, e, q: qs.append(q)
    sim.record_every = 4
    rng = np.random.default_rng([seed, 1])
    op = HumanOperator(sim.model, lambda: sim.data, sim.tick, sim.joints, sim.home, sim.control_dt, rng, style=Style.sample(rng, pace))
    TASKS[scene].plan(op, sim.model, sim.data, [])
    return sim, np.array(qs)


_load = P.load_episode


def load_episode(path, cfg):
    ep = _load(path, cfg)
    take = TAKES[int(path.split("#")[-1])]
    sim, qs = replay_qpos(take["scene"], take["seed"], take["pace"])
    if len(qs) < ep.T:
        raise SystemExit(f"{path}: replay has {len(qs)} frames, the recording {ep.T}")
    m, d = sim.model, mujoco.MjData(sim.model)
    renderer = mujoco.Renderer(m, 480, 640)
    renderer.enable_depth_rendering()

    def depth_m(t):
        d.qpos[:] = qs[t]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera="ego")
        return renderer.render().astype(np.float32)

    ep.depth_m = depth_m
    close = ep.close

    def close_all():
        renderer.close(); sim.close(); close()

    ep.close = close_all
    return ep


P.load_episode = load_episode

if __name__ == "__main__":
    P.main()
