"""Perfect grounding: true object poses appended to the state, as an upper bound on what better perception could give.

peg_insert: the peg's pose and the hole's centre. If a policy that is told where they are still fails, the
cameras are not the limit.
duplo: where the brick the current instruction names is, and the next step's brick. The policy never sees a
colour word, only "the brick at this position": does a small policy follow the instruction once it is grounded?

    python grounding_state.py add <dataset_root>            # replays every recorded take (no cameras), edits in place
    python grounding_state.py eval <scene> --policy <pretrained_model> --episodes 200 --seed 1000 --max_s 40
    python grounding_state.py first --policy <pretrained_model> --episodes 100     # duplo_first_brick.py, grounded
    python grounding_state.py add <duplo dataset_root> --detect     # image positions from brick_detector.py instead of
    python grounding_state.py eval duplo_detect --policy ...        # true poses (GROUNDING_DETECT=1 for ``first``)

``add`` works on a copy of a ``synthetic_teleop.py --record`` dataset: each take is replayed from its seed
(meta/synthetic_takes.jsonl) and must reproduce the recorded states exactly. ``eval`` is sim_policy_eval.py with
the same values appended to the state it gives the policy. GROUNDING_SWAP=1 (duplo) reports the poses of the wrong
brick for the current step: a policy that uses the grounding then goes for that brick. GROUNDING_NOISE=0.01 (duplo)
adds 10 mm (1 sigma, drawn every frame) to the reported brick positions, as an imperfect detector would.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import synthetic_teleop  # noqa: E402,F401  (puts the repo packages on sys.path)

MARKER = "meta/grounding_state.json"
_NOISE = np.random.default_rng(0)


def _pose(model, qpos, joint: str, symmetry: float):
    """(xyz, w x y z, yaw folded into +-symmetry/2) of a free body."""
    adr = model.joint(joint).qposadr[0]
    pos, (w, x, y, z) = qpos[adr:adr + 3], qpos[adr + 3:adr + 7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return pos, (w, x, y, z), (yaw + symmetry / 2) % symmetry - symmetry / 2


def peg_insert(model, data, qpos) -> np.ndarray:
    """Peg position (m), how far its long axis leans in x and y, its yaw (rad, mod 90 deg: square peg), hole centre xy."""
    pos, (w, x, y, z), yaw = _pose(model, qpos, "peg", np.pi / 2)
    tilt = [2 * (x * z + w * y), 2 * (y * z - w * x)]                     # the peg's z axis, world x and y components
    return np.array([*pos, *tilt, yaw, *model.body_pos[model.body("hole_block").id, :2]], dtype=np.float32)


def duplo(model, data, qpos) -> np.ndarray:
    """This step's brick (xy, yaw mod 180 deg) and the next step's. Where it goes is left out: the stack is always
    built on the same spot of the baseplate, so that position carries nothing."""
    from humanoid_mujoco_scenes.duplo import scene as S

    order = S._order(data)
    k = min(int(data.userdata[S._U_STEP]), len(order) - 1)

    def brick(colour):
        pos, _, yaw = _pose(model, qpos, f"brick_{colour}", np.pi)
        return [pos[0], pos[1], yaw]

    now = order[(k + 1) % len(order)] if os.environ.get("GROUNDING_SWAP") else order[k]
    out = np.array([*brick(now), *brick(order[min(k + 1, len(order) - 1)])], dtype=np.float32)
    noise = float(os.environ.get("GROUNDING_NOISE", 0))                   # m, per frame, on the bricks' xy
    if noise:
        out[[0, 1, 3, 4]] += _NOISE.normal(0.0, noise, 4)
    return out


def duplo_ids(model, data, qpos) -> np.ndarray:
    """Colour indices of this step's brick and the next step's (what the instruction says, nothing about where)."""
    from humanoid_mujoco_scenes.duplo import scene as S

    order, names = S._order(data), list(S.COLOURS)
    k = min(int(data.userdata[S._U_STEP]), len(order) - 1)
    now = order[(k + 1) % len(order)] if os.environ.get("GROUNDING_SWAP") else order[k]
    return np.array([names.index(now), names.index(order[min(k + 1, len(order) - 1)])], dtype=np.float32)


SCENES = {
    "peg_insert": (peg_insert, ["peg_x", "peg_y", "peg_z", "peg_tilt_x", "peg_tilt_y", "peg_yaw", "hole_x", "hole_y"]),
    "duplo": (duplo, ["brick_x", "brick_y", "brick_yaw", "next_brick_x", "next_brick_y", "next_brick_yaw"]),
    "duplo_detect": (duplo_ids, None),      # the two bricks' image positions from brick_detector.py
}


def replay(scene: str, seed: int, pace: float):
    """(recorded states, grounding per recorded frame) of one take, without cameras."""
    from human_operator import HumanOperator, Style
    from synthetic_tasks import TASKS, TaskFailed
    from synthetic_teleop import TeleopSim

    fn, scene = SCENES[scene][0], scene.removesuffix("_detect")
    sim = TeleopSim(scene, seed)
    states, extra = [], []
    sim.frame_sink = lambda action, state, task, extras, qpos: (states.append(state), extra.append(fn(sim.model, sim.data, qpos)))
    sim.record_every = 4                                                   # 25 fps, as recorded
    rng = np.random.default_rng([seed, 1])
    op = HumanOperator(sim.model, lambda: sim.data, sim.tick, sim.joints, sim.home, sim.control_dt, rng, style=Style.sample(rng, pace))
    try:
        TASKS[scene].plan(op, sim.model, sim.data, [])
    except TaskFailed:
        pass
    sim.close()
    return np.stack(states), np.stack(extra)


def _replay(take):
    return take["episode_index"], *replay(take["scene"], take["seed"], take["pace"])


def add(root: str, detect: bool = False) -> None:
    import multiprocessing

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    if (root / MARKER).exists():
        raise SystemExit("already done")
    takes = [json.loads(line) for line in (root / "meta/synthetic_takes.jsonl").read_text().splitlines() if line.strip()]
    scene = takes[0]["scene"] + ("_detect" if detect else "")
    names = SCENES[scene][1]
    for take in takes:
        take["scene"] = scene
    with multiprocessing.get_context("spawn").Pool(8) as pool:
        replays = {ep: (s, g) for ep, s, g in pool.imap_unordered(_replay, takes)}
    if detect:
        import brick_detector
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        names, episodes, seen = brick_detector.NAMES, LeRobotDatasetMetadata("local", root=root).episodes, []
        for ep, (s, ids) in replays.items():
            video = root / "videos/observation.images.ego" / f"chunk-{episodes[ep]['videos/observation.images.ego/chunk_index']:03d}" \
                / f"file-{episodes[ep]['videos/observation.images.ego/file_index']:03d}.mp4"
            g, frac = brick_detector.from_video(video, ids[:episodes[ep]["length"]])
            replays[ep] = (s[:len(g)], g)
            seen.append(frac)
        print(f"[GROUNDING] this step's brick visible in {np.mean(seen):.0%} of frames (min episode {np.min(seen):.0%})")

    info = json.loads((root / "meta/info.json").read_text())
    feat = info["features"]["observation.state"]
    dim, k, states = feat["shape"][0], len(names), []
    for path in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        table = pq.read_table(path)
        state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
        episode, frame = table["episode_index"].to_numpy(), table["frame_index"].to_numpy()
        extra = np.zeros((len(state), k), dtype=np.float32)   # (the recording ends up to a second before the replay)
        for ep in np.unique(episode):
            rows = episode == ep
            s, g = replays[int(ep)]
            if len(s) < rows.sum() or np.abs(s[frame[rows]] - state[rows]).max() > 1e-5:
                raise SystemExit(f"episode {ep}: the replay does not reproduce the recorded states")
            extra[rows] = g[frame[rows]]
        new = np.concatenate([state, extra], 1)
        states.append(new)
        col = pa.FixedSizeListArray.from_arrays(pa.array(new.reshape(-1), pa.float32()), dim + k) \
            if pa.types.is_fixed_size_list(table.schema.field("observation.state").type) else pa.array(list(new), pa.list_(pa.float32()))
        meta = dict(table.schema.metadata or {})
        if b"huggingface" in meta:
            hf = json.loads(meta[b"huggingface"])
            hf["info"]["features"]["observation.state"]["length"] = dim + k
            meta[b"huggingface"] = json.dumps(hf).encode()
        i = table.schema.get_field_index("observation.state")
        pq.write_table(table.set_column(i, "observation.state", col).replace_schema_metadata(meta), path)
    feat["shape"] = [dim + k]
    feat["names"] = list(feat["names"]) + names
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    allv = np.concatenate(states)
    stats = json.loads((root / "meta/stats.json").read_text())
    old = stats["observation.state"]
    new = {"mean": allv.mean(0), "std": allv.std(0), "min": allv.min(0), "max": allv.max(0)}
    for q in [key for key in old if key.startswith("q")]:
        new[q] = np.quantile(allv, float(q[1:]) / 100, axis=0)
    for key, v in new.items():
        old[key] = np.asarray(v, dtype=np.float64).tolist()
    (root / "meta/stats.json").write_text(json.dumps(stats, indent=4))
    (root / MARKER).write_text(json.dumps({"scene": scene, "names": names}, indent=1))
    print(f"[GROUNDING] {root}: state {dim} -> {dim + k} over {len(replays)} episodes; std of the added values {allv[:, dim:].std(0).round(4)}")


def ground(scene: str) -> None:
    """Make sim_policy_eval's observations carry the scene's grounding."""
    import sim_policy_eval

    fn, observe = SCENES[scene][0], sim_policy_eval.observe
    trackers = {}

    def observe_grounded(sim):
        state, images = observe(sim)
        g = fn(sim.model, sim.data, sim.data.qpos)
        if scene.endswith("_detect"):                                    # a fresh tracker per episode (per sim)
            import brick_detector
            if id(sim) not in trackers:
                trackers.clear()
                trackers[id(sim)] = brick_detector.Tracker()
            g = trackers[id(sim)](images["ego"], int(g[0]), int(g[1]))
        return np.concatenate([state, g]), images

    sim_policy_eval.observe = observe_grounded


if __name__ == "__main__":
    command = sys.argv.pop(1)
    if command == "add":
        add(sys.argv[1], detect="--detect" in sys.argv)
    elif command == "eval":
        scene = "peg_insert" if sys.argv[1].startswith("--") else sys.argv.pop(1)
        sys.argv[1:1] = ["--scene", scene.removesuffix("_detect")]
        ground(scene)
        import sim_policy_eval
        sim_policy_eval.main()
    elif command == "first":
        ground("duplo_detect" if os.environ.get("GROUNDING_DETECT") else "duplo")
        import duplo_first_brick
        duplo_first_brick.main()
    else:
        raise SystemExit(__doc__)
