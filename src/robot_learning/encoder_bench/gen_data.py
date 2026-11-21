"""Benchmark dataset: synthetic-operator peg_insert episodes, rendered from the ego (D455) and wrist cameras.

Each successful episode (seed) gives
  * the 25 Hz state/action stream (for behaviour cloning),
  * images at 12.5 Hz (every 2nd recorded frame): RGB uint8 and depth uint16 (mm) for ego + wrist_left,
  * per-image geometric labels from the simulator (TCP, peg, hole, phase ...).

Test episodes are also rendered under visual shifts (``--variants``):
  light  - dimmer, warmer, re-aimed lights (RGB distribution shift; depth is unaffected)
  color  - different table / block / peg colours (appearance shift; depth unaffected)
  camera - HELD-OUT shift no encoder or augmentation ever trained on: both cameras moved 1.5-2.5 cm and tilted
           2-4 deg (a re-mounted / bumped camera). Changes RGB *and* depth, so it is saved with its own depth.
           Written to its own split, e.g. --split test_cam --variants camera.
  camjit - TRAINING augmentation (PioneerNet v4): a fresh small camera pose for every image, both cameras moved
           0-1.5 cm and tilted 0-2 deg. Strictly smaller than the held-out ``camera`` shift, so that test stays
           unseen (it then measures extrapolation). Written to its own split: --split train_camjit --variants camjit.

    .venv/bin/python gen_data.py --split train --seeds 0:120
    .venv/bin/python gen_data.py --split test --seeds 5000:5030 --variants clean,light,color
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

import common as C

IMG_EVERY = 2          # of the 25 Hz recorded frames -> 12.5 Hz images


def apply_variant(model, variant: str, rng: np.random.Generator) -> None:
    import mujoco
    if variant == "light":
        for i in range(model.nlight):
            model.light_diffuse[i] *= rng.uniform(0.35, 0.6) * np.array([1.0, 0.85, 0.65])
            model.light_ambient[i] *= 0.5
            model.light_pos[i, :2] += rng.uniform(-0.4, 0.4, 2)
        model.vis.headlight.ambient[:] = np.array([0.25, 0.2, 0.15])
        model.vis.headlight.diffuse[:] = np.array([0.35, 0.3, 0.2])
    elif variant == "color":
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("wall_"):
                model.geom_rgba[g, :3] = [0.2, 0.65, 0.25]          # blue block -> green
            elif name == "peg":
                model.geom_rgba[g, :3] = [0.9, 0.8, 0.15]           # red peg -> yellow
        for m in range(model.nmat):
            model.mat_rgba[m, :3] = np.clip(model.mat_rgba[m, :3] * rng.uniform(0.6, 0.9, 3), 0, 1)
    elif variant in ("camera", "camjit"):
        (lo_m, hi_m), (lo_deg, hi_deg) = ((0.015, 0.025), (2.0, 4.0)) if variant == "camera" else ((0.0, 0.015), (0.0, 2.0))
        for cam in C.CAMS:
            i = model.camera(cam).id
            d = rng.normal(size=3)
            model.cam_pos[i] += d / np.linalg.norm(d) * rng.uniform(lo_m, hi_m)
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            ang = np.radians(rng.uniform(lo_deg, hi_deg))
            dq = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * ax)])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, model.cam_quat[i], dq)
            model.cam_quat[i] = q
    elif variant == "dr":
        # Domain randomisation for TRAINING demos: random lighting, colours and a small camera move, re-drawn for
        # every image. Kept apart from the held-out test shifts: those are darker with a strong warm cast (light),
        # repaint the block green / peg yellow (color), and move the cameras 1.5-2.5 cm / 2-4 deg (camera).
        for i in range(model.nlight):
            model.light_diffuse[i] *= rng.uniform(0.55, 1.4) * rng.uniform(0.85, 1.1, 3)
            model.light_ambient[i] *= rng.uniform(0.6, 1.4)
            model.light_pos[i, :2] += rng.uniform(-0.3, 0.3, 2)
        model.vis.headlight.diffuse[:] = np.clip(model.vis.headlight.diffuse * rng.uniform(0.6, 1.3), 0, 1)
        for m in range(model.nmat):
            model.mat_rgba[m, :3] = np.clip(model.mat_rgba[m, :3] * rng.uniform(0.75, 1.15, 3), 0, 1)
        for g in range(model.ngeom):
            if model.geom_matid[g] < 0 and model.geom_rgba[g, 3] > 0:   # plain-coloured geoms (task objects)
                model.geom_rgba[g, :3] = np.clip(model.geom_rgba[g, :3] + rng.uniform(-0.15, 0.15, 3), 0, 1)
        apply_variant(model, "camjit", rng)
    elif variant != "clean":
        raise ValueError(variant)


GEOMETRIC = {"camera", "camjit", "dr"}   # variants that change depth too
PER_IMAGE = {"camjit", "dr"}             # variants re-drawn for every image (others: one draw per episode)


def labels(model, data, jaws, block_id, peg_id):
    import humanoid_mujoco_scenes.peg_insert.scene as S
    tcp = data.geom_xpos[jaws].mean(0)
    peg = data.xpos[peg_id].copy()
    hole = data.xpos[block_id].copy()
    tilt = float(np.degrees(np.arccos(np.clip(data.xmat[peg_id][8], -1, 1))))
    lifted = peg[2] - S.PEG_HEIGHT / 2 - S.TABLE_TOP_Z
    inserted = S.is_inserted(model, data)
    near = np.linalg.norm(peg - tcp) < 0.03
    over_hole = np.linalg.norm(peg[:2] - hole[:2]) < 0.015
    if inserted:
        phase = 4
    elif lifted > 0.01 and over_hole:
        phase = 3
    elif lifted > 0.01:
        phase = 2
    elif near:
        phase = 1
    else:
        phase = 0
    return dict(tcp=tcp, peg=peg, hole=hole, tilt=tilt, phase=phase)


def work(job):
    seed, variants = job
    import mujoco
    from synthetic_teleop import ROBOTS, TeleopSim, run_episode
    frames = []
    sub = []   # the operator's subtask index per frame (task progress; the phase label for non-peg scenes)
    ok, sim_t, notes, _ = run_episode(C.SCENE, seed, frame_sink=lambda a, s, t, e, q: (frames.append((a, s, q)),
                                      sub.append(int(e["subtask_index"][0]) if e else 0)),
                                      record_every=C.RECORD_EVERY, robot=C.ROBOT)
    if not ok:
        return seed, None
    sim = TeleopSim(C.SCENE, seed, cameras={c: (C.H, C.W) for c in C.CAMS}, robot=C.ROBOT)
    model, data = sim.model, sim.data
    if C.BIMANUAL:
        # The operator drives the left arm (its recorded action / state); the right arm holds its home targets the
        # whole take, so its action is those targets and its state is read from the recorded qpos. One 16-D vector
        # per frame in the 8dof schema's order (left 7 + gripper, right 7 + gripper).
        io = C.ArmIO(model)
        n_left = len(io.arms[0]["acts"]) + 1
        right_act = io.action_from_ctrl(data.ctrl)[n_left:]
        act = np.stack([np.concatenate([f[0], right_act]) for f in frames])
        st = np.stack([io.state(f[2]) for f in frames])
        assert np.allclose(st[:, :n_left], np.stack([f[1] for f in frames]), atol=1e-5), "left state != recorded state"
        drift = np.abs(st[:, n_left:-1] - right_act[:-1]).max()
        assert drift < 0.05, f"seed {seed}: right arm drifted {drift:.3f} rad from its held targets"
        out = {"action": act.astype(np.float32), "state": st}
    else:
        out = {"action": np.stack([f[0] for f in frames]), "state": np.stack([f[1] for f in frames])}
    fingers = ROBOTS[C.ROBOT]["fingers"]
    jaws = [next(g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(b).id and model.geom_contype[g])
            for b in fingers]
    peg_task = C.SCENE == "peg_insert"
    block_id, peg_id = (model.body("hole_block").id, model.body("peg").id) if peg_task else (None, None)
    r = mujoco.Renderer(model, C.H, C.W)
    idx = np.arange(0, len(frames), IMG_EVERY)
    out["img_idx"] = idx
    lab = []
    base_light = (model.light_diffuse.copy(), model.light_ambient.copy(), model.light_pos.copy(),
                  model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy(),
                  model.geom_rgba.copy(), model.mat_rgba.copy(), model.cam_pos.copy(), model.cam_quat.copy())
    for v in variants:
        (model.light_diffuse[:], model.light_ambient[:], model.light_pos[:], model.vis.headlight.ambient[:],
         model.vis.headlight.diffuse[:], model.geom_rgba[:], model.mat_rgba[:], model.cam_pos[:], model.cam_quat[:]) = base_light
        apply_variant(model, v, np.random.default_rng([seed, 7]))
        rgbs = {c: [] for c in C.CAMS}
        deps = {c: [] for c in C.CAMS}
        for k in idx:
            if v in PER_IMAGE:   # back to the clean scene, then a fresh draw for this image
                (model.light_diffuse[:], model.light_ambient[:], model.light_pos[:], model.vis.headlight.ambient[:],
                 model.vis.headlight.diffuse[:], model.geom_rgba[:], model.mat_rgba[:], model.cam_pos[:],
                 model.cam_quat[:]) = base_light
                apply_variant(model, v, np.random.default_rng([seed, 11, int(k)]))
            data.qpos[:] = frames[k][2]
            mujoco.mj_forward(model, data)
            if v == variants[0]:
                lab.append(labels(model, data, jaws, block_id, peg_id) if peg_task else
                           dict(tcp=data.geom_xpos[jaws].mean(0), phase=sub[k]))
            for c in C.CAMS:
                r.update_scene(data, c)
                rgbs[c].append(r.render().copy())
                if v == "clean" or v == variants[0] or v in GEOMETRIC:
                    r.enable_depth_rendering()
                    r.update_scene(data, c)
                    deps[c].append(np.clip(r.render() * 1000, 0, 65535).astype(np.uint16))
                    r.disable_depth_rendering()
        for c in C.CAMS:
            out[f"{v}_{c}_rgb"] = np.stack(rgbs[c])
            if deps[c]:
                out[f"{v}_{c}_depth" if v in GEOMETRIC else f"{c}_depth"] = np.stack(deps[c])
                deps[c] = []
    for key in ("tcp", "peg", "hole") if peg_task else ("tcp",):
        out[key] = np.stack([l[key] for l in lab]).astype(np.float32)
    if peg_task:
        out["tilt"] = np.array([l["tilt"] for l in lab], np.float32)
    out["phase"] = np.array([l["phase"] for l in lab], np.int64)
    r.close()
    sim.close()
    return seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seeds", required=True, help="a:b")
    ap.add_argument("--variants", default="clean")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    C.contract_check()
    a, b = map(int, args.seeds.split(":"))
    variants = args.variants.split(",")
    outdir = C.DATA / args.split
    outdir.mkdir(exist_ok=True)
    t0, n_ok = time.time(), 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for seed, out in pool.imap_unordered(work, [(s, variants) for s in range(a, b)]):
            if out is None:
                print(f"seed {seed}: operator failed, skipped", flush=True)
                continue
            np.savez_compressed(outdir / f"ep_{seed:05d}.npz", **out)
            n_ok += 1
            print(f"seed {seed}: {len(out['action'])} steps, {len(out['img_idx'])} images  [{time.time()-t0:.0f}s]", flush=True)
    print(f"done: {n_ok} episodes in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
