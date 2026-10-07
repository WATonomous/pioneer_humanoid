"""Synthetic leader-arm teleop demos in MuJoCo: a simulated person (human_operator.py) does the task.

    python synthetic_teleop.py --scene drawer_stow --record --num_episodes 10
    python synthetic_teleop.py --scene peg_insert --num_episodes 3 --preview peg.mp4   # look first, no dataset

Same pipeline as ``pioneer_leader_arm_teleop.py --target mujoco --record``: the operator's leader angles
go through the same LeaderMapping (engage at home, low-pass filter, URDF clamp), and frames are recorded
with the same features (state, action, cameras, per-step task and subtask_index), so a synthetic
episode is indistinguishable in format from a real one. Headless; no leader hardware.

Each episode is seeded (``--seed`` + index): the scene layout and the operator (speed, aim, tremor, ...)
are drawn from it. Episodes are first run without cameras on ``--workers`` processes; only ones that
finish the task are re-run (deterministically) with cameras and saved, like a person discarding a
botched take. Task plans: synthetic_tasks.py. Rendering is headless: MUJOCO_GL=egl (default; GPU) or
MUJOCO_GL=osmesa (CPU only, slower).
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))
sys.path.insert(0, str(_SRC / "simulation" / "mujoco_scenes"))
sys.path.insert(0, str(_SRC / "robot_learning"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from humanoid_robot_learning.sim_teleop_record import add_record_args, load_record_schema, make_sim_recorder  # noqa: E402

from arm_limits import GRIPPER_OPEN_DEG  # noqa: E402
from leader_mapping import CONTROL_DT, DEFAULT_SIGNS, WRIST_DAMPING, LeaderMapping  # noqa: E402
from servo_leader import parse_signs  # noqa: E402


class TeleopSim:
    """mujoco_sim.py's control/record loop, driven by leader angles instead of the serial leader."""

    def __init__(self, scene: str, seed: int, cameras: dict[str, tuple[int, int]] | None = None, filter_alpha: float = 0.35):
        """``cameras``: {name: (height, width)} to render with ``render_at`` (the recording's render workers)."""
        import mujoco
        import numpy as np
        from humanoid_mujoco_scenes import make_model, scene_progress, scene_reset, scene_step
        from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, LEFT_GRIPPER_CLOSED, LEFT_GRIPPER_JOINTS, LEFT_GRIPPER_OPEN
        from pioneer_humanoid.mujoco_bimanual_arm import set_home

        self.mujoco, self.np = mujoco, np
        self.scene = scene
        self.model = model = make_model(scene, cameras=cameras)
        self.data = data = mujoco.MjData(model)
        model.actuator_biasprm[model.actuator("joint6l").id, 2] = -WRIST_DAMPING
        self.hook, self.progress = scene_step(scene), scene_progress(scene)
        self.joints = LEFT_ARM_JOINTS
        self.arm_acts = [model.actuator(j).id for j in LEFT_ARM_JOINTS]
        self.grip_acts = [model.actuator(j).id for j in LEFT_GRIPPER_JOINTS]
        self.grip_open = [LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS]
        self.grip_closed = [LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS]
        self.arm_qpos = [model.joint(j).qposadr[0] for j in LEFT_ARM_JOINTS]
        self.grip_qpos = [model.joint(j).qposadr[0] for j in LEFT_GRIPPER_JOINTS]
        set_home(model, data)
        randomise = scene_reset(scene)
        if randomise is not None:
            randomise(model, data, np.random.default_rng([seed, 0]))
        mujoco.mj_forward(model, data)
        self.home = [data.ctrl[a] for a in self.arm_acts]
        self.signs = parse_signs(DEFAULT_SIGNS)
        self.mapping = LeaderMapping(
            argparse.Namespace(signs=DEFAULT_SIGNS, filter_alpha=filter_alpha),
            home_rad=self.home,
            limits_rad=[tuple(model.jnt_range[model.joint(j).id]) for j in LEFT_ARM_JOINTS],
        )
        self.substeps = max(1, round(CONTROL_DT / model.opt.timestep))
        self.control_dt = self.substeps * model.opt.timestep
        # One renderer per image size, shared by the cameras of that size (each one is a whole GL context).
        self.cameras = dict(cameras or {})
        self.renderers = {size: mujoco.Renderer(model, *size) for size in set(self.cameras.values())}
        self.record_every, self.steps = 0, 0
        self.frame_sink = None        # frame_sink(action, state, task, extras, qpos) on every recorded frame
        self.on_frame = None          # optional callback(sim) on every recorded/preview frame

    def leader_angles(self, q_arm, grip: float) -> tuple[float, ...]:
        """Servo readings that LeaderMapping turns into exactly ``q_arm`` (rad) and ``grip`` (0..1)."""
        import math
        arm = [q / s for q, s in zip(q_arm, self.signs)]
        return (*arm, -math.radians(grip * GRIPPER_OPEN_DEG) / self.signs[len(arm)])

    def tick(self, q_leader, grip: float) -> None:
        """One control step: leader -> mapping -> follower targets, record, scene hook, physics."""
        np, model, data = self.np, self.model, self.data
        target, g = self.mapping.update(self.leader_angles(q_leader, grip))
        data.ctrl[self.arm_acts] = target
        data.ctrl[self.grip_acts] = [o + g * (c - o) for o, c in zip(self.grip_open, self.grip_closed)]
        if self.mapping.engaged and self.record_every and self.steps % self.record_every == 0:
            if self.frame_sink is not None:
                # Same features as mujoco_sim.py: 6 joints + mean finger closure / 6 targets + leader grip.
                closure = sum((data.qpos[q] - o) / (c - o) for q, o, c in zip(self.grip_qpos, self.grip_open, self.grip_closed)) / len(self.grip_qpos)
                state = np.append(data.qpos[self.arm_qpos], min(max(closure, 0.0), 1.0)).astype(np.float32)
                action = np.append(target, g).astype(np.float32)
                task, extras = None, None
                if self.progress is not None:
                    index, _, task = self.progress(model, data)
                    extras = {"subtask_index": np.array([index], dtype=np.float32)}
                self.frame_sink(action, state, task, extras, data.qpos.copy())
            if self.on_frame is not None:
                self.on_frame(self)
        if self.hook is not None:
            self.hook(model, data)
        self.mujoco.mj_step(model, data, nstep=self.substeps)
        self.steps += 1

    def render_at(self, qpos):
        """Camera images of the scene posed at ``qpos`` (rendering needs only the poses)."""
        self.data.qpos[:] = qpos
        self.mujoco.mj_forward(self.model, self.data)
        images = {}
        for name, size in self.cameras.items():
            renderer = self.renderers[size]
            renderer.update_scene(self.data, name)
            images[name] = renderer.render()
        return images

    def close(self) -> None:
        for r in self.renderers.values():
            r.close()


_WORKER_SIMS: dict = {}


def _render_job(job):
    """Render pool worker: images of episode (scene, seed) at one recorded frame's qpos."""
    scene, seed, cameras, qpos = job
    key = (scene, seed)
    if key not in _WORKER_SIMS:
        for old in _WORKER_SIMS.values():
            old.close()
        _WORKER_SIMS.clear()
        _WORKER_SIMS[key] = TeleopSim(scene, seed, cameras)   # same seed: same layout (e.g. peg_insert's block)
    return _WORKER_SIMS[key].render_at(qpos)


class _Recording:
    """Feeds one take's frames into the recorder in order, rendering the cameras on a process pool while the
    simulation runs (CPU rendering is the slow part; physics and the operator stay on this process)."""

    def __init__(self, recorder, pool, scene: str, seed: int, cameras):
        self.recorder, self.pool, self.job = recorder, pool, (scene, seed, cameras)
        self.pending = []

    def __call__(self, action, state, task, extras, qpos) -> None:
        self.pending.append((action, state, task, extras, self.pool.submit(_render_job, (*self.job, qpos))))
        self.drain(block=len(self.pending) > 64)   # bounded queue: images wait in the recorder, not here

    def drain(self, block: bool = False) -> None:
        while self.pending and (block or self.pending[0][4].done()):
            action, state, task, extras, fut = self.pending.pop(0)
            self.recorder.push_frame_to_buffer(action, state, fut.result(), extras=extras, task=task)
            block = block and len(self.pending) > 32


def run_episode(scene: str, seed: int, frame_sink=None, record_every: int = 0, on_frame=None):
    """Play one synthetic take; returns (success, sim seconds, notes, the operator's style as a dict)."""
    import dataclasses

    import numpy as np
    from human_operator import HumanOperator
    from synthetic_tasks import TASKS, TaskFailed

    sim = TeleopSim(scene, seed)
    sim.frame_sink, sim.record_every, sim.on_frame = frame_sink, record_every, on_frame
    op = HumanOperator(sim.model, lambda: sim.data, sim.tick, sim.joints, sim.home, sim.control_dt,
                       np.random.default_rng([seed, 1]))
    task = TASKS[scene]
    notes = []
    try:
        task.plan(op, sim.model, sim.data, notes)
    except TaskFailed as exc:
        notes.append(f"gave up: {exc}")
    ok = bool(task.success(sim.model, sim.data))
    if os.environ.get("SYNTH_TRACE"):
        print("\n".join(op.trace))
    sim.close()
    return ok, sim.data.time, notes, {k: round(float(v), 5) for k, v in dataclasses.asdict(op.style).items()}


def _probe(job):
    scene, seed = job
    os.environ.setdefault("MUJOCO_GL", "egl")
    t0 = time.time()
    ok, sim_t, notes, _ = run_episode(scene, seed)
    return seed, ok, sim_t, time.time() - t0, notes


def self_test() -> None:
    """Check the virtual leader without recording: angle math, motion model, a short reach."""
    import numpy as np
    from human_operator import HumanOperator, _Channel, minimum_jerk

    sim = TeleopSim("bare", 0, filter_alpha=1.0)   # no low-pass: targets must equal the leader exactly
    sim.mapping.update(sim.leader_angles(sim.home, 0.0))
    assert sim.mapping.engaged
    q = np.array(sim.home) + 0.1
    target, grip = sim.mapping.update(sim.leader_angles(q, 0.7))
    assert np.allclose(target, q) and abs(grip - 0.7) < 1e-9, (target, grip)

    assert minimum_jerk(0.0) == 0.0 and minimum_jerk(1.0) == 1.0 and abs(minimum_jerk(0.5) - 0.5) < 1e-12
    ch = _Channel([0.0, 0.0])
    ch.add(0.0, 1.0, [1.0, 0.0])
    ch.add(0.5, 1.0, [0.0, 2.0])
    assert np.allclose(ch.goal(), [1.0, 2.0]) and np.allclose(ch.at(2.0), [1.0, 2.0])

    sim = TeleopSim("bare", 0)
    op = HumanOperator(sim.model, lambda: sim.data, sim.tick, sim.joints, sim.home, sim.control_dt, np.random.default_rng(0))
    goal = op.follower_tcp() + (-0.05, 0.05, -0.05)
    err = op.move(goal, tol=0.005)
    assert err < 0.01, f"reach missed by {err * 1000:.1f} mm"
    print(f"self-test ok (reach error {err * 1000:.1f} mm in {sim.data.time:.1f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic human-like leader-arm teleop demos (MuJoCo, headless).")
    parser.add_argument("--scene", default=None, help="scene with a plan in synthetic_tasks.py")
    parser.add_argument("--seed", type=int, default=0, help="first episode seed (episodes use seed, seed+1, ...)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--max_tries", type=int, default=0, help="seeds to try at most (default: 3 x num_episodes)")
    parser.add_argument("--preview", type=str, default=None, help="also write an mp4 of the kept takes (scene camera)")
    parser.add_argument("--self-test", action="store_true", help="check the leader math and motion model and exit")
    add_record_args(parser, task_description="")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    record = load_record_schema(parser, args)
    os.environ.setdefault("MUJOCO_GL", "egl")

    from synthetic_tasks import TASKS
    if args.scene is None or args.scene not in TASKS:
        raise SystemExit(f"no synthetic plan for --scene {args.scene!r}; have: {sorted(TASKS)}")
    args.task_description = args.task_description or TASKS[args.scene].instruction
    if args.record and args.dataset_root is None:   # never mixed into the human recordings under <root>/sim
        from humanoid_robot_learning.record_utils import resolve_dataset_root
        args.dataset_root = str(resolve_dataset_root(record.cfg, subdir=f"sim_synthetic/{args.scene}"))

    # Takes already in the dataset (an interrupted run picks up where it stopped).
    done = _recorded_takes(args.dataset_root) if args.record else []
    done_seeds = {t["seed"] for t in done}
    if done:
        print(f"[SYNTH] {len(done)} takes already in {args.dataset_root}: seeds {sorted(done_seeds)}")

    # 1. Find seeds whose take succeeds (no cameras: fast, parallel).
    want, tries = args.num_episodes - len(done), args.max_tries or 3 * args.num_episodes
    if want <= 0:
        return
    kept, takes = [], {}
    jobs = [(args.scene, args.seed + i) for i in range(tries) if args.seed + i not in done_seeds]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for seed, ok, sim_t, wall, notes in pool.imap(_probe, jobs):
            print(f"[SYNTH] seed {seed}: {'ok  ' if ok else 'FAIL'} {sim_t:5.1f} s take ({wall:.0f} s wall)"
                  + (f"  [{'; '.join(notes)}]" if notes else ""), flush=True)
            if ok:
                kept.append(seed)
                takes[seed] = sim_t
            if len(kept) >= want:
                pool.terminate()
                break
    print(f"[SYNTH] {len(kept)}/{want} successful takes: seeds {kept}", flush=True)

    # 2. Re-run the kept seeds with cameras into the dataset (and/or the preview video).
    if not (args.record or args.preview) or not kept:
        return
    preview = _Preview(args.preview, args.scene) if args.preview else None
    try:
        for seed in kept:
            t0 = time.time()
            if args.record:
                # Each take in a fresh process (it reopens the dataset and appends): a take's frame buffers, render
                # workers and video encode are all freed when it exits, instead of fragmenting this process's heap.
                ctx_spawn = multiprocessing.get_context("spawn")
                result = ctx_spawn.Queue()
                child = ctx_spawn.Process(target=_record_take, args=(args, record, seed, takes[seed], result))
                child.start()
                child.join()
                try:
                    ok, sim_t, _ = result.get(timeout=5)
                except Exception:   # the child died before reporting (e.g. out of memory)
                    ok, sim_t = False, 0.0
                print(f"[SYNTH] seed {seed}: {'saved' if ok else 'NOT SAVED'} ({sim_t:.1f} s, {time.time() - t0:.0f} s wall)", flush=True)
                if child.exitcode != 0:
                    raise SystemExit(f"recording seed {seed} failed (exit code {child.exitcode}); rerun to resume")
            if preview is not None:
                run_episode(args.scene, seed, record_every=4, on_frame=preview.frame)
    finally:
        if preview is not None:
            preview.close()
            print(f"[SYNTH] preview: {args.preview}")
    if args.record:
        print(f"[RECORD] Saved under {args.dataset_root}")


TAKES_FILE = "meta/synthetic_takes.jsonl"   # one line per saved episode: seed, operator style, what happened


def _recorded_takes(dataset_root) -> list[dict]:
    import json
    path = Path(dataset_root) / TAKES_FILE
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _record_take(args, record, seed: int, take_s: float, result) -> None:
    """Child process: replay one (known good) take with cameras, append it to the dataset, note it in TAKES_FILE."""
    import json
    from concurrent.futures import ProcessPoolExecutor

    cameras = {n: (int(s["height"]), int(s["width"])) for n, s in record.images.items()}
    probe = TeleopSim(args.scene, seed, None)
    # Frame buffers sized to this take (not the default 120 s): ~0.9 GB per 10 s with two 640x480 cameras.
    recorder, record_every = make_sim_recorder(args, record, device="cpu", sim_dt=probe.control_dt,
                                               extra_features={"subtask_index": ["subtask_index"]} if probe.progress else None,
                                               buffer_capacity_s=take_s + 5.0)
    probe.close()
    episode = recorder.dataset.meta.total_episodes
    # CPU rendering is the slow part: 2 spawned render workers (~1.3 GB each with osmesa).
    pool = ProcessPoolExecutor(max_workers=min(args.workers, 2), mp_context=multiprocessing.get_context("spawn"))
    try:
        rec = _Recording(recorder, pool, args.scene, seed, cameras)
        ok, sim_t, notes, style = run_episode(args.scene, seed, rec, record_every)
        rec.drain(block=True)
    finally:
        pool.shutdown(cancel_futures=True)
    if ok:
        recorder.save_episode()
        recorder.wait_saved()
        with open(Path(recorder.dataset_root) / TAKES_FILE, "a") as f:
            f.write(json.dumps({"episode_index": episode, "scene": args.scene, "seed": seed, "take_s": round(sim_t, 2),
                                "notes": notes, "operator": style}) + "\n")
    else:   # physics is deterministic, so this would be a bug; never save a failed take
        recorder.cancel_recording()
    recorder.finalize()
    result.put((ok, sim_t, notes))


class _Preview:
    """Scene-camera video of the takes (25 fps), with the current instruction burnt in."""

    def __init__(self, path: str, scene: str, size=(480, 640)):
        import subprocess

        import mujoco
        from humanoid_mujoco_scenes import scene_camera

        self.size, self.cam_cfg, self.renderer, self.r_model = size, scene_camera(scene) or {}, None, None
        h, w = size
        self.proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                      "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path],
                                     stdin=subprocess.PIPE)
        self.mujoco = mujoco

    def frame(self, sim: TeleopSim) -> None:
        import numpy as np
        from PIL import Image, ImageDraw
        if self.r_model is not sim.model:
            if self.renderer is not None:
                self.renderer.close()
            self.renderer, self.r_model = self.mujoco.Renderer(sim.model, *self.size), sim.model
        cam = self.mujoco.MjvCamera()
        for k, v in self.cam_cfg.items():
            setattr(cam, k, v) if k != "lookat" else cam.lookat.__setitem__(slice(None), v)
        self.renderer.update_scene(sim.data, cam)
        img = Image.fromarray(self.renderer.render())
        if sim.progress is not None:
            k, n, txt = sim.progress(sim.model, sim.data)
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, self.size[1], 22], fill=(0, 0, 0))
            draw.text((8, 5), f"step {min(k + 1, n)}/{n}: {txt}" if k < n else "all steps done", fill=(255, 255, 255))
        self.proc.stdin.write(np.asarray(img).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()
        if self.renderer is not None:
            self.renderer.close()


if __name__ == "__main__":
    main()
