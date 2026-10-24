"""Closed-loop, unassisted evaluation of a trained policy (ACT or flow transformer) in the peg-insert scene.

The policy sees exactly its training inputs (ego + wrist RGB, seven-value measured state) and its
absolute actuator targets are applied at 25 Hz. Privileged poses are used only for reset and
metrics. Success = peg seated (``PegExpert.inserted``) and free of both fingers for 0.5 s.

  # held-out workspace layouts, 50-action chunks
  MUJOCO_GL=egl python -m humanoid_mujoco_scenes.peg_insert.eval --checkpoint <ckpt> --device cuda \\
      --layouts outputs/peg_workspace/heldout_layouts.json --n-action-steps 50 --report outputs/eval/run/heldout.json
  # watch one rollout on macOS
  mjpython -m humanoid_mujoco_scenes.peg_insert.eval --checkpoint <ckpt> --device mps --viewer --episodes 1 \\
      --report outputs/eval/run/watch.json
  # insertion-only diagnostic: the expert hands over a held peg offset 3 mm from the hole
  ... --start-stage insertion --insertion-case offset_positive

The report keeps per-episode stage flags (grasped/lifted/inserted), first-event times, clipped
actions and a summary per layout group/band. ``--trace-dir`` saves full qpos/qvel/ctrl per step,
enough to restore any frame for corrections (``collect_corrections --source traces``).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from contextlib import ExitStack, nullcontext
from pathlib import Path

import mujoco
import numpy as np

from humanoid_mujoco_scenes import make_model, scene_camera
from .expert import DT, FPS, ExpertFailure, PegExpert
from .handoffs import INSERTION_CASES, prepare_insertion
from .recording import snapshot, write_json

SUCCESS_HOLD = 0.5  # s seated and released


def render(renderers, data):
    images = {}
    for name, renderer in renderers.items():
        renderer.update_scene(data, camera=name)
        images[name] = renderer.render().copy()
    return images


class EpisodeTracker:
    """Stage flags, first-event times and optional per-step trace for one rollout."""

    def __init__(self, expert, seed, keep_trace):
        self.e, self.keep_trace, self.trace, self.stable = expert, keep_trace, [], 0
        self.result = {"seed": seed, "success": False, "grasped": False, "lifted": False, "inserted": False,
                       "control_steps": 0, "clipped_actions": 0, "clipped_components": [0] * 7,
                       "initial_state": expert.state().tolist()}

    def after_step(self, action, bounded, clipped):
        r, e = self.result, self.e
        if r["control_steps"] == 0:
            r["first_action"] = np.asarray(action).tolist()
        r["control_steps"] += 1
        r["clipped_actions"] += int(clipped)
        for i, changed in enumerate(np.abs(np.asarray(action) - bounded) > 1e-6):
            r["clipped_components"][i] += int(changed)
        held, inserted, released = e.contacts(), e.inserted(), e.released()
        seconds = r["control_steps"] * DT
        for event, value in [("grasped", held), ("lifted", e.lifted()), ("inserted", inserted)]:
            if value and not r[event]:
                r[event], r[f"first_{event}_seconds"] = True, seconds
        if self.keep_trace:
            self.trace.append({"seconds": seconds, "peg_position": e.peg_pos, "hole_position": e.hole_pos,
                               "peg_rotation": e.peg_rotation().copy(), "held": held, "block_contact": e.block_contacts(),
                               "inserted": inserted, "released": released, **snapshot(e)})
        self.stable = self.stable + 1 if inserted and released else 0
        r["success"] = self.stable >= round(SUCCESS_HOLD * FPS)
        return r["success"]

    def finish(self, wall_start):
        r, e = self.result, self.e
        r.setdefault("failure", None if r["success"] else "time_limit")
        r.update(sim_seconds=r["control_steps"] * DT, wall_seconds=round(time.monotonic() - wall_start, 3),
                 final_peg_position=e.peg_pos.tolist(), final_hole_position=e.hole_pos.tolist(),
                 final_peg_rotation=e.peg_rotation().tolist(), final_released=e.released(), final_state=e.state().tolist())
        return r


def rollout(policy, expert, renderers, max_steps, tracker, stop=None, frame_hook=None):
    """Run the policy until success, ``stop(expert, lifted, seconds)`` returns a reason, or time runs out."""
    for _ in range(max_steps):
        action = policy.select_action(expert.state(), render(renderers, expert.data))
        bounded, clipped = expert.apply(action)
        expert.step()
        if tracker.after_step(action, bounded, clipped):
            return None
        if frame_hook:
            frame_hook()
        reason = stop(expert, tracker.result["lifted"], tracker.result["control_steps"] * DT) if stop else None
        if reason:
            return reason
    return None


class Video:
    """Side-by-side overview/ego/wrist MP4 at simulation time (independent of inference speed)."""

    def __init__(self, model, renderers, views, stack):
        self.renderers, self.views = renderers, views
        self.overview = stack.enter_context(mujoco.Renderer(model, 480, 640)) if "overview" in views else None
        self.camera = mujoco.MjvCamera()
        for key, value in scene_camera("peg_insert").items():
            setattr(self.camera, key, value)
        self.writer = None

    def open(self, path):
        import imageio.v2 as imageio
        self.writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=7)

    def append(self, data, seed):
        from PIL import Image, ImageDraw
        if self.writer is None:
            return
        panels = []
        for view in self.views:
            renderer = self.overview if view == "overview" else self.renderers[view]
            renderer.update_scene(data, camera=self.camera if view == "overview" else view)
            panel = Image.fromarray(renderer.render().copy()).resize((640, 480))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 640, 26), fill="black")
            draw.text((8, 6), f"{view} | seed {seed} | {data.time:.2f}s", fill="white")
            panels.append(np.asarray(panel))
        self.writer.append_data(np.concatenate(panels, axis=1))

    def close(self):
        if self.writer:
            self.writer.close()
        self.writer = None


def summarize(results):
    """Successes and stage counts per layout group/band (``nominal`` without explicit layouts)."""
    groups = {}
    for row in results:
        layout = row.get("layout", {})
        group = groups.setdefault(f"{layout.get('group', 'nominal')}_{layout.get('band', 'near')}", Counter())
        group["episodes"] += 1
        for key in ["success", "grasped", "lifted", "inserted"]:
            group["successes" if key == "success" else key] += int(row[key])
    return {k: dict(v) for k, v in groups.items()}


def episodes(args):
    if args.layouts:
        layouts = json.loads(args.layouts.read_text())["layouts"]
        return [(layout["seed"], layout) for layout in layouts]
    return [(args.seed + i, None) for i in range(args.episodes)]


def evaluate_episode(args, policy, expert, renderers, video, viewer, seed, layout):
    with viewer.lock() if viewer else nullcontext():
        expert.reset(seed, args.randomize, *((layout["peg_xy"], layout["hole_xy"]) if layout else ()))
    setup = prepare_insertion(expert, args.insertion_case) if args.start_stage == "insertion" else None
    policy.reset(seed)
    tracker = EpisodeTracker(expert, seed, args.trace_dir is not None)
    if layout:
        tracker.result["layout"] = layout
    if setup:
        tracker.result.update(insertion_setup=setup, grasped=expert.contacts(), lifted=expert.lifted())
    wall = time.monotonic()
    record = args.video_dir and (args.video_seeds is None or seed in args.video_seeds)
    if record:
        video.open(args.video_dir / f"seed_{seed:06d}.mp4")

    last_frame = [time.monotonic()]

    def frame_hook():
        video.append(expert.data, seed)
        if viewer:
            viewer.sync()
            if not viewer.is_running():
                raise ExpertFailure("viewer_closed")
            time.sleep(max(0.0, DT - (time.monotonic() - last_frame[0])))  # real-time playback
            last_frame[0] = time.monotonic()
    try:
        rollout(policy, expert, renderers, round(args.max_seconds * FPS), tracker, frame_hook=frame_hook)
    except ExpertFailure as exc:
        tracker.result["failure"] = str(exc)
    finally:
        video.close()
    if tracker.trace:
        np.savez_compressed(args.trace_dir / f"seed_{seed:06d}.npz", **{k: np.array([f[k] for f in tracker.trace]) for k in tracker.trace[0]})
    return tracker.finish(wall)


def run(args):
    from humanoid_robot_learning.policy_adapters import load_policy

    policy = load_policy(args.checkpoint, args.device, args.n_action_steps, **policy_options(args))
    model = make_model("peg_insert", cameras=policy.cameras)
    expert = PegExpert(model, mujoco.MjData(model))
    report = {**policy.describe(), "device": args.device, "fps": FPS, "start_stage": args.start_stage,
              "insertion_case": args.insertion_case, "randomize": args.randomize, "max_seconds": args.max_seconds,
              "layouts": str(args.layouts) if args.layouts else None, "mujoco_version": mujoco.__version__, "results": []}
    for directory in [args.trace_dir, args.video_dir, args.report.parent]:
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        renderers = {n: stack.enter_context(mujoco.Renderer(model, h, w)) for n, (h, w) in policy.cameras.items()}
        video = Video(model, renderers, args.video_views, stack)
        viewer = open_viewer(model, expert.data, stack) if args.viewer else None
        for seed, layout in episodes(args):
            if viewer and not viewer.is_running():
                break
            report["results"].append(evaluate_episode(args, policy, expert, renderers, video, viewer, seed, layout))
            report.update(success_rate=float(np.mean([r["success"] for r in report["results"]])), summary=summarize(report["results"]))
            write_json(args.report, report)
            print(json.dumps({k: report["results"][-1][k] for k in ["seed", "success", "grasped", "lifted", "inserted", "failure"]}), flush=True)
    print(f"Success: {sum(r['success'] for r in report['results'])}/{len(report['results'])}; report: {args.report}")


def open_viewer(model, data, stack):
    from mujoco import viewer as mujoco_viewer
    viewer = stack.enter_context(mujoco_viewer.launch_passive(model, data))
    for key, value in scene_camera("peg_insert").items():
        setattr(viewer.cam, key, value)
    return viewer


def policy_options(args):
    if args.refinement_steps is not None:
        return {"refinement_steps": args.refinement_steps}
    return {"temporal_ensemble_coeff": args.temporal_ensemble_coeff, "no_temporal_ensemble": args.no_temporal_ensemble} \
        if args.temporal_ensemble_coeff is not None or args.no_temporal_ensemble else {}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="ACT pretrained_model directory or flow <step>.pt")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--report", type=Path, required=True, help="new JSON report path")
    p.add_argument("--layouts", type=Path, help="explicit layouts JSON (workspace sample); replaces --episodes/--seed")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000, help="first layout seed")
    p.add_argument("--randomize", type=float, default=0.01, help="reset jitter (m) for seeded layouts")
    p.add_argument("--max-seconds", type=float, default=60, help="simulated rollout limit")
    p.add_argument("--n-action-steps", type=int, help="actions executed per predicted chunk")
    ensemble = p.add_mutually_exclusive_group()
    ensemble.add_argument("--temporal-ensemble-coeff", type=float, help="ACT temporal ensembling (queries every step)")
    ensemble.add_argument("--no-temporal-ensemble", action="store_true", help="disable ensembling saved in the ACT config")
    p.add_argument("--refinement-steps", type=int, help="flow: Euler steps per chunk (default: the checkpoint's)")
    p.add_argument("--start-stage", choices=("full", "insertion"), default="full")
    p.add_argument("--insertion-case", choices=INSERTION_CASES, default="aligned")
    p.add_argument("--trace-dir", type=Path, help="per-step physical traces (restorable states)")
    p.add_argument("--video-dir", type=Path, help="MP4 per episode")
    p.add_argument("--video-views", nargs="+", choices=("overview", "ego", "wrist_left"), default=["overview"])
    p.add_argument("--video-seeds", type=int, nargs="+", help="record only these seeds")
    p.add_argument("--viewer", action="store_true", help="interactive viewer (mjpython on macOS)")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.report.exists():
        p.error("--report exists; use a new path")
    if args.episodes < 1 or args.max_seconds <= 0 or not 0 <= args.randomize <= 0.02:
        p.error("episodes/max-seconds must be positive and randomize in [0, 0.02]")
    if args.layouts and args.start_stage != "full":
        p.error("explicit layouts are full-task evaluations")
    run(args)


if __name__ == "__main__":
    main()
