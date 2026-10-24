"""Collect expert corrections (DAgger-style labels) from held-peg states, for fine-tuning a policy.

Only expert frames from the handoff to verified insertion + release are recorded; the policy's
own actions are never labels. Every handoff is physical: a policy rollout, an expert setup, or a
restored eval-trace state. A failed correction is logged in ``manifest.json`` and discarded.

Sources:
  takeover  roll out ``--checkpoint``; hand off when ``--trigger`` fires
            (approach: within 3 cm of the hole | stall: stalled/tilted/misaligned at the rim | rim-jam)
  tasks     expert-prepared held starts: ``--tasks broad`` (80 offsets/heights/tilts around the hole),
            ``large-tilt`` (24 starts tilted 6-12 deg), ``paired`` (23 opposite hole offsets around shared
            held anchors; needs ``--map``, optional ``--selection`` reserved cells) or a JSON task file
  traces    failed episodes of an ``eval --trace-dir`` run: first rim contact, hover or pre-drop state

  python -m humanoid_mujoco_scenes.peg_insert.collect_corrections --source takeover --trigger stall \\
      --checkpoint <ckpt> --device cuda --seed 3000 --episodes 100 --corrections 10 --output datasets/corr_r1
  python -m humanoid_mujoco_scenes.peg_insert.collect_corrections --source traces \\
      --eval-report outputs/eval/run/heldout.json --trace-dir outputs/eval/run/traces --output datasets/corr_traces
"""
from __future__ import annotations

import argparse
import copy
import json
from contextlib import ExitStack
from pathlib import Path

import mujoco
import numpy as np

from humanoid_mujoco_scenes import make_model
from .eval import EpisodeTracker, rollout
from .expert import FPS, ExpertFailure, PegExpert
from .handoffs import DETECTORS, broad_coverage_tasks, large_tilt_tasks, paired_hole_tasks, prepare_insertion, prepare_task, restore, trace_starts
from .recording import CAMERAS, RawDataset, snapshot

REWIND_STEPS = [0, 2, 5, 10, 25, 50]  # trace frames to step back when a state is already unrecoverable


def verified(expert, hold_seconds):
    """Plan on a copy so the live handoff is untouched if the expert cannot finish from it."""
    pilot = PegExpert(expert.model, copy.copy(expert.data), guarded=expert.guarded)
    pilot.closure = expert.closure
    try:
        pilot.takeover(hold_seconds)
        return pilot.inserted() and pilot.released()
    except ExpertFailure:
        return False


def correct(expert, raw, renderers, episode_id, hold_seconds, info):
    """Record the expert finishing from the current held state; returns success."""
    handoff = snapshot(expert)
    try:
        with raw.episode(episode_id, renderers) as writer:
            expert.on_frame = writer
            expert.takeover(hold_seconds)
            if not expert.inserted() or not expert.released():
                raise ExpertFailure("correction failed post-release check")
            summary = writer.commit(handoff)
        raw.add({**info, "seed": episode_id, "success": True, "takeover_strategy": expert.takeover_strategy, **summary})
        return True
    except ExpertFailure as exc:
        raw.add({**info, "seed": episode_id, "success": False, "failure": str(exc)})
        return False
    finally:
        expert.on_frame = None


def from_takeover(args, expert, raw, renderers):
    from humanoid_robot_learning.policy_adapters import load_policy

    policy = load_policy(args.checkpoint, args.device, args.n_action_steps)
    for seed in range(args.seed, args.seed + args.episodes):
        if raw.successes() >= args.corrections:
            break
        if raw.has("source_seed", seed):
            continue
        expert.reset(seed, args.randomize)
        if args.start_stage == "insertion":
            prepare_insertion(expert, args.insertion_case)
        policy.reset(seed)
        tracker = EpisodeTracker(expert, seed, keep_trace=False)
        reason = rollout(policy, expert, renderers, round(args.max_seconds * FPS), tracker, stop=DETECTORS[args.trigger]())
        info = {"source": "takeover", "source_seed": seed, "trigger": args.trigger, "takeover_reason": reason,
                "takeover_at_seconds": tracker.result["control_steps"] / FPS, "policy": policy.describe()}
        if reason is None:
            print(json.dumps({**info, "skipped": "policy succeeded" if tracker.result["success"] else "no handoff"}), flush=True)
            continue
        if args.trigger != "rim-jam" or verified(expert, 0):
            correct(expert, raw, renderers, next_id(raw, args), args.hold_seconds, info)


def pilot(model):
    """Physical dry run of a task (no recording): prepare it, then guarded insertion and release."""
    def run(task):
        expert = PegExpert(model, mujoco.MjData(model), guarded=True)
        layout = task["layout"]
        expert.reset(task["layout_seed"], 0, layout["peg_xy"], layout["hole_xy"])
        try:
            prepare_task(expert, task)
            expert.takeover(0)
            return expert.inserted() and expert.released()
        except ExpertFailure:
            return False
    return run


def load_tasks(args, model):
    if args.tasks == "paired":
        reserved = json.loads(args.selection.read_text()).get("heldout_cells", []) if args.selection else []
        return paired_hole_tasks(json.loads(args.map.read_text())["cells"], reserved, verify=pilot(model))
    generators = {"broad": broad_coverage_tasks, "large-tilt": large_tilt_tasks}
    return generators[args.tasks]() if args.tasks in generators else json.loads(Path(args.tasks).read_text())["tasks"]


def from_tasks(args, expert, raw, renderers):
    for index, task in enumerate(load_tasks(args, expert.model)):
        identity = f"task_{index}_{task['layout_seed']}"
        if raw.has("identity", identity):
            continue
        layout = task.get("layout")
        expert.reset(task["layout_seed"], args.randomize, *((layout["peg_xy"], layout["hole_xy"]) if layout else ()))
        try:
            setup = prepare_task(expert, task)
        except ExpertFailure as exc:
            raw.add({"identity": identity, "seed": -1, "success": False, "failure": f"setup: {exc}", **task})
            continue
        correct(expert, raw, renderers, next_id(raw, args), args.hold_seconds, {"source": "tasks", "identity": identity, "setup": setup, **task})


def from_traces(args, expert, raw, renderers):
    report = json.loads(args.eval_report.read_text())
    for row in (r for r in report["results"] if not r["success"]):
        with np.load(args.trace_dir / f"seed_{row['seed']:06d}.npz") as loaded:
            trace = dict(loaded)
        for kind, index in trace_starts(trace):
            identity = f"{row['seed']}_{kind}_{index}"
            if raw.has("identity", identity):
                continue
            chosen = next((max(0, index - d) for d in REWIND_STEPS if restorable(expert, trace, max(0, index - d))), None)
            info = {"source": "traces", "identity": identity, "source_seed": row["seed"], "kind": kind,
                    "layout": row.get("layout", {}), "condition": "contact_recovery" if "contact" in kind else "alignment"}
            if chosen is None:
                raw.add({**info, "seed": -1, "success": False, "failure": "no recoverable state within rewind window"})
                continue
            restore(expert, trace, chosen)
            correct(expert, raw, renderers, next_id(raw, args), 0, {**info, "handoff_index": chosen})


def restorable(expert, trace, index):
    restore(expert, trace, index)
    return expert.contacts() and verified(expert, 0)


def next_id(raw, args):
    return max([r["seed"] for r in raw.results] + [args.first_id - 1]) + 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["takeover", "tasks", "traces"], required=True)
    p.add_argument("--output", type=Path, required=True, help="raw collection (resumed if it exists)")
    p.add_argument("--first-id", type=int, default=100000, help="first episode id in a new collection")
    p.add_argument("--hold-seconds", type=float, default=0.5, help="secure the inherited grasp before moving")
    p.add_argument("--guarded", action="store_true", help="guarded expert insertion (see expert.py); always on for traces")
    p.add_argument("--randomize", type=float, default=0.01)
    policy = p.add_argument_group("takeover")
    policy.add_argument("--checkpoint", type=Path)
    policy.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    policy.add_argument("--n-action-steps", type=int)
    policy.add_argument("--trigger", choices=sorted(DETECTORS), default="approach")
    policy.add_argument("--seed", type=int, default=2000)
    policy.add_argument("--episodes", type=int, default=100, help="policy rollouts to attempt")
    policy.add_argument("--corrections", type=int, default=25, help="stop after this many successful corrections")
    policy.add_argument("--max-seconds", type=float, default=60)
    policy.add_argument("--start-stage", choices=("full", "insertion"), default="full")
    policy.add_argument("--insertion-case", default="aligned")
    p.add_argument("--tasks", help="broad | large-tilt | paired | JSON file with {'tasks': [...]}")
    p.add_argument("--map", type=Path, help="workspace_map.json (paired tasks)")
    p.add_argument("--selection", type=Path, help="selection plan whose heldout_cells paired tasks must avoid")
    p.add_argument("--eval-report", type=Path)
    p.add_argument("--trace-dir", type=Path)
    args = p.parse_args()
    required = {"takeover": ["checkpoint"], "tasks": ["tasks"], "traces": ["eval_report", "trace_dir"]}[args.source]
    if any(getattr(args, name) is None for name in required):
        p.error(f"--source {args.source} needs " + ", ".join("--" + n.replace("_", "-") for n in required))
    raw = RawDataset(args.output, expert_only=True, source=args.source)
    model = make_model("peg_insert", cameras=CAMERAS)
    expert = PegExpert(model, mujoco.MjData(model), guarded=args.guarded or args.source == "traces")
    with ExitStack() as stack:
        renderers = {n: stack.enter_context(mujoco.Renderer(model, h, w)) for n, (h, w) in CAMERAS.items()}
        {"takeover": from_takeover, "tasks": from_tasks, "traces": from_traces}[args.source](args, expert, raw, renderers)
    print(f"Corrections: {raw.successes()} successful / {len(raw.results)} attempts in {args.output}")
    if args.source == "takeover" and raw.successes() < args.corrections:
        raise SystemExit("correction target not reached; do not train on this round")


if __name__ == "__main__":
    main()
