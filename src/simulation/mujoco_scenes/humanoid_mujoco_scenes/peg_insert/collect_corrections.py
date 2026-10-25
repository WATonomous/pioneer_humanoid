"""Collect expert corrections (DAgger-style labels) from held-peg states, for fine-tuning a policy.

Only expert frames from the handoff to verified insertion + release are recorded; the policy's
own actions are never labels. Every handoff is physical: a policy rollout, an expert setup, or a
restored eval-trace state. A failed correction is logged in ``manifest.json`` and discarded.

Sources:
  takeover  roll out ``--checkpoint``; hand off when ``--trigger`` fires
            (approach: within 3 cm of the hole | stall: stalled/tilted/misaligned at the rim | rim-jam |
            pre-grasp: hand near the upright peg, recorded with ``--skill grasp`` up to a verified lift)
  tasks     expert-prepared held starts: ``--tasks broad`` (80 offsets/heights/tilts around the hole),
            ``large-tilt`` (24 starts tilted 6-12 deg) or a JSON file of insertion cases / task dicts
  traces    failed episodes of ``eval --trace-dir`` runs: first rim contact, hover or pre-drop state; with
            ``--max-corrections`` the cap is spread round-robin over layout groups, and ``--group-quota``
            reserves part of it for a group (e.g. ``hole_3_7=20``)

  python -m humanoid_mujoco_scenes.peg_insert.collect_corrections --source takeover --trigger stall \\
      --checkpoint <ckpt> --device cuda --seed 3000 --episodes 100 --corrections 10 --output datasets/corr_r1
  python -m humanoid_mujoco_scenes.peg_insert.collect_corrections --source traces \\
      --eval-report outputs/eval/run/heldout.json --trace-dir outputs/eval/run/traces --output datasets/corr_traces
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import mujoco
import numpy as np

from humanoid_mujoco_scenes import make_model
from .eval import EpisodeTracker, rollout
from .expert import FPS, ExpertFailure, PegExpert
from .handoffs import DETECTORS, broad_coverage_tasks, large_tilt_tasks, prepare_insertion, prepare_task, restore, trace_starts
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


def finish(expert, skill, hold_seconds):
    """Expert behaviour after the handoff, with its verification."""
    if skill == "grasp":
        expert.takeover_grasp()
        if not expert.lifted():
            raise ExpertFailure("grasp correction did not end with a lifted, held peg")
        return
    expert.takeover(hold_seconds)
    if not expert.inserted() or not expert.released():
        raise ExpertFailure("correction failed post-release check")


def correct(expert, raw, renderers, episode_id, hold_seconds, info, skill="insert"):
    """Record the expert finishing from the current state; returns success."""
    handoff = snapshot(expert)
    try:
        with raw.episode(episode_id, renderers) as writer:
            expert.on_frame = writer
            finish(expert, skill, hold_seconds)
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
            correct(expert, raw, renderers, next_id(raw, args), args.hold_seconds, info, args.skill)


def enough(args, raw):
    return args.max_corrections is not None and raw.successes() >= args.max_corrections


def from_tasks(args, expert, raw, renderers):
    tasks = {"broad": broad_coverage_tasks, "large-tilt": large_tilt_tasks}.get(args.tasks, lambda: json.loads(Path(args.tasks).read_text())["tasks"])()
    for index, task in enumerate(tasks):
        if enough(args, raw):
            return
        identity = f"task_{index}_{task['layout_seed']}"
        if raw.has("identity", identity):
            continue
        expert.reset(task["layout_seed"], args.randomize)
        try:
            setup = prepare_task(expert, task)
        except ExpertFailure as exc:
            raw.add({"identity": identity, "seed": -1, "success": False, "failure": f"setup: {exc}", **task})
            continue
        correct(expert, raw, renderers, next_id(raw, args), args.hold_seconds, {"source": "tasks", "identity": identity, "setup": setup, **task})


def layout_group(row):
    """Eval layout group: ``<moved object>_<band>`` (e.g. ``hole_3_7``), or ``nominal`` for seeded layouts."""
    layout = row.get("layout") or {}
    return f"{layout['group']}_{layout['band']}" if "group" in layout else "nominal"


def trace_candidates(reports, trace_dirs):
    """Failed episodes of every report, interleaved round-robin across layout groups (not first-come)."""
    by_group = {}
    for report, trace_dir in zip(reports, trace_dirs):
        for row in json.loads(Path(report).read_text())["results"]:
            if not row["success"]:
                by_group.setdefault(layout_group(row), []).append((row, Path(trace_dir)))
    queues = list(by_group.values())
    while any(queues):
        for queue in queues:
            if queue:
                yield queue.pop(0)


def has_room(args, raw, group):
    """Quota groups fill up to their quota; all other groups share what is left of ``--max-corrections``."""
    counts = Counter(layout_group(r) for r in raw.results if r["success"])
    if group in args.group_quota:
        return counts[group] < args.group_quota[group]
    if args.max_corrections is None:
        return True
    shared = args.max_corrections - sum(args.group_quota.values())
    return sum(n for g, n in counts.items() if g not in args.group_quota) < shared


def from_traces(args, expert, raw, renderers):
    for row, trace_dir in trace_candidates(args.eval_report, args.trace_dir):
        group = layout_group(row)
        with np.load(trace_dir / f"seed_{row['seed']:06d}.npz") as loaded:
            trace = dict(loaded)
        for kind, index in trace_starts(trace):
            if enough(args, raw):
                return
            if not has_room(args, raw, group):
                break
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
    p.add_argument("--max-corrections", type=int, help="tasks/traces: stop once the collection holds this many successful corrections")
    p.add_argument("--skill", choices=["insert", "grasp"], default="insert",
                   help="what the expert finishes: insert + release, or (with --trigger pre-grasp) grasp + lift only")
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
    p.add_argument("--tasks", help="broad | large-tilt | JSON file with {'tasks': [...]}")
    p.add_argument("--eval-report", type=Path, action="append", help="traces: eval report; repeat with a matching --trace-dir")
    p.add_argument("--trace-dir", type=Path, action="append")
    p.add_argument("--group-quota", nargs="*", default=[], metavar="GROUP=N",
                   help="traces: reserve N corrections for a layout group (e.g. hole_3_7=20); other groups share the rest of --max-corrections")
    args = p.parse_args()
    required = {"takeover": ["checkpoint"], "tasks": ["tasks"], "traces": ["eval_report", "trace_dir"]}[args.source]
    if any(getattr(args, name) is None for name in required):
        p.error(f"--source {args.source} needs " + ", ".join("--" + n.replace("_", "-") for n in required))
    if args.source == "traces" and len(args.eval_report) != len(args.trace_dir):
        p.error("give one --trace-dir per --eval-report")
    args.group_quota = {k: int(v) for k, v in (q.split("=") for q in args.group_quota)}
    if args.max_corrections is not None and sum(args.group_quota.values()) > args.max_corrections:
        p.error("group quotas exceed --max-corrections")
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
