#!/usr/bin/env python3
"""Generic check/adjust convergence loop for iterative Isaac Sim scene corrections.

This is NOT a daemon command — isaac_session_daemon.py's JSON protocol is
deliberately unchanged by this file. The daemon stays atomic (one action or
one state-read per command); the actual "check, correct, recheck" loop lives
here, on the host, composed entirely out of the daemon's existing atomic
commands (bbox/overlap/distance/query/set_pose/spawn_*) via send_command().
Same split as a Kubernetes controller reconciliation loop against a plain
CRUD API, or a WebDriver test framework's wait/retry helpers against a plain
click/screenshot RPC protocol — keep the execution layer dumb and fast, keep
the convergence logic in a client library where it can be arbitrary Python
instead of a constrained wire-protocol DSL.

Prerequisite: `set_pose` actually surviving a later spawn/remove (fixed —
see isaac_session_daemon.py's handle_set_pose docstring). Before that fix,
a loop like this had no way to make a correction that would actually stick.

Only reach for this when the target can't be computed up front — a scaled
prefab asset's real physics-contact height not matching its visual bbox top
(see SKILL.md), or an inherently trial-and-error placement (does this fit,
do these overlap after a nudge). If you already know the exact target
value, just spawn there directly; this loop is pure overhead for that case.

Usage as a library — write a small check_fn/adjust_fn per scenario:

    from verify_loop import until, is_resting_on, close_gap_along_axis

    result = until(
        check_fn=lambda: is_resting_on("VialRack", "Table", tol=0.005),
        adjust_fn=close_gap_along_axis("VialRack"),
    )
    print(result.success, result.iterations, result.trace)

Usage as a quick CLI check (see `if __name__ == "__main__"` below):

    python3 verify_loop.py rest --name Rack --on Table --tol 0.005
"""
import argparse
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

from isaac_session_client import send_command


@dataclass
class LoopResult:
    success: bool
    iterations: int
    trace: list = field(default_factory=list)


def until(
    check_fn: Callable[[], tuple],
    adjust_fn: Callable[[dict], None],
    max_iters: int = 20,
    timeout_s: float = 30.0,
) -> LoopResult:
    """Run check_fn() -> if not ok, adjust_fn(info) -> check_fn() again, until
    ok or a bound (iteration count or wall-clock) is hit. Returns the full
    per-iteration trace, not just the final verdict, so a caller can see
    *why* it converged or didn't rather than getting a bare pass/fail.
    """
    trace = []
    start = time.time()
    for i in range(1, max_iters + 1):
        ok, info = check_fn()
        trace.append({"iteration": i, "ok": ok, **info})
        if ok:
            return LoopResult(success=True, iterations=i, trace=trace)
        if time.time() - start > timeout_s:
            break
        adjust_fn(info)
    return LoopResult(success=False, iterations=len(trace), trace=trace)


# ---------------------------------------------------------------------------
# Prebuilt checks — thin wrappers around bbox/overlap so callers don't
# hand-roll AABB math per scenario.
# ---------------------------------------------------------------------------

def is_resting_on(a: str, b: str, tol: float = 0.005, axis: int = 2):
    """Is a's bottom face within `tol` of b's top face along `axis` (default z)?
    Positive gap = a is floating above b; negative = a is embedded in b.

    Single-axis only — this does NOT check that a is actually within b's
    footprint on the other two axes. Confirmed by testing: an object placed
    entirely outside a table's x/y range still reports `ok=True` once its z
    gap closes, because nothing here checks x/y containment. Combine with
    `no_overlap`/an explicit x/y bounds check (or just verify x/y placement
    up front, since that's usually known ahead of time) if "genuinely
    resting on top of, not just aligned in z with" is what's actually needed.
    """
    bbox_a = send_command({"cmd": "bbox", "name": a})
    bbox_b = send_command({"cmd": "bbox", "name": b})
    gap = bbox_a["min"][axis] - bbox_b["max"][axis]
    return abs(gap) < tol, {"gap": gap, "bbox_a": bbox_a, "bbox_b": bbox_b}


def no_overlap(names: list):
    """Is every pair among `names` free of AABB overlap?"""
    for x, y in itertools.combinations(names, 2):
        r = send_command({"cmd": "overlap", "name_a": x, "name_b": y})
        if r["overlap"]:
            return False, {"overlapping_pair": [x, y], "bbox_a": r["bbox_a"], "bbox_b": r["bbox_b"]}
    return True, {}


def is_stacked_on(a: str, b: str, z_tol: float = 0.003, xy_tol: float = 0.005, axis: int = 2):
    """Is `a` properly stacked on `b` — both resting (z-gap within z_tol)
    AND centered (offset from b's center on the other two axes within
    xy_tol)?

    Exists because of a real failure, not a hypothetical: is_resting_on
    alone (z-gap only) reported success closing a stack tightly, but the
    stack then progressively collapsed/slid apart over further physics
    steps — a landing box lands with a few degrees of tilt, and with
    nothing correcting x/y alignment, that tilt compounds into sliding
    instead of settling. Confirmed by testing (not assumed): a 3-box stack
    built with is_resting_on alone reached a tight z-gap immediately, but
    re-querying after 60 more physics steps showed one box moved ~0.3m
    sideways and down — genuinely fell off, not just noise. Use this
    instead of is_resting_on whenever the object needs to stay put under
    continued simulation, not just report a passing check once.
    """
    bbox_a = send_command({"cmd": "bbox", "name": a})
    bbox_b = send_command({"cmd": "bbox", "name": b})
    other_axes = [i for i in range(3) if i != axis]
    gap = bbox_a["min"][axis] - bbox_b["max"][axis]
    center_a = [(bbox_a["min"][i] + bbox_a["max"][i]) / 2 for i in range(3)]
    center_b = [(bbox_b["min"][i] + bbox_b["max"][i]) / 2 for i in range(3)]
    offsets = {i: center_a[i] - center_b[i] for i in other_axes}
    ok = abs(gap) < z_tol and all(abs(o) < xy_tol for o in offsets.values())
    return ok, {"gap": gap, "offsets": offsets, "axis": axis, "bbox_a": bbox_a, "bbox_b": bbox_b}


def is_within_bounds(name: str, min_xyz, max_xyz):
    """Does name's bbox center lie within an axis-aligned region?"""
    bbox = send_command({"cmd": "bbox", "name": name})
    center = [(bbox["min"][i] + bbox["max"][i]) / 2 for i in range(3)]
    ok = all(min_xyz[i] <= center[i] <= max_xyz[i] for i in range(3))
    return ok, {"center": center, "bbox": bbox}


# ---------------------------------------------------------------------------
# Prebuilt adjustments
# ---------------------------------------------------------------------------

def close_gap_along_axis(name: str, gap_key: str = "gap", bbox_key: str = "bbox_a", axis: int = 2, damping: float = 0.8):
    """Returns an adjust_fn that nudges `name` along `axis` by `damping` times
    the signed gap in info[gap_key] (e.g. from is_resting_on). Damped rather
    than a full 1:1 correction — a scaled/prefab asset's visual bbox and its
    actual PhysX contact surface can diverge (see SKILL.md's packing-table
    gotcha), so a full-gap jump risks overshoot/oscillation instead of
    smooth convergence.

    Reads the current position from `info[bbox_key]`'s bbox center — NOT
    from query() — on purpose. query() reads the PhysX tensor view, which a
    kinematic object's set_pose (a direct USD-prim write) never touches;
    using query() here made every iteration nudge from the same stale
    baseline instead of the actual last position, and it genuinely
    oscillated instead of converging (caught by testing this loop against
    a live daemon, not just reading the code). bbox is the correct source
    for exactly the same reason set_pose's own docstring documents.
    """
    def adjust(info: dict):
        gap = info[gap_key]
        bbox = info[bbox_key]
        cur = [(bbox["min"][i] + bbox["max"][i]) / 2 for i in range(3)]
        new_pos = list(cur)
        new_pos[axis] -= gap * damping
        send_command({"cmd": "set_pose", "name": name, "pos": new_pos})
    return adjust


def settle_stack(name: str, damping: float = 0.7):
    """Adjust_fn pairing with is_stacked_on — corrects the z-gap AND the
    x/y center offset together in one combined move per iteration, not as
    two separate corrections that could fight each other across
    iterations (e.g. a z-only pass re-introducing an x/y offset if the
    object rotates slightly as it drops).
    """
    def adjust(info: dict):
        bbox_a = info["bbox_a"]
        cur = [(bbox_a["min"][i] + bbox_a["max"][i]) / 2 for i in range(3)]
        new_pos = list(cur)
        new_pos[info["axis"]] -= info["gap"] * damping
        for i, offset in info["offsets"].items():
            new_pos[i] -= offset * damping
        send_command({"cmd": "set_pose", "name": name, "pos": new_pos})
    return adjust


def _main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("rest", help="nudge `--name` until it rests on `--on` within `--tol` (z-gap only)")
    p.add_argument("--name", required=True)
    p.add_argument("--on", required=True)
    p.add_argument("--tol", type=float, default=0.005)
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--damping", type=float, default=0.8)

    p = sub.add_parser("stack", help="nudge `--name` until it's resting AND centered on `--on` "
                       "(use this over `rest` for anything that needs to stay put under continued physics)")
    p.add_argument("--name", required=True)
    p.add_argument("--on", required=True)
    p.add_argument("--z-tol", type=float, default=0.003)
    p.add_argument("--xy-tol", type=float, default=0.005)
    p.add_argument("--max-iters", type=int, default=25)
    p.add_argument("--damping", type=float, default=0.7)

    args = parser.parse_args()
    if args.cmd == "rest":
        result = until(
            check_fn=lambda: is_resting_on(args.name, args.on, tol=args.tol),
            adjust_fn=close_gap_along_axis(args.name, damping=args.damping),
            max_iters=args.max_iters,
        )
        print(f"success={result.success} iterations={result.iterations}")
        for step in result.trace:
            print(f"  iter {step['iteration']}: gap={step['gap']:.5f} ok={step['ok']}")
    elif args.cmd == "stack":
        result = until(
            check_fn=lambda: is_stacked_on(args.name, args.on, z_tol=args.z_tol, xy_tol=args.xy_tol),
            adjust_fn=settle_stack(args.name, damping=args.damping),
            max_iters=args.max_iters,
        )
        print(f"success={result.success} iterations={result.iterations}")
        for step in result.trace:
            off = step["offsets"]
            print(f"  iter {step['iteration']}: gap={step['gap']:.5f} offsets={off} ok={step['ok']}")


if __name__ == "__main__":
    _main()
