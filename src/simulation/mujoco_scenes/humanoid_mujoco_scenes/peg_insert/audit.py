"""Policy audits that need no training: does the policy *see* the hole, and how fast does it plan?

hole-shift  Freeze a held-peg state (robot/peg qpos, qvel, ctrl identical), move only the static
            block by +-3 / +-10 mm in x and y for the *rendered* observation, predict one chunk
            with shared noise, then execute every chunk in the same unshifted world. The central
            difference of peg displacement along the shifted axis, divided by twice the shift, is
            the response gain: 1 = fully follows the hole, 0 = ignores it. The expert's own gain
            from the same state is the reference.
latency     Warmed, synchronized chunk-generation time on one observation (no rendering/IO).

  python -m humanoid_mujoco_scenes.peg_insert.audit hole-shift --checkpoint <ckpt> --seeds 1000 1001 1002 \\
      --report outputs/audit/hole_shift.json
  python -m humanoid_mujoco_scenes.peg_insert.audit latency --checkpoint <ckpt> --device cuda --report outputs/audit/latency.json
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import mujoco
import numpy as np

from humanoid_mujoco_scenes import make_model
from .eval import render
from .expert import PegExpert
from .handoffs import prepare_insertion, restore
from .recording import write_json

SHIFTS_MM = [3, 10]
HORIZONS = [10, 25, 50]     # control steps (0.4 s, 1 s, 2 s)
NOISE_SEEDS = [7101, 7102, 7103]


class _Captured(Exception):
    pass


def held_states(args, model):
    """Prepared aligned insertion handoffs per seed, plus any saved handoff.npz files."""
    for seed in args.seeds:
        expert = PegExpert(model, mujoco.MjData(model))
        expert.reset(seed)
        prepare_insertion(expert, "aligned")
        yield f"seed_{seed}", expert
    for path in args.handoff:
        expert = PegExpert(model, mujoco.MjData(model))
        with np.load(path) as state:
            restore(expert, dict(state))
        yield str(path), expert


def fork(model, data, hole):
    model.body_pos[model.body("hole_block").id] = hole
    copy_ = mujoco.MjData(model)
    mujoco.mj_copyData(copy_, model, data)
    mujoco.mj_forward(model, copy_)
    expert = PegExpert(model, copy_, guarded=True)
    expert.closure = float(np.clip(np.mean(copy_.ctrl[expert.grip_a] / [-0.05, 0.05]), 0, 1))
    return expert


def displacement(model, data, hole, actions):
    """Peg displacement (mm) after each horizon when ``actions`` run in the world with ``hole``."""
    expert = fork(model, data, hole)
    origin, out = expert.peg_pos, {}
    for step, action in enumerate(actions, 1):
        expert.apply(action)
        expert.step()
        if step in HORIZONS:
            out[step] = ((expert.peg_pos - origin) * 1000).tolist()
    return out


def expert_chunk(model, data, hole, length):
    expert, actions = fork(model, data, hole), []

    def capture(_, action):
        if len(actions) == length:
            raise _Captured
        actions.append(action)
    expert.on_frame = capture
    try:
        expert.takeover(0)
    except _Captured:
        pass
    return np.array(actions)


def audit_state(policy, renderers, model, name, expert):
    hole = model.body_pos[expert.block].copy()
    hole[:2] = expert.peg_pos[:2]  # centre the hole under the held peg; shifts are then pure offsets
    reference = copy.copy(expert.data)
    rows = []
    for mm in SHIFTS_MM:
        for axis in (0, 1):
            responses = {}
            for sign in (1, -1):
                shifted = hole.copy()
                shifted[axis] += sign * mm / 1000
                view = fork(model, reference, shifted)
                images = render(renderers, view.data)
                chunks = [policy.predict_chunk(view.state(), images, noise_seed=s) for s in NOISE_SEEDS]
                responses[sign] = {"policy": [displacement(model, reference, hole, c) for c in chunks],
                                   "expert": displacement(model, reference, hole, expert_chunk(model, reference, shifted, max(HORIZONS)))}
            rows.append(gain_row(name, mm, axis, responses))
    model.body_pos[expert.block] = hole
    return rows


def gain_row(name, mm, axis, responses):
    def gain(plus, minus, n):
        return (plus[n][axis] - minus[n][axis]) / (2 * mm)
    row = {"state": name, "shift_mm": mm, "axis": "xy"[axis]}
    for n in HORIZONS:
        row[f"policy_gain_{n}"] = float(np.median([gain(a, b, n) for a, b in zip(responses[1]["policy"], responses[-1]["policy"])]))
        row[f"expert_gain_{n}"] = gain(responses[1]["expert"], responses[-1]["expert"], n)
    return row


def hole_shift(args, policy):
    model = make_model("peg_insert", cameras=policy.cameras)
    rows = []
    with mujoco.Renderer(model, *policy.cameras["ego"]) as ego, mujoco.Renderer(model, *policy.cameras["wrist_left"]) as wrist:
        for name, expert in held_states(args, model):
            rows += audit_state(policy, {"ego": ego, "wrist_left": wrist}, model, name, expert)
            print("AUDIT_STATE", name, rows[-1], flush=True)
    summary = {f"{mm}mm": {f"median_{who}_gain_{n}": float(np.median([r[f"{who}_gain_{n}"] for r in rows if r["shift_mm"] == mm]))
                           for who in ("policy", "expert") for n in HORIZONS} for mm in SHIFTS_MM}
    return {"rows": rows, "summary": summary}


def latency(args, policy):
    model = make_model("peg_insert", cameras=policy.cameras)
    expert = PegExpert(model, mujoco.MjData(model))
    expert.reset(args.seeds[0])
    with mujoco.Renderer(model, *policy.cameras["ego"]) as renderer:
        images = render({n: renderer for n in policy.cameras}, expert.data)
    synchronize = __import__("torch").cuda.synchronize if args.device == "cuda" else (lambda: None)
    for _ in range(10):
        policy.predict_chunk(expert.state(), images, noise_seed=0)
    times = []
    for i in range(50):
        synchronize()
        start = time.perf_counter()
        policy.predict_chunk(expert.state(), images, noise_seed=i)
        synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {"chunks": 50, "mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "note": "includes input normalization and vision encoding; excludes rendering"}


def main():
    from humanoid_robot_learning.policy_adapters import load_policy

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audit", choices=["hole-shift", "latency"])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002], help="layouts for prepared aligned handoffs")
    p.add_argument("--handoff", type=Path, nargs="*", default=[], help="saved handoff.npz states (e.g. from corrections)")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    policy = load_policy(args.checkpoint, args.device)
    result = {"audit": args.audit, **policy.describe(), **(hole_shift if args.audit == "hole-shift" else latency)(args, policy)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, result)
    print(result.get("summary", result), flush=True)


if __name__ == "__main__":
    main()
