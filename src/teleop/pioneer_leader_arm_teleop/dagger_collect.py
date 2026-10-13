"""DAgger data for a trained policy: the policy drives, then the synthetic operator takes over and finishes.

    MUJOCO_GL=egl python dagger_collect.py --scene peg_insert --record --num_episodes 60 \
        --policy outputs/train/pioneer_act_peg_insert/checkpoints/last/pretrained_model \
        --dataset_root datasets/pioneer_v1_left_arm/sim_synthetic/peg_insert_dagger

A policy trained on clean demos drifts into states the demos never visit (a peg a millimetre off the hole) and
has no example of getting out of them. Here the policy runs from the start of an episode (sim_policy_eval.py's
loop); at a takeover moment the simulated operator (human_operator.py) picks the leader up where the arm is and
completes the task with the scene's plan in ``resume`` mode. Only the operator's part is recorded, as one episode,
and only if the task then succeeds: so the new episodes start in states the policy actually reaches and show the
correction. Append them to a copy of the demo dataset, fine-tune, repeat (HG-DAgger with a scripted expert).

Takeover moment, per episode (``--late`` of them: only at ``--limit_s``, i.e. real failures): half the time once the peg is carried to within ``--near`` of the hole plus up to
``--delay_s`` (where the policy fails), otherwise a uniformly random time; never later than ``--limit_s``.
Episodes the policy finishes on its own give no data. Seeds: keep clear of the demos (0+) and eval (1000+).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_teleop import TeleopSim  # noqa: E402  (also puts the repo packages on sys.path)

from humanoid_robot_learning.sim_teleop_record import add_record_args, load_record_schema, make_sim_recorder  # noqa: E402

TAKES_FILE = "meta/dagger_takes.jsonl"
MAX_EXPERT_S = 25.0


class _TooLong(RuntimeError):
    pass


def take_over(sim: TeleopSim, rng, pace: float):
    """An operator whose leader is where the arm's targets are now (no jump), with the mapping engaged."""
    import mujoco
    import numpy as np
    from human_operator import HumanOperator, Style, _Channel

    op = HumanOperator(sim.model, lambda: sim.data, sim.tick, sim.joints, sim.home, sim.control_dt, rng, style=Style.sample(rng, pace))
    op.home_tcp = op.pos.goal().copy()
    q = np.array(sim.data.ctrl[sim.arm_acts], float)
    g = float(np.clip((sim.data.ctrl[sim.grip_acts[0]] - sim.grip_open[0]) / (sim.grip_closed[0] - sim.grip_open[0]), 0.0, 1.0))
    op.q = q.copy()
    op.kin.qpos[op.qadr] = q
    mujoco.mj_kinematics(sim.model, op.kin)
    op.pos = _Channel(op._tcp(op.kin))
    op.rot = _Channel(op._rotvec(op.kin.xmat[op.hand].reshape(3, 3)))
    op.grip = _Channel([g])
    sim.mapping.engaged, sim.mapping.target, sim.mapping.grip = True, list(q), g
    return op, g


def main() -> None:
    parser = argparse.ArgumentParser(description="DAgger: policy rollouts finished by the synthetic operator.")
    parser.add_argument("--scene", default="peg_insert")
    parser.add_argument("--policy", required=True, help="pretrained_model directory of a checkpoint")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--max_tries", type=int, default=0, help="rollouts at most (default: 4 x num_episodes)")
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--near", type=float, default=0.04, help="m: peg this close to the hole (xy) starts the takeover clock")
    parser.add_argument("--delay_s", type=float, default=4.0)
    parser.add_argument("--limit_s", type=float, default=22.0)
    parser.add_argument("--late", type=float, default=0.0,
                        help="share of rollouts left to the policy until --limit_s: takeovers only where it really fails")
    parser.add_argument("--device", default="cuda")
    add_record_args(parser, task_description="")
    args = parser.parse_args()
    if args.scene != "peg_insert":
        raise SystemExit("only peg_insert has a resume plan so far")
    args.record = True
    record = load_record_schema(parser, args)
    if args.dataset_root is None:
        raise SystemExit("--dataset_root: a copy of the demo dataset to append to")

    import numpy as np
    import torch
    from sim_policy_eval import RENDER_SIZE, load_policy, policy_frame
    from synthetic_tasks import TASKS, TaskFailed

    task = TASKS[args.scene]
    args.task_description = args.task_description or task.instruction
    policy, preprocessor, postprocessor, cfg, cameras = load_policy(args.policy, args.device)
    device = torch.device(args.device)
    probe = TeleopSim(args.scene, 0, None)
    recorder, record_every = make_sim_recorder(args, record, device="cpu", sim_dt=probe.control_dt, buffer_capacity_s=MAX_EXPERT_S + 5.0)
    fps = round(1.0 / (record_every * probe.control_dt))
    probe.close()
    root = Path(recorder.dataset_root)
    done = [json.loads(line) for line in (root / TAKES_FILE).read_text().splitlines()] if (root / TAKES_FILE).exists() else []
    tried = {t["seed"] for t in done if t.get("policy") == str(args.policy)}
    saved = sum(1 for t in done if t.get("policy") == str(args.policy) and t["saved"])
    stats = {"policy_solved": 0, "saved": saved, "expert_failed": 0}

    seed, t0 = args.seed - 1, time.time()
    while saved < args.num_episodes and seed - args.seed + 1 < (args.max_tries or 4 * args.num_episodes):
        seed += 1
        if seed in tried:
            continue
        rng = np.random.default_rng([seed, 2])
        sim = TeleopSim(args.scene, seed, {n: RENDER_SIZE for n in cameras})
        model, data = sim.model, sim.data
        peg, block = model.body("peg").id, model.body("hole_block").id
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        mode = "late" if rng.random() < args.late else "near" if rng.random() < 0.5 else "random"
        at = rng.uniform(2.0, 18.0) if mode == "random" else None
        solved = False
        while data.time < args.limit_s:
            policy_frame(sim, policy, preprocessor, postprocessor, device, fps, task.instruction, sizes=cameras)
            if task.success(model, data):
                solved = True
                break
            if mode == "near" and at is None and data.xpos[peg][2] > data.xpos[block][2] and np.linalg.norm((data.xpos[peg] - data.xpos[block])[:2]) < args.near:
                at = data.time + rng.uniform(0.0, args.delay_s)
            if at is not None and data.time >= at:
                break
        entry = {"seed": seed, "policy": str(args.policy), "mode": mode, "takeover_s": round(float(data.time), 2), "saved": False}
        if solved:
            stats["policy_solved"] += 1
            entry["notes"] = ["policy finished on its own"]
        else:
            op, _ = take_over(sim, rng, args.pace)
            t_take, notes = data.time, []

            def sink(action, state, task_text, extras, qpos):
                if data.time - t_take > MAX_EXPERT_S:
                    raise _TooLong()
                recorder.push_frame_to_buffer(action, state, sim.render_at(qpos), extras=extras, task=task_text)

            sim.frame_sink, sim.record_every = sink, record_every
            try:
                task.plan(op, model, data, notes, resume=True)
            except TaskFailed as exc:
                notes.append(f"gave up: {exc}")
            except _TooLong:
                notes.append("gave up: too long")
            ok = bool(task.success(model, data)) and not any(n.startswith("gave up") for n in notes)
            entry.update(notes=notes, expert_s=round(float(data.time - t_take), 2))
            if ok and recorder.num_buffered_frames > fps:
                entry.update(saved=True, episode_index=recorder.dataset.meta.total_episodes)
                recorder.save_episode()
                recorder.wait_saved()
                saved += 1
                stats["saved"] = saved
            else:
                recorder.cancel_recording()
                stats["expert_failed"] += 1
        sim.close()
        with open(root / TAKES_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[DAGGER] seed {seed}: {mode:6s} takeover {entry['takeover_s']:5.1f} s  "
              f"{'saved' if entry['saved'] else 'policy ok' if solved else 'DISCARDED'}  {entry.get('notes') or ''}  "
              f"{stats}  ({time.time() - t0:.0f} s wall)", flush=True)
    recorder.finalize()
    print(f"[DAGGER] {stats} -> {root}")


if __name__ == "__main__":
    main()
