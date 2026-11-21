"""Iterated correction rounds: collect expert takeovers from the current policy, convert them, continue
training on all data so far, evaluate unassisted; stop at ``--target`` success or after ``--rounds``.

Each step is the normal CLI of its tool (``collect_corrections``, ``raw_to_lerobot``, the policy
trainer, ``eval``) run as a subprocess, and is skipped when its output already exists, so a
resubmitted job resumes where it stopped. ``<root>/summary.json`` records every round.

  python -m humanoid_mujoco_scenes.peg_insert.dagger --root outputs/dagger/act_r1 \\
      --checkpoint outputs/train/act/checkpoints/150000 --base-dataset datasets/pioneer_peg_insert \\
      --trigger stall --corrections 10 --steps-per-round 6000 --batch-size 8 \\
      --phase-config src/robot_learning/config/peg_insert_phases.json \\
      --eval-layouts outputs/peg_workspace/heldout_layouts.json --n-action-steps 25 --device cuda
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .recording import write_json


def run_module(module, *argv):
    command = [sys.executable, "-m", module, *map(str, argv)]
    print("RUN", json.dumps(command), flush=True)
    subprocess.run(command, check=True)


def act_step(checkpoint):
    return json.loads((Path(checkpoint) / "training_state/training_step.json").read_text())["step"]


def latest_checkpoint(training_dir, fallback):
    saved = sorted((p for p in (training_dir / "checkpoints").glob("[0-9]*") if (p / "training_state").exists()), key=lambda p: int(p.name))
    return saved[-1] if saved else fallback


def train_act(args, checkpoint, datasets, directory):
    """Resume the ACT optimizer/scheduler from ``checkpoint`` for ``--steps-per-round`` more updates."""
    training = directory / "training"
    end = act_step(checkpoint) + args.steps_per_round
    resume = latest_checkpoint(training, checkpoint)
    if act_step(resume) < end:
        extra = [f"--extra-dataset={d}" for d in datasets[1:]]
        phases = [f"--phase-config={args.phase_config}", "--phase-metrics"] if args.phase_config else []
        run_module("humanoid_robot_learning.lerobot_finetune", f"--config_path={resume / 'pretrained_model/train_config.json'}",
                   "--resume=true", f"--dataset.root={datasets[0]}", "--dataset.video_backend=pyav", *extra, *phases,
                   f"--steps={end}", f"--batch_size={args.batch_size}", f"--save_freq={args.steps_per_round}",
                   f"--policy.n_action_steps={args.n_action_steps}", "--policy.temporal_ensemble_coeff=null",
                   f"--policy.device={args.device}", "--policy.push_to_hub=false", "--eval_freq=0",
                   "--wandb.enable=false", "--num_workers=4", f"--output_dir={training}")
    return training / "checkpoints" / f"{end:06d}", training / "checkpoints" / f"{end:06d}" / "pretrained_model"


TRAINERS = {"act": train_act}


def run_round(args, index, checkpoint, policy_path, datasets):
    directory = args.root / f"round_{index:02d}"
    raw, dataset, report = directory / "raw", directory / "dataset", directory / "eval.json"
    if not (raw / "manifest.json").exists() or json.loads((raw / "manifest.json").read_text()).get("complete") is not True:
        run_module("humanoid_mujoco_scenes.peg_insert.collect_corrections", "--source", "takeover", "--trigger", args.trigger,
                   "--checkpoint", policy_path, "--device", args.device, "--n-action-steps", args.n_action_steps,
                   "--seed", args.seed + index * args.attempts, "--episodes", args.attempts,
                   "--corrections", args.corrections, "--output", raw)
        manifest = json.loads((raw / "manifest.json").read_text())
        write_json(raw / "manifest.json", {**manifest, "complete": True})
    if not (dataset / "phase_index.json").exists():
        run_module("humanoid_robot_learning.raw_to_lerobot", "--raw", raw, "--root", dataset,
                   "--repo-id", "humanoid/pioneer_peg_insert_corrected")
    datasets = [*datasets, str(dataset.resolve())]
    checkpoint, policy_path = TRAINERS[args.policy_type](args, checkpoint, datasets, directory)
    if not report.exists():
        evaluation = ["--layouts", args.eval_layouts] if args.eval_layouts else ["--seed", args.eval_seed, "--episodes", args.eval_episodes]
        run_module("humanoid_mujoco_scenes.peg_insert.eval", "--checkpoint", policy_path, "--device", args.device,
                   "--n-action-steps", args.n_action_steps, "--report", report, *evaluation)
    success = json.loads(report.read_text())["success_rate"]
    return checkpoint, policy_path, datasets, {"round": index, "checkpoint": str(checkpoint), "success_rate": success,
                                               "datasets": datasets}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--policy-type", choices=sorted(TRAINERS), default="act")
    p.add_argument("--checkpoint", type=Path, required=True, help="training checkpoint to continue (with training state)")
    p.add_argument("--base-dataset", type=Path, required=True)
    p.add_argument("--extra-dataset", type=Path, action="append", default=[], help="prior correction datasets to keep training on")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--target", type=float, default=1.0, help="stop once unassisted success reaches this rate")
    p.add_argument("--trigger", choices=["approach", "stall", "rim-jam"], default="stall")
    p.add_argument("--corrections", type=int, default=10)
    p.add_argument("--attempts", type=int, default=100, help="policy rollouts per round; seeds never repeat across rounds")
    p.add_argument("--seed", type=int, default=3000, help="first collection seed (keep disjoint from evaluation seeds)")
    p.add_argument("--steps-per-round", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--phase-config", type=Path)
    p.add_argument("--n-action-steps", type=int, default=25)
    p.add_argument("--eval-layouts", type=Path)
    p.add_argument("--eval-seed", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    policy_path = checkpoint / "pretrained_model" if args.policy_type == "act" else checkpoint
    datasets = [str(d.resolve()) for d in [args.base_dataset, *args.extra_dataset]]
    summary = {"policy_type": args.policy_type, "initial_checkpoint": str(checkpoint), "rounds": []}
    for index in range(args.rounds):
        checkpoint, policy_path, datasets, row = run_round(args, index, checkpoint, policy_path, datasets)
        summary["rounds"].append(row)
        write_json(args.root / "summary.json", summary)
        if row["success_rate"] >= args.target:
            break
    print("DAGGER_COMPLETE", json.dumps(summary["rounds"][-1]), flush=True)


if __name__ == "__main__":
    main()
