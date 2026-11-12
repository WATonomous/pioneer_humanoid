"""Evaluate completed ACT checkpoints in headless Isaac while training continues.

Run inside the simulation container using IsaacLab Python. Training and evaluation
are separate processes. No human corrections or autonomous training labels are made.
"""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def main():
    # Run cleanup when the monitor is stopped while its evaluator is active.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--training-pid", type=Path, required=True)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--first-step", type=int, default=2000)
    parser.add_argument("--interval", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-seconds", type=float, default=35.)
    parser.add_argument("--randomization", choices=("baseline", "varied", "wide", "tray"), default="baseline")
    args = parser.parse_args()
    if args.pid_file:
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(str(os.getpid()) + "\n")
    root = Path(__file__).resolve().parents[3]
    run = args.run.resolve()
    output = run / "headless_evaluations"
    output.mkdir(parents=True, exist_ok=True)
    launcher = Path(os.environ.get("ISAACLAB", "/workspace/isaaclab")) / "isaaclab.sh"
    next_step = args.first_step
    best_rank = (-1, -1, -float("inf"))
    if (output / "best.json").exists():
        best_rank = tuple(json.loads((output / "best.json").read_text())["rank"])
    while True:
        log_path = args.training_log or args.training_pid.with_suffix(".log")
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        losses = re.findall(r"loss:([^\s]+)", text)
        if any(value.lower() in {"nan", "inf", "-inf"} for value in losses):
            raise RuntimeError("Non-finite training loss detected; training needs attention")
        try:
            os.kill(int(args.training_pid.read_text().strip()), 0)
            training_alive = True
        except (ProcessLookupError, FileNotFoundError):
            training_alive = False
        checkpoint = run / "checkpoints/last"
        ready = checkpoint.is_symlink() and (checkpoint.resolve() / "pretrained_model/model.safetensors").is_file()
        step = int(checkpoint.resolve().name) if ready else 0
        config = run / "checkpoints/last/pretrained_model/train_config.json"
        total = json.loads(config.read_text())["steps"] if config.is_file() else None
        final = ready and step == total
        (output / "health.json").write_text(json.dumps({"training_alive": training_alive,
            "latest_complete_checkpoint": step, "latest_loss": float(losses[-1]) if losses else None,
            "free_disk_gib": shutil.disk_usage(run).free / 1024**3,
            "updated_unix_time": time.time()}, indent=2) + "\n")
        if ready and (step >= next_step or (not training_alive or final)):
            snapshot = checkpoint.resolve()
            suffix = "" if args.randomization == "baseline" else f"_{args.randomization}"
            report = output / f"step_{step:06d}{suffix}.json"
            if not report.exists() or "success_rate" not in json.loads(report.read_text()):
                command = [str(launcher), "-p", str(root / "src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py"),
                           "--mode", "evaluate", "--headless", "--device", "cuda",
                           "--episodes", str(args.episodes), "--seed", str(args.seed),
                           "--max-seconds", str(args.max_seconds), "--policy", str(snapshot / "pretrained_model"),
                           "--randomization", args.randomization,
                           "--report", str(report), "--debug-images"]
                print(f"[Monitor] Evaluating immutable checkpoint {step}", flush=True)
                eval_log = report.with_suffix(".log")
                with eval_log.open("w") as log:
                    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    try:
                        deadline = time.monotonic() + max(900, args.episodes * 600)
                        while process.poll() is None:
                            if time.monotonic() > deadline or time.time() - eval_log.stat().st_mtime > 180:
                                raise RuntimeError("Headless evaluation stopped making progress; its process group was stopped")
                            time.sleep(5)
                        returncode = process.returncode
                    finally:
                        # Isaac can swallow SIGTERM while its launcher shell exits.
                        # Kill the owned group even if the shell has already ended.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                if returncode != 0:
                    raise RuntimeError(f"Headless evaluation failed: {report.with_suffix('.log')}")
            data = json.loads(report.read_text())
            episodes = data["episodes"]
            rank = (sum(e["success"] for e in episodes), sum(e["ever_lifted"] for e in episodes),
                    -sum(e["minimum_nominal_handle_distance_m"] for e in episodes) / len(episodes))
            if rank > best_rank:
                best_rank = rank
                link = run / "checkpoints/best_eval"
                temporary = link.with_name(".best_eval_next")
                temporary.unlink(missing_ok=True)
                temporary.symlink_to(snapshot.name)
                temporary.replace(link)
                (output / "best.json").write_text(json.dumps({"step": step, "report": str(report),
                    "success_rate": data["success_rate"], "lift_rate": rank[1]/len(episodes), "rank": rank,
                    "checkpoint": str(link / "pretrained_model")}, indent=2) + "\n")
            print(f"[Monitor] step={step} placement={rank[0]}/{len(episodes)} lift={rank[1]}/{len(episodes)}", flush=True)
            next_step = (step // args.interval + 1) * args.interval
            if final or not training_alive:
                if not final:
                    raise RuntimeError("Training stopped before its requested total; last checkpoint evaluated")
                break
        elif not training_alive:
            raise RuntimeError("Training stopped before a complete checkpoint was available")
        free_gb = shutil.disk_usage(run).free / 1024**3
        if free_gb < 3:
            raise RuntimeError(f"Only {free_gb:.1f} GiB disk remaining; training needs attention")
        time.sleep(20)
    print("[Monitor] Training finished; final checkpoint evaluated", flush=True)


if __name__ == "__main__":
    main()
