"""Finish the in-flight randomized collection, then run one bounded fine-tune.

Run inside simulation_isaac. This does not tune hyperparameters or regenerate
failed learned-policy rollouts as expert data. It watches an existing generator.
"""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--cap-existing-generator", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    lab = Path(os.environ.get("ISAACLAB", "/workspace/isaaclab")) / "isaaclab.sh"
    data = root / "datasets/nasa_cargo/randomized_v2/new_demos"
    aggregate = root / "datasets/nasa_cargo/randomized_v2/aggregate"
    run = root / "outputs/train/nasa_cargo_act_randomized_v2"
    logs = root / "outputs/train_logs"
    generation_log = logs / "cargo_randomized_generation.log"
    if run.exists() or aggregate.exists():
        raise FileExistsError("Use new output paths for another randomized update")
    while True:
        text = generation_log.read_text(errors="replace") if generation_log.exists() else ""
        info = data / "meta/info.json"
        manifest = data / "collection_manifest.jsonl"
        if args.cap_existing_generator and info.exists() and manifest.exists():
            count = json.loads(info.read_text())["total_episodes"]
            rows = manifest.read_text().splitlines()
            if count >= args.episodes and len(rows) == count:
                if count != args.episodes:
                    raise RuntimeError("Generator passed the cap; retain its data and inspect before proceeding")
                # Every episode is finalized before its manifest row is written.
                # Stop only this generator's Python process; pending frames are
                # an unsaved in-memory episode and are deliberately discarded.
                for entry in Path("/proc").iterdir():
                    if not entry.name.isdigit():
                        continue
                    try:
                        command = (entry / "cmdline").read_bytes().split(b"\0")
                    except OSError:
                        continue
                    if (command and b"python3" in command[0]
                            and any(a.endswith(b"/hg_dagger.py") for a in command)
                            and b"generate" in command
                            and any(a.endswith(b"randomized_v2/new_demos") for a in command)):
                        pid = int(entry.name)
                        os.kill(pid, signal.SIGINT)
                        time.sleep(1)
                        if (entry / "cmdline").exists() and (entry / "cmdline").read_bytes():
                            os.kill(pid, signal.SIGKILL)
                print(f"[Update] Capped collection at {count} finalized randomized demos", flush=True)
                break
        if "CARGO_GENERATION_COMPLETE" in text:
            break
        if "Traceback (most recent call last)" in text:
            raise RuntimeError(f"Generation failed; see {generation_log}")
        if generation_log.exists() and time.time() - generation_log.stat().st_mtime > 600:
            raise RuntimeError("Generator stopped making progress; inspect its log")
        time.sleep(10)
    assert json.loads((data / "meta/info.json").read_text())["total_episodes"] == args.episodes
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src/robot_learning") + ":" + env.get("PYTHONPATH", "")
    print("[Update] Validating randomized demonstrations", flush=True)
    subprocess.run([str(lab), "-p", "-m", "humanoid_robot_learning.sim.cargo_data", "validate", str(data),
                    "--episodes", str(args.episodes)], env=env, check=True)
    subprocess.run([str(lab), "-p", "-m", "humanoid_robot_learning.sim.cargo_data", "merge",
                    str(root / "datasets/nasa_cargo/synthetic_50"), str(data), "--output", str(aggregate)],
                   env=env, check=True)
    env.update(CARGO_DATASET_ROOT=str(aggregate), CARGO_OUTPUT_DIR=str(run), CARGO_STEPS=str(args.steps))
    pid_file = logs / "cargo_randomized_v2.pid"
    print(f"[Update] Starting one {args.steps}-step fine-tune", flush=True)
    with (logs / "cargo_randomized_v2.log").open("w") as training_log:
        process = subprocess.Popen(["bash", "src/robot_learning/scripts/cargo_finetune_quick.sh"],
                                   env=env, stdout=training_log, stderr=subprocess.STDOUT, start_new_session=True)
        pid_file.write_text(str(process.pid) + "\n")
        # Do not let the watcher create the output before trainer validation.
        while not (run / "checkpoints/last").is_symlink():
            if process.poll() is not None:
                raise RuntimeError("Fine-tuning stopped before its first checkpoint; inspect training log")
            time.sleep(5)
        with (logs / "cargo_randomized_watch.log").open("w") as monitor_log:
            monitor = subprocess.Popen([str(lab), "-p", "src/robot_learning/scripts/cargo_watch_training.py",
                "--run", str(run), "--training-pid", str(pid_file), "--first-step", "2000",
                "--interval", "10000", "--episodes", "1", "--seed", "60000",
                "--randomization", "varied", "--pid-file", str(logs / "cargo_randomized_watch.pid")],
                env=env, stdout=monitor_log, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait()
            if code:
                raise RuntimeError(f"Fine-tuning exited with code {code}")
            if monitor.wait():
                raise RuntimeError("Fine-tuning completed but its sanity evaluation failed to run")
    print(f"CARGO_RANDOMIZED_UPDATE_COMPLETE run={run}", flush=True)


if __name__ == "__main__":
    main()
