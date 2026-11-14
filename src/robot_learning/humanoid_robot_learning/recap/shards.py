"""RECAP rollouts in parallel on one machine: K `sim.evaluate --record` processes, then one merged dataset.

    python -m humanoid_robot_learning.recap.shards --out datasets/recap/drawer_stow_r1_rollouts --shards 4 \\
        --episodes 40 --seed 1000 --round 1 --merge -- --scene drawer_stow --policy <run>/pretrained_model

Shard i records episodes with layout seeds seed + i * per_shard ... (per_shard = ceil(episodes / shards)) into
<out>/shard_<i> (log: <out>/shard_<i>.log). Arguments after `--` go to every shard's evaluate (scene, policy,
--max_s, --advantage, ...). --merge then merges the shards and their sidecars into <out>/merged
(recording/merge_takes.py).

Memory: a shard starts only while at least --reserve_gb + --shard_gb stay available, and if available memory drops
below --reserve_gb the newest shard is stopped and no further shard starts (the seeds not recorded are reported;
rerun them with a higher --shard_gb). Measure --shard_gb with one shard
first (`/usr/bin/time -l`, maximum resident set size). On WATcloud use tools/slurm/rollouts.sbatch instead.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

GB = 1e9
POLL_S = 2.0


@dataclass
class Shard:
    index: int
    seed: int
    episodes: int
    root: Path
    proc: subprocess.Popen | None = None
    stopped: bool = False


def available_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / GB


def plan_shards(out: Path, episodes: int, shards: int, seed: int) -> list[Shard]:
    per = math.ceil(episodes / shards)
    plan = []
    for i in range(shards):
        n = min(per, episodes - i * per)
        if n > 0:
            plan.append(Shard(i, seed + i * per, n, out / f"shard_{i}"))
    return plan


def shard_command(shard: Shard, round_: int, evaluate_args: list[str]) -> list[str]:
    return [sys.executable, "-m", "humanoid_robot_learning.sim.evaluate", *evaluate_args, "--episodes",
            str(shard.episodes), "--seed", str(shard.seed), "--record", str(shard.root), "--round", str(round_)]


def start(shard: Shard, cmd: list[str]) -> None:
    log = open(shard.root.parent / f"shard_{shard.index}.log", "w")
    shard.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"[SHARDS] shard {shard.index}: seeds {shard.seed}..{shard.seed + shard.episodes - 1}, pid {shard.proc.pid}",
          flush=True)


def guard_memory(running: list[Shard], reserve_gb: float) -> None:
    """Stop the newest running shard while available memory is under the reserve."""
    live = [s for s in running if s.proc.poll() is None]
    if live and available_gb() < reserve_gb:
        newest = live[-1]
        newest.proc.terminate()
        newest.stopped = True
        print(f"[SHARDS] {available_gb():.1f} GB available < {reserve_gb} GB: stopped shard {newest.index}", flush=True)


def unclaimed_gb(running: list[Shard], shard_gb: float) -> float:
    """Memory the live shards are still expected to take: a shard ramps up over its first seconds (policy, renderers)."""
    import psutil

    total = 0.0
    for s in running:
        if s.proc.poll() is None:
            try:
                total += max(0.0, shard_gb - psutil.Process(s.proc.pid).memory_info().rss / GB)
            except psutil.NoSuchProcess:
                pass
    return total


def run_shards(plan: list[Shard], args, evaluate_args: list[str]) -> None:
    """Run the shards, at most args.parallel at a time and within the memory reserve. Once one is stopped for memory,
    no more start: --shard_gb is too low for this machine."""
    pending, running = list(plan), []
    while pending or any(s.proc.poll() is None for s in running):
        if any(s.stopped for s in running):
            pending = []
        live = sum(s.proc.poll() is None for s in running)
        fits = available_gb() - unclaimed_gb(running, args.shard_gb) >= args.reserve_gb + args.shard_gb
        if pending and not live and not fits:
            raise SystemExit(f"{available_gb():.1f} GB available: no room for a {args.shard_gb} GB shard "
                             f"plus the {args.reserve_gb} GB reserve")
        if pending and live < args.parallel and fits:
            shard = pending.pop(0)
            start(shard, shard_command(shard, args.round, evaluate_args))
            running.append(shard)
        guard_memory(running, args.reserve_gb)
        time.sleep(POLL_S)


def report(plan: list[Shard]) -> list[Shard]:
    """Print each shard's outcome (rerun the seeds of any that did not finish); returns the ones that finished."""
    ok = []
    for s in plan:
        status = "not started" if s.proc is None else "stopped (memory)" if s.stopped else f"exit {s.proc.returncode}"
        print(f"[SHARDS] shard {s.index} (seeds {s.seed}..{s.seed + s.episodes - 1}): {status}", flush=True)
        if s.proc is not None and s.proc.returncode == 0 and not s.stopped:
            ok.append(s)
    return ok


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__.split("\n\n", 2)[2],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                usage="%(prog)s --out DIR [options] -- EVALUATE_ARGS")
    p.add_argument("--out", required=True, type=Path, help="folder of the shard datasets and logs (must not exist)")
    p.add_argument("--episodes", type=int, default=40, help="episodes in all")
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--parallel", type=int, default=None, help="shards at a time (default: all)")
    p.add_argument("--seed", type=int, default=1000, help="layout seed of the first episode")
    p.add_argument("--round", type=int, default=0, help="RECAP round stored with the rollouts")
    p.add_argument("--shard_gb", type=float, default=4.0, help="expected peak memory of one shard (GB)")
    p.add_argument("--reserve_gb", type=float, default=5.0, help="memory that must stay available (GB)")
    p.add_argument("--merge", action="store_true", help="merge the finished shards into <out>/merged")
    args = p.parse_args(argv[:split])
    args.parallel = args.parallel or args.shards
    return args, argv[split + 1:]


def main(argv=None) -> int:
    args, evaluate_args = parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"--out {args.out} already exists; pick a new folder")
    if not evaluate_args:
        raise SystemExit("pass evaluate's arguments after `--` (at least --scene and --policy)")
    args.out.mkdir(parents=True)
    plan = plan_shards(args.out, args.episodes, args.shards, args.seed)
    run_shards(plan, args, evaluate_args)
    ok = report(plan)
    if args.merge and ok:
        from humanoid_robot_learning.recording.merge_takes import main as merge

        merge([str(s.root) for s in ok] + ["--output", str(args.out / "merged")])
    return 0 if len(ok) == len(plan) else 1


if __name__ == "__main__":
    sys.exit(main())
