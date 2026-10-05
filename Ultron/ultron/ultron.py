#!/usr/bin/env python3
"""
Ultron CLI — fire-and-forget AI task runner on SLURM
Usage:
    ultron "run ACT training for 3 hours"
    ultron "train VLA for 1 hour" -overlap        # run inside your already-running SLURM job
    ultron "train VLA for 1 hour" -new            # take a fresh allocation (new node)
    ultron --dry-run "preprocess data for 30 min" # show the plan and job script, submit nothing
    ultron --list | --status ID | --logs [ID] | --cancel [ID]
"""

import sys
import os
import re
import json
import time
import shutil
import argparse
import subprocess
import datetime
from pathlib import Path

# Ensure ultron package is importable from this script's location
ULTRON_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ULTRON_DIR))

# .env is loaded here, so nothing has to be exported in your shell.
import envfile
envfile.load_env()

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

import config
import jobgen
import notifier
import backends
import planning
import knowledge

console = Console() if HAS_RICH else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _print(msg: str, style: str = ""):
    if HAS_RICH:
        try:
            console.print(msg, style=style or None)
            return
        except Exception:
            pass   # text that isn't valid rich markup (e.g. brackets in a task)
    print(re.sub(r'\[/?[a-z ]+\]', '', msg))


def _header():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]ULTRON[/bold cyan] [dim]— Autonomous SLURM AI Agent[/dim]",
            border_style="cyan",
        ))
    else:
        print("=" * 50)
        print("  ULTRON — Autonomous SLURM AI Agent")
        print("=" * 50)


def _slugify(task: str) -> str:
    """Create a short, filesystem-safe job name from the task."""
    slug = re.sub(r'[^a-zA-Z0-9 ]', '', task.lower())
    words = slug.split()[:4]
    return "_".join(words) or "ultron_job"


def _load_job_index() -> list:
    index_path = config.JOBS_DIR / "index.json"
    if index_path.exists():
        try:
            return json.loads(index_path.read_text())
        except Exception:
            return []
    return []


def _save_job_index(jobs: list):
    index_path = config.JOBS_DIR / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(jobs, indent=2))


def _find_entry(job_id: str = None) -> dict:
    jobs = _load_job_index()
    if not jobs:
        return None
    if not job_id:
        return jobs[-1]
    return next((j for j in reversed(jobs) if str(j.get("slurm_id")) == str(job_id) or j.get("job_name") == job_id), None)


# ── SLURM queries ─────────────────────────────────────────────────────────────

def get_running_jobs() -> list:
    """The current user's RUNNING jobs, best overlap candidate first."""
    try:
        user = os.environ.get("USER", "")
        res = subprocess.run(
            ["squeue", "-u", user, "-t", "RUNNING", "--noheader", "-o", "%i|%j|%L|%b|%N|%P"],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            return []
    except Exception:
        return []

    jobs = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 6 or not parts[0].isdigit():
            continue   # skips array/het job ids, which srun --overlap can't target by plain id
        job_id, name, left, gres, node, partition = parts
        jobs.append({
            "id": job_id, "name": name, "time_left": left, "time_left_min": jobgen.time_to_minutes(left),
            "gres": gres, "node": node, "partition": partition,
            "has_gpu": bool(re.search(r"gpu|shard", gres)),
            "is_ultron": name.startswith("ultron_"),
        })
    # prefer jobs with a GPU, then your own (non-Ultron) sessions, then the most time left
    jobs.sort(key=lambda j: (j["has_gpu"], not j["is_ultron"], j["time_left_min"]), reverse=True)
    return jobs


def get_job_gres(job_id: str) -> str:
    """Extract allocated GRES for a given job ID from scontrol."""
    try:
        res = subprocess.run(["scontrol", "show", "job", str(job_id)], capture_output=True, text=True)
        if res.returncode == 0:
            m = re.search(r'TresPerNode=([^\s]+)', res.stdout)
            if m:
                raw_tres = m.group(1)
                items = [item.replace("gres/", "") for item in raw_tres.split(",") if item]
                return ",".join(items)
            m2 = re.search(r'Gres=([^\s]+)', res.stdout)
            if m2 and m2.group(1).lower() != "(null)":
                return m2.group(1)
    except Exception:
        pass
    return ""


def find_step_id(log_path: str, proc, wait: int = 30) -> str:
    """After `srun --overlap`, read the step id the job script reports, so the run can be tracked and cancelled on its own."""
    marker = Path(f"{log_path}.stepid")
    deadline = time.time() + wait
    while time.time() < deadline:
        if marker.exists():
            sid = marker.read_text().strip()
            if re.fullmatch(r"\d+\.\d+", sid):
                return sid
        if proc.poll() is not None and not marker.exists():
            break
        time.sleep(0.5)
    return ""


# ── Mode selection: overlap an existing job, or take a new allocation ─────────

def choose_mode(mode: str, overlap_job: str, needed_min: int, stated: bool) -> dict:
    """
    Returns the overlap target job dict, or None for a new sbatch allocation.
    needed_min: wall time the task asks for (or the configured minimum when the task states none).
    """
    if mode == "new":
        _print("\n[cyan]Mode: NEW — starting a fresh SLURM allocation via sbatch.[/cyan]")
        return None

    running = get_running_jobs()
    need_txt = f"{needed_min} min" + ("" if stated else " (minimum; the task states no duration)")

    if mode == "overlap":
        if overlap_job:
            target = next((j for j in running if j["id"] == str(overlap_job)), None)
            if not target:
                _print(f"\n[red]ERROR: job {overlap_job} is not one of your running jobs.[/red]")
                sys.exit(1)
        elif running:
            target = running[0]
        else:
            _print("\n[red]ERROR: no running SLURM job to overlap. Use -new (or --mode auto) for a fresh allocation.[/red]")
            sys.exit(1)
        _print(f"\n[cyan]Mode: OVERLAP — running inside job {target['id']} ({target['name']}) on {target['node']}, "
               f"{target['time_left']} left.[/cyan]")
        if target["time_left_min"] < needed_min:
            _print(f"[yellow]Warning: the task wants {need_txt} but the job only has {target['time_left']} left. "
                   f"The run will be cut off when that job ends.[/yellow]")
        return target

    # auto
    suitable = [j for j in running if j["time_left_min"] >= needed_min]
    if suitable:
        target = suitable[0]
        _print(f"\n[cyan]Mode: AUTO → OVERLAP — job {target['id']} ({target['name']}) on {target['node']} has "
               f"{target['time_left']} left, enough for {need_txt}.[/cyan]")
        return target
    if running:
        best = max(running, key=lambda j: j["time_left_min"])
        _print(f"\n[cyan]Mode: AUTO → NEW — your running job {best['id']} has only {best['time_left']} left, "
               f"not enough for {need_txt}. Starting a fresh allocation via sbatch.[/cyan]")
    else:
        _print("\n[cyan]Mode: AUTO → NEW — you have no running SLURM job to overlap. Starting a fresh allocation via sbatch.[/cyan]")
    return None


# ── Submit-time planning with Claude Code ─────────────────────────────────────

def make_plan(task: str, ctx: dict, workdir: str) -> dict:
    """Claude Code (read-only tools) inspects the workdir and writes the detailed plan. Exits if Claude is unavailable."""
    model = config.CLAUDE_CODE["models"]["plan"]
    _print(f"\n[cyan]Planning with Claude Code ({model}) — reading {workdir} to ground the plan. This can take a few minutes...[/cyan]")
    base_prompt = planning.build_plan_prompt(task, ctx, with_tools=True)
    last_err = None
    for attempt in range(1, 3):
        prompt = base_prompt
        if last_err:
            prompt += (f"\n\nATTENTION: your previous reply could not be parsed ({last_err}). "
                       f"Reply with ONLY the JSON object, no markdown fences, no commentary.")
        try:
            raw = backends.claude_code_chat(
                prompt, model=model, tools=config.CLAUDE_CODE["plan_tools"], cwd=workdir,
                add_dirs=[config.INFORMATION_DIR], timeout=config.CLAUDE_CODE["plan_timeout"],
            )
        except backends.ClaudeUnavailable as e:
            _print(f"\n[red]ERROR: Claude Code could not produce the plan: {e}[/red]")
            _print("Check that `claude` is installed and logged in (run `claude` once and use /login).\n"
                   "To run fully on the local backup model instead, add:  --backend ollama")
            sys.exit(1)
        try:
            steps, resources = planning.parse_plan(raw)
            return {"model": model, "task": task, "resources": resources, "steps": steps}
        except Exception as e:
            last_err = e
            _print(f"[yellow]Plan reply could not be parsed ({e}); asking again...[/yellow]")
    _print(f"\n[red]ERROR: Claude Code did not return a valid plan: {last_err}[/red]")
    sys.exit(1)


def print_plan(plan: dict):
    _print(f"\n[bold]Plan ({len(plan['steps'])} steps, by {plan['model']}):[/bold]")
    for i, s in enumerate(plan["steps"], 1):
        detail = s.get("cmd") or s.get("message") or ("<python>" if s.get("code") else "")
        detail = knowledge.sanitize(" ".join(str(detail).split()))
        print(f"  {i:2d}. [{s.get('type', '?')}] {s.get('name', '')}: {detail[:140]}")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_submit(task: str, backend: str, dry_run: bool, workdir: str, mode: str,
               overlap_job: str = None, mock: bool = False, overrides: dict = None):
    _header()
    overrides = overrides or {}

    log_dir, jobs_dir = config.LOG_DIR, config.JOBS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    ts        = _now()
    job_name  = f"ultron_{_slugify(task)}_{ts}"
    log_path  = str(log_dir / f"{job_name}.log")
    job_file  = jobs_dir / f"{job_name}.sh"
    spec_file = jobs_dir / f"{job_name}.json"
    plan_file = jobs_dir / f"{job_name}.plan.json"

    _print(f"\n[bold]Task:[/bold]    {knowledge.sanitize(task)}")
    _print(f"[bold]Backend:[/bold] {'mock' if mock else backend}")
    _print(f"[bold]Workdir:[/bold] {workdir}")

    # 1. Overlap an existing job, or take a new allocation?
    stated_time = overrides.get("time") or jobgen.extract_time(task)
    needed_min = jobgen.time_to_minutes(stated_time) if stated_time else config.SLURM["min_overlap_time_left"]
    target = choose_mode(mode, overlap_job, needed_min, stated=bool(stated_time))
    run_mode = "overlap" if target else "new"

    if not dry_run:
        required_cmd = "srun" if target else "sbatch"
        if not shutil.which(required_cmd):
            _print(f"\n[red]ERROR: {required_cmd} not found in PATH.[/red]")
            sys.exit(1)

    # 2. Plan with Claude Code, told where the job will run
    context = {"mode": run_mode, "workdir": workdir}
    if target:
        context.update(parent_job=target["id"], node=target["node"], gres=target["gres"], time_left=target["time_left"])
    else:
        context.update(
            partition=overrides.get("partition") or config.SLURM["partition"],
            node=overrides.get("node"), gpu_type=overrides.get("gpu_type"),
        )

    plan = None
    if not mock and backend == "claude":
        plan = make_plan(task, context, workdir)
        print_plan(plan)
    elif not mock:
        _print("\n[yellow]Backend is ollama: the plan will be generated inside the job by the local model.[/yellow]")

    # 3. Resources: CLI flags > Claude's plan > task text > defaults
    resources = jobgen.resolve_resources(task, plan["resources"] if plan else None, overrides)
    if target:
        cap = max(1, target["time_left_min"] - 1)
        if jobgen.time_to_minutes(resources["time"]) > cap:
            _print(f"[yellow]Planned wall time {resources['time']} exceeds the {target['time_left']} left on job "
                   f"{target['id']}; capping this run to {jobgen.minutes_to_time(cap)}.[/yellow]")
            resources["time"] = jobgen.minutes_to_time(cap)
        _print(f"[bold]Resources:[/bold] shares job {target['id']} ({target['gres'] or 'no GRES'}), step time limit {resources['time']}")
    else:
        _print(f"[bold]Resources:[/bold] partition={resources['partition']} gres={jobgen.gres_string(resources) or 'none'} "
               f"cpus={resources['cpus']} mem={resources['mem']} time={resources['time']}"
               + (f" node={resources['node']}" if resources.get("node") else ""))

    # 4. Job spec + script (no secrets in either: the agent loads .env itself)
    spec = {
        "task": task, "backend": backend, "mock": mock, "mode": run_mode,
        "job_name": job_name, "log_path": log_path, "workdir": workdir,
        "plan_file": str(plan_file) if plan else None,
        "resources": resources, "context": context,
    }
    script = jobgen.generate_script(
        task=knowledge.sanitize(task), backend=backend, job_name=job_name, log_path=log_path,
        spec_path=str(spec_file), ultron_dir=str(ULTRON_DIR), workdir=workdir,
        resources=resources, mode=run_mode,
    )

    target_gres = get_job_gres(target["id"]) if target else ""
    step_minutes = jobgen.time_to_minutes(resources["time"])
    srun_cmd = []
    if target:
        srun_cmd = ["srun", "--overlap", f"--jobid={target['id']}", f"--job-name={job_name}", f"--time={step_minutes}"]
        if target_gres:
            srun_cmd.append(f"--gres={target_gres}")
        srun_cmd += ["bash", str(job_file)]

    if dry_run:
        preview = " ".join(srun_cmd) if target else f"sbatch {job_file}"
        _print(f"\n[yellow]--- DRY RUN (nothing written or submitted). Would run: {preview} ---[/yellow]")
        print(script)
        return

    if plan:
        plan_file.write_text(json.dumps(plan, indent=2))
    spec_file.write_text(json.dumps(spec, indent=2))
    job_file.write_text(script)
    job_file.chmod(0o755)
    _print(f"\n[green]Job script written:[/green] {job_file}")

    # 5. Submit / Run
    pid = None
    if target:
        if target_gres:
            _print(f"[cyan]Inheriting job GRES: {target_gres}[/cyan]")
        out_f = open(f"{log_path}.slurm.out", "w")
        err_f = open(f"{log_path}.slurm.err", "w")
        proc = subprocess.Popen(srun_cmd, stdin=subprocess.DEVNULL, stdout=out_f, stderr=err_f, start_new_session=True)
        pid = proc.pid
        slurm_id = find_step_id(log_path, proc)
        if not slurm_id:
            if proc.poll() is not None:
                _print(f"\n[red]srun failed to start the step (exit code {proc.returncode}). See {log_path}.slurm.err[/red]")
                sys.exit(1)
            slurm_id = target["id"]
            _print("[yellow]Could not determine the step id; tracking this run under the parent job id.[/yellow]")
        _print(f"\n[bold green]✓ Task running inside existing job {target['id']} via --overlap![/bold green] "
               f"Step: {slurm_id} (srun PID: {pid})")
    else:
        result = subprocess.run(["sbatch", str(job_file)], capture_output=True, text=True)
        if result.returncode != 0:
            _print(f"\n[red]sbatch failed:[/red]\n{result.stderr}")
            sys.exit(1)
        # Parse job ID from "Submitted batch job 12345"
        m = re.search(r'(\d+)', result.stdout)
        slurm_id = m.group(1) if m else "unknown"
        _print(f"\n[bold green]✓ Job submitted![/bold green] SLURM ID: {slurm_id}")
    _print(f"  Log: {log_path}")

    # Save to index
    jobs = _load_job_index()
    jobs.append({
        "slurm_id": slurm_id,
        "job_name": job_name,
        "task": knowledge.sanitize(task),
        "backend": backend,
        "mode": run_mode,
        "parent_job": target["id"] if target else None,
        "pid": pid,
        "submitted": ts,
        "log": log_path,
        "script": str(job_file),
        "spec": str(spec_file),
        "plan": str(plan_file) if plan else None,
        "workdir": workdir,
    })
    _save_job_index(jobs)

    # Discord notification
    notifier.send(
        status="started",
        title="Ultron Job Queued",
        description=knowledge.sanitize(task),
        job_id=slurm_id,
        fields=[
            {"name": "Backend",  "value": backend,   "inline": True},
            {"name": "Mode",     "value": run_mode,  "inline": True},
            {"name": "SLURM ID", "value": slurm_id,  "inline": True},
            {"name": "Log",      "value": log_path,  "inline": False},
        ],
    )

    _print(f"\n[dim]Follow it with `ultron --logs {slurm_id} -f`; stop it with `ultron --cancel {slurm_id}`.[/dim]")


def cmd_list():
    _header()
    jobs = _load_job_index()
    if not jobs:
        _print("\nNo jobs submitted yet.")
        return

    if HAS_RICH:
        table = Table(title="Ultron Jobs", border_style="cyan")
        table.add_column("SLURM ID", style="bold cyan", no_wrap=True)
        table.add_column("Mode",     style="yellow",    no_wrap=True)
        table.add_column("Task",     style="white",     max_width=50)
        table.add_column("Backend",  style="magenta",   no_wrap=True)
        table.add_column("Submitted",style="dim",       no_wrap=True)
        table.add_column("Log",      style="dim green", max_width=40)
        for j in reversed(jobs[-20:]):  # show last 20
            table.add_row(
                str(j.get("slurm_id", "?")),
                j.get("mode", "-"),
                j.get("task", "")[:50],
                j.get("backend", "?"),
                j.get("submitted", "?"),
                j.get("log", "?"),
            )
        console.print(table)
    else:
        for j in reversed(jobs[-20:]):
            print(f"[{j.get('slurm_id')}] ({j.get('mode', '-')}) {j.get('task','')[:50]} ({j.get('backend')}) @ {j.get('submitted')}")


def cmd_status(job_id: str):
    """Check SLURM job (or overlap step) status via sacct."""
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobID,JobName%40,State,Elapsed,ExitCode", "--noheader"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        _print(f"\nStatus for job {job_id}:\n{result.stdout}")
    else:
        _print(f"\nCould not get status for job {job_id}.", "yellow")


def cmd_cancel(job_id: str = None):
    """Cancel a run that Ultron started (a whole sbatch job, or just the overlap step)."""
    entry = _find_entry(job_id)
    if not entry:
        _print(f"Job {job_id or '(latest)'} not found in Ultron's index. Ultron only cancels runs it started.", "red")
        return
    slurm_id = str(entry.get("slurm_id", ""))
    if entry.get("mode") == "overlap" and "." not in slurm_id:
        # step id unknown: cancelling the id would kill the whole parent job, so stop only our srun
        pid = entry.get("pid")
        try:
            os.killpg(os.getpgid(pid), 15)
            _print(f"Stopped overlap run (srun PID {pid}) inside job {slurm_id}.", "green")
        except Exception as e:
            _print(f"Could not stop overlap run (srun PID {pid}): {e}", "red")
        return
    result = subprocess.run(["scancel", slurm_id], capture_output=True, text=True)
    if result.returncode == 0:
        _print(f"Cancelled {slurm_id}: {entry.get('task', '')[:80]}", "green")
    else:
        _print(f"scancel {slurm_id} failed: {result.stderr.strip()}", "red")


def cmd_logs(job_id: str = None, follow: bool = False):
    """Tail/print the log for a given SLURM job ID (or the latest job)."""
    if not _load_job_index():
        _print("No jobs found.")
        return

    entry = _find_entry(job_id)
    if not entry:
        _print(f"Job {job_id} not found in index.", "red")
        return

    log = entry.get("log", "")
    if not log or not Path(log).exists():
        _print(f"Log file not found: {log}", "red")
        return

    if follow:
        try:
            subprocess.run(["tail", "-n", "50", "-f", str(log)])
        except KeyboardInterrupt:
            pass
    else:
        print(Path(log).read_text())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="ultron",
        description="Ultron — fire-and-forget AI task runner on SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task", nargs="?", help="Natural language task to run")
    parser.add_argument(
        "--backend", "-b",
        default="claude",
        choices=list(backends.BACKENDS),
        help="AI backend (default: claude = Claude Code with local Ollama as backup; ollama = local model only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan and print the job script, but don't submit")
    parser.add_argument("--mock",    action="store_true", help="Run with mock plan (instant verification without LLM)")
    parser.add_argument("--list",    action="store_true", help="List all submitted jobs")
    parser.add_argument("--status",  metavar="JOB_ID",   help="Check SLURM job status")
    parser.add_argument("--logs",    metavar="JOB_ID",   nargs="?", const="latest",
                        help="Print log for a job (omit ID for latest)")
    parser.add_argument("--cancel",  metavar="JOB_ID",   nargs="?", const="latest",
                        help="Cancel a run Ultron started (omit ID for latest)")
    parser.add_argument("-f", "--follow", action="store_true",
                        help="Follow/stream log output in real-time (like tail -f)")
    parser.add_argument("--workdir", "-w", default=os.getcwd(),
                        help="Working directory for the job (default: cwd)")

    where = parser.add_argument_group("where to run")
    where.add_argument("--mode", choices=["auto", "overlap", "new"], default=None,
                       help=f"overlap = run inside an already-running job; new = fresh sbatch allocation; "
                            f"auto = overlap if a running job has enough time left, else new (default: {config.SLURM['default_mode']})")
    where.add_argument("-overlap", "--overlap", nargs="?", const="auto", default=None, metavar="JOB_ID",
                       help="Shorthand for --mode overlap (omit JOB_ID to pick your best running job)")
    where.add_argument("-new", "--new", "--no-overlap", dest="new_job", action="store_true",
                       help="Shorthand for --mode new")

    res = parser.add_argument_group("resources for a new allocation (override Claude's plan)")
    res.add_argument("--partition", help="SLURM partition")
    res.add_argument("--time",      help="Wall time, e.g. 3:00:00 (also caps an overlap run)")
    res.add_argument("--gpus",      type=int, help="GPU count")
    res.add_argument("--gpu-type",  dest="gpu_type", help="GPU model, e.g. rtx_3090")
    res.add_argument("--shard",     type=int, help="Request a VRAM share in MB instead of whole GPUs, e.g. 24000")
    res.add_argument("--cpus",      type=int, help="CPUs per task")
    res.add_argument("--mem",       help="Memory, e.g. 48G")
    res.add_argument("--node",      help="Run on this specific node")

    args = parser.parse_args()

    if args.list:
        cmd_list()
        return

    if args.status:
        cmd_status(args.status)
        return

    if args.cancel is not None:
        cmd_cancel(None if args.cancel == "latest" else args.cancel)
        return

    if args.logs is not None or args.follow:
        jid = args.logs if args.logs and args.logs != "latest" else None
        cmd_logs(jid, follow=args.follow)
        return

    if not args.task:
        parser.print_help()
        sys.exit(0)

    task = args.task.strip()
    mode = args.mode or config.SLURM["default_mode"]
    overlap_job = None
    if args.new_job:
        mode = "new"
    elif args.overlap:
        mode = "overlap"
        overlap_job = None if args.overlap == "auto" else args.overlap

    # The mode can also be given as a suffix of the prompt: "... -overlap" / "... -new"
    if re.search(r'[\s\-]+overlap$', task, re.IGNORECASE):
        task = re.sub(r'[\s\-]+overlap$', '', task, flags=re.IGNORECASE).strip()
        mode = "overlap"
    elif re.search(r'\s-+new$', task, re.IGNORECASE):
        task = re.sub(r'\s-+new$', '', task, flags=re.IGNORECASE).strip()
        mode = "new"

    cmd_submit(
        task=task,
        backend=args.backend,
        dry_run=args.dry_run,
        workdir=os.path.abspath(args.workdir),
        mode=mode,
        overlap_job=overlap_job,
        mock=args.mock,
        overrides={
            "partition": args.partition, "time": args.time, "gpus": args.gpus, "gpu_type": args.gpu_type,
            "shard": args.shard, "cpus": args.cpus, "mem": args.mem, "node": args.node,
        },
    )


if __name__ == "__main__":
    main()
