"""
Ultron job script generator.
Resolves the resources a job needs and generates the script that is either
submitted with sbatch (new allocation) or run with srun --overlap (existing job).
"""

import re
import shlex
from pathlib import Path


def extract_time(task: str, default: str = None) -> str:
    """Try to extract wall time from natural language. Returns `default` when none is stated."""
    # e.g. "3 hours", "90 min", "2h30m"
    m = re.search(r'(\d+)\s*h(?:our)?s?\s*(\d+)\s*m(?:in)?', task, re.I)
    if m:
        return f"{m.group(1)}:{m.group(2):0>2}:00"

    m = re.search(r'(\d+)\s*(?:hours?|hrs?|h)\b', task, re.I)
    if m:
        return f"{m.group(1)}:00:00"

    m = re.search(r'(\d+)\s*(?:minutes?|mins?|m)\b', task, re.I)
    if m:
        minutes = int(m.group(1))
        return f"{minutes//60}:{minutes%60:02d}:00"

    return default


def extract_gpus(task: str, default: int = 1) -> int:
    """Try to extract GPU count from task description."""
    m = re.search(r'(\d+)\s*gpu', task, re.I)
    if m:
        return int(m.group(1))
    return default


def time_to_minutes(t: str) -> int:
    """SLURM time ([D-]HH:MM:SS, MM:SS or MM) to minutes, rounded up. 0 if unparseable."""
    m = re.fullmatch(r'(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?', str(t).strip())
    if not m:
        return 0
    days = int(m.group(1) or 0)
    parts = [int(p) for p in m.groups()[1:] if p is not None]
    if days and len(parts) == 1:        # D-HH
        h, mi, s = parts[0], 0, 0
    elif len(parts) == 3:               # HH:MM:SS
        h, mi, s = parts
    elif len(parts) == 2:               # MM:SS, or D-HH:MM
        h, mi, s = (parts[0], parts[1], 0) if days else (0, parts[0], parts[1])
    else:                               # MM
        h, mi, s = 0, parts[0], 0
    return days * 1440 + h * 60 + mi + (1 if s else 0)


def minutes_to_time(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60:02d}:00"


def resolve_resources(task: str, plan_res: dict = None, overrides: dict = None) -> dict:
    """Merge resources: CLI flags > Claude's plan > what the task text says > config defaults."""
    from config import SLURM
    plan_res = plan_res or {}
    o = {k: v for k, v in (overrides or {}).items() if v is not None}

    is_heavy_task = any(k in task.lower() for k in ["docker", "container", "isaac", "train", "act", "gr00t"])

    def valid_time(t):
        return t if t and time_to_minutes(t) > 0 else None

    def valid_mem(m):
        m = str(m or "").upper().strip()
        return m if re.fullmatch(r'\d+[MG]', m) else None

    def valid_int(v, minimum=0):
        try:
            return int(v) if int(v) >= minimum else None
        except (TypeError, ValueError):
            return None

    time = valid_time(o.get("time")) or valid_time(plan_res.get("time")) or extract_time(task) or SLURM["time"]

    gpus = valid_int(o.get("gpus"))
    if gpus is None:
        gpus = valid_int(plan_res.get("gpus"))
    if gpus is None:
        gpus = extract_gpus(task, SLURM["gpus"])

    cpus = valid_int(o.get("cpus"), 1) or valid_int(plan_res.get("cpus"), 1) or (8 if is_heavy_task else SLURM["cpus_per_task"])
    mem = valid_mem(o.get("mem")) or valid_mem(plan_res.get("mem")) or ("48G" if is_heavy_task else SLURM["mem"])

    tmpdisk = valid_int(plan_res.get("tmpdisk_mb"))
    if tmpdisk is None:
        tmpdisk = 102400 if is_heavy_task else 0

    partition = o.get("partition")
    if not partition:
        partition = SLURM["long_partition"] if time_to_minutes(time) > 1440 else SLURM["partition"]

    return {
        "time": time, "gpus": gpus, "cpus": cpus, "mem": mem, "tmpdisk_mb": tmpdisk,
        "partition": partition, "gpu_type": o.get("gpu_type"), "shard": valid_int(o.get("shard"), 1),
        "node": o.get("node"),
    }


def gres_string(res: dict) -> str:
    items = []
    if res.get("shard"):
        items.append(f"shard:{res['shard']}")            # a VRAM share (MB) instead of whole GPUs
    elif res.get("gpus"):
        items.append(f"gpu:{res['gpu_type']}:{res['gpus']}" if res.get("gpu_type") else f"gpu:{res['gpus']}")
    if res.get("tmpdisk_mb"):
        items.append(f"tmpdisk:{res['tmpdisk_mb']}")
    return ",".join(items)


def generate_script(
    task: str,
    backend: str,
    job_name: str,
    log_path: str,
    spec_path: str,
    ultron_dir: str,
    workdir: str,
    resources: dict,
    mode: str = "new",
) -> str:
    """
    Generate the job script. No secrets are written into it: the agent loads
    ultron/.env itself at startup.
    The #SBATCH header only takes effect in `new` mode (sbatch); under
    srun --overlap the script runs as a plain bash script inside the existing job.
    """
    from config import SLURM, OLLAMA

    account = SLURM.get("account", "")
    header = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resources['partition']}",
    ]
    gres = gres_string(resources)
    if gres:
        header.append(f"#SBATCH --gres={gres}")
    if resources.get("node"):
        header.append(f"#SBATCH --nodelist={resources['node']}")
    header += [
        f"#SBATCH --cpus-per-task={resources['cpus']}",
        f"#SBATCH --mem={resources['mem']}",
        f"#SBATCH --time={resources['time']}",
        f"#SBATCH --output={log_path}.slurm.out",
        f"#SBATCH --error={log_path}.slurm.err",
    ]
    if account:
        header.append(f"#SBATCH --account={account}")
    header_block = "\n".join(header)

    # Ollama is only pre-started when it is the chosen backend. As the backup
    # for Claude it is started on demand by backends.py.
    ollama_block = ""
    if backend == "ollama":
        ollama_block = f"""
# ── Start Ollama server on this node ───────────────────────────────────────────
OLLAMA_BIN=$(command -v ollama || for p in /usr/local/bin/ollama ~/.local/bin/ollama ~/bin/ollama; do [ -x "$p" ] && echo "$p" && break; done)
if [ -n "$OLLAMA_BIN" ]; then
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "[ULTRON] Starting ollama server via $OLLAMA_BIN..."
        $OLLAMA_BIN serve &
        OLLAMA_PID=$!
        sleep 5  # wait for server to be ready
        $OLLAMA_BIN pull {OLLAMA['model']} || echo "[ULTRON] Model pull failed, will try anyway"
        echo "[ULTRON] Ollama PID: $OLLAMA_PID"
    else
        echo "[ULTRON] Ollama server is already running on port 11434."
    fi
else
    echo "[ULTRON] WARNING: ollama binary not found. The ollama backend requires ollama installed."
fi
"""

    # Under srun --overlap this script is a step of the parent job, so the step id tells runs apart.
    job_id_expr = '"${SLURM_JOB_ID}.${SLURM_STEP_ID}"' if mode == "overlap" else '"$SLURM_JOB_ID"'
    # Report the step id back to the CLI, which is waiting for it.
    stepid_line = f'echo {job_id_expr} > {shlex.quote(log_path + ".stepid")}' if mode == "overlap" else ""
    home = Path.home()

    script = f"""#!/bin/bash
{header_block}

{stepid_line}
echo "============================================================"
echo " ULTRON JOB STARTING"
echo " Node     : $(hostname)"
echo " Job ID   : "{job_id_expr}
echo " Mode     : {mode}"
echo " Task     : "{shlex.quote(task)}
echo " Backend  : {backend}"
echo " Time     : $(date -u)"
echo "============================================================"

# ── Environment (secrets are loaded from ultron/.env by the agent, not stored here) ──
export ULTRON_DIR={shlex.quote(ultron_dir)}
export PYTHONPATH={shlex.quote(ultron_dir)}:$PYTHONPATH
export PATH=/usr/local/bin:/usr/bin:{home}/bin:{home}/.local/bin:$PATH

# Load any modules you need (edit as needed)
# module load cuda/12.1
# module load python/3.10

# ── Working directory ──────────────────────────────────────────────────────────
cd {shlex.quote(workdir)} || {{ echo "ERROR: workdir not found: {workdir}"; exit 1; }}
{ollama_block}
# ── Run the Ultron agent ───────────────────────────────────────────────────────
python3 {shlex.quote(ultron_dir + "/agent.py")} \\
    --spec {shlex.quote(spec_path)} \\
    --job-id {job_id_expr}

EXIT_CODE=$?

# ── Cleanup ────────────────────────────────────────────────────────────────────
if [ -n "$OLLAMA_PID" ]; then
    kill "$OLLAMA_PID" 2>/dev/null && echo "[ULTRON] Ollama server stopped"
fi

echo "[ULTRON] Agent exited with code: $EXIT_CODE"
exit $EXIT_CODE
"""
    return script
