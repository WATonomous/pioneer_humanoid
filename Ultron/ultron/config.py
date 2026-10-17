"""
Ultron Configuration
Edit this file to configure your Ultron installation.
"""

import os
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
ULTRON_ROOT = Path(__file__).parent.resolve()
LOG_DIR = ULTRON_ROOT / "logs"
JOBS_DIR = ULTRON_ROOT / "jobs"
ENV_FILE = ULTRON_ROOT / ".env"
INFORMATION_DIR = ULTRON_ROOT / "information"
LEARNED_DIR = INFORMATION_DIR / "learned"

# ── SLURM Defaults ──────────────────────────────────────────────────────────
SLURM = {
    "partition": "compute",        # default SLURM partition (1 day limit)
    "long_partition": "compute_dense",  # used automatically when wall time > 24h
    "gpus": 1,                     # number of GPUs
    "cpus_per_task": 4,
    "mem": "16G",
    "time": "3:00:00",             # default wall time if none extracted
    "account": "",                 # leave blank to use default account
    "default_mode": "auto",        # auto | overlap | new
    "min_overlap_time_left": 30,   # minutes a running job must have left to be overlapped when the task states no duration
}

# ── Claude Code (main brain — uses your `claude` login, no API key) ─────────
CLAUDE_CODE = {
    "bin": os.environ.get("ULTRON_CLAUDE_BIN", str(Path.home() / ".local/bin/claude")),
    "models": {
        "plan":     "opus",    # initial detailed plan, at submit time
        "heal":     "haiku",   # first responder for every failed step
        "escalate": "sonnet",  # takes over when haiku can't resolve a step
        "verify":   "haiku",   # end-of-job goal verification
        "learn":    "haiku",   # writes the lessons for information/learned/
    },
    "plan_tools": "Read,Glob,Grep",   # read-only tools for the submit-time planner
    "plan_timeout": 900,              # seconds
    "timeout": 300,                   # seconds per in-job call
    "usage_limit_backoff": 900,       # seconds to stay on Ollama before retrying Claude after a usage limit
    "error_backoff": 120,             # seconds to stay on Ollama after any other Claude failure
}

# ── Local LLM (Ollama — complete backup when Claude is unreachable) ─────────
OLLAMA = {
    "host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    "model": os.environ.get("ULTRON_OLLAMA_MODEL", "qwen2.5-coder:7b"),
    "timeout": 600,                             # seconds per request
}

# ── Self-healing (no attempt limit: stops only on success, wall time, or `ultron --cancel`) ──
HEALING = {
    "haiku_attempts": 2,     # failed attempts on a step before escalating to sonnet
    "replan_after": 6,       # failed attempts on a step before the healer may rewrite the remaining plan
    "notify_every": 5,       # Discord update every N attempts
    "history_full": 4,       # most recent attempts shown to the model in full; older ones are summarised
    "all_backends_down_wait": 60,  # seconds to wait when neither Claude nor Ollama answers
    "stall_replan": 2,       # consecutive fixes that leave the error unchanged before the healer may rewrite the plan
    "progress_cmd_chars": 2500,   # per-step command text shown to the healer (earlier steps' commands, scripts written)
    "progress_chars": 12000,      # total budget for the plan-progress section of the heal prompt
    "ref_file_chars": 4000,       # per-file cap when inlining local files referenced by the failing command
}

# ── Destructive-command blocklist ───────────────────────────────────────────
# Commands matching these are refused by the executor; the refusal is reported
# back to the healer, which must find another route.
SAFETY = {
    # recursive deletes are only allowed under the job workdir and these prefixes
    "rm_safe_prefixes": ["/tmp/ultron"],
    "blocked_patterns": [
        (r"\bdocker\s+(system|volume|image|builder|container|network)\s+prune\b", "docker prune"),
        (r"\bdocker\s+volume\s+rm\b", "docker volume removal"),
        (r"\b(mkfs(\.\w+)?|wipefs|fdisk|parted|sfdisk)\b", "disk formatting / partitioning"),
        (r"\bdd\b[^|;&]*\bof=/dev/", "raw device write"),
        (r">\s*/dev/(sd|nvme|vd|hd)", "raw device write"),
        (r"\bscancel\b", "cancelling SLURM jobs"),
        (r"\bgit\s+push\b[^|;&]*(\s--force\b|\s-f\b|\s--force-with-lease\b|\s\+\S)", "git force push"),
        (r"(\.claude/|\.credentials\.json)", "touching Claude credentials"),
        (r"(>>?|\btee\b(\s+-a)?|\bsed\s+-i\S*|\brm\b|\bmv\b|\bcp\b|\btruncate\b)[^|;&]*\.env\b(?!\.template)", "modifying a .env file"),
        (r"\b(shutdown|reboot|poweroff|halt)\b", "powering off the node"),
        (r"\bchmod\s+-R\s+\S+\s+(/|~|\$HOME)(\s|$)", "recursive chmod on a root or home directory"),
        (r":\(\)\s*\{", "fork bomb"),
    ],
}

# ── Discord Webhook ──────────────────────────────────────────────────────────
DISCORD = {
    "webhook_url_env": "ULTRON_DISCORD_WEBHOOK",  # env var name, set in .env
}
