# Ultron

**Autonomous AI task runner on SLURM** — type what you want, it figures out how to do it.

```
ultron "run ACT training for 3 hours"
ultron "train VLA for 1 hour" -overlap
ultron --dry-run "preprocess imitation learning data for 30 min"
```

---

## How it works

```
  You type:  ultron "run ACT training for 3 hours"
                    │
                    ▼
          ┌──────────────────┐
          │   ultron CLI      │  1. picks overlap vs new allocation
          │                  │  2. Claude Code (opus) reads the workdir and writes the plan
          └────────┬─────────┘
                   │  srun --overlap  (inside your running job)
                   │  or sbatch       (fresh allocation / new node)
                   ▼
          ┌──────────────────┐
          │  Ultron Agent     │  executes the plan step by step
          │                  │  failed step → self-heal, no attempt limit:
          │                  │     haiku first → sonnet when haiku can't fix it
          │                  │  (local Ollama if Claude is unreachable)
          └────────┬─────────┘
                   │  goal verified → what worked is saved to information/learned/
                   ▼
          ┌──────────────────┐
          │  Full audit log  │  every step, timestamp, output  (+ Discord)
          └──────────────────┘
```

Ultron uses **Claude Code in headless mode** (`claude -p`) with your normal Claude login.
No API key is used and nothing is billed to the API; calls count against your Claude plan's usage.

---

## Setup

### 1. Install
```bash
cd ultron/
bash install.sh
```
This creates `~/bin/ultron`. Add `~/bin` to your PATH if needed:
```bash
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

### 2. Log in to Claude Code (once)
```bash
claude        # then /login
```
The login lives in `~/.claude`, which is on the shared home directory, so every compute node sees it.

### 3. Configure
```bash
vim ultron/config.py   # SLURM defaults, which model does what, healing and safety settings
vim ultron/.env        # Discord webhook, HF token
```
Ultron reads `ultron/.env` itself every time it runs — in the CLI and inside jobs. You do not need to
export anything in your shell, and secrets are never written into job scripts.

**Discord notifications:** Server Settings → Integrations → Webhooks → New Webhook, then set
`ULTRON_DISCORD_WEBHOOK=<url>` in `ultron/.env`.

---

## Usage

```bash
# Default: auto mode, planned by Claude
ultron "run ACT training for 3 hours"

# See the plan and the job script without submitting anything
ultron --dry-run "run ACT training for 3 hours"

# Where to run
ultron "train VLA for 1 hour" -overlap          # inside your best running SLURM job
ultron "train VLA for 1 hour" -overlap 682119   # inside a specific running job
ultron "train VLA for 1 hour" -new              # fresh allocation via sbatch
ultron "train VLA for 1 hour -overlap"          # the mode also works as a prompt suffix
ultron --mode auto "train VLA for 1 hour"       # overlap if a job has enough time left, else new

# Resources for a new allocation (these override Claude's plan)
ultron -new --gpus 2 --gpu-type rtx_3090 --mem 64G --cpus 16 --time 6:00:00 "distributed training run"
ultron -new --shard 24000 "evaluate the checkpoint"       # a 24 GB VRAM share instead of a whole GPU
ultron -new --node thor-slurm1 "rebuild the container"

# Manage runs
ultron --list
ultron --status 682185
ultron --logs            # latest run;  add -f to follow
ultron --logs 682119.2
ultron --cancel 682119.2 # stops only that run (an overlap step, or a whole sbatch job)

# Run entirely on the local backup model
ultron --backend ollama "check the GPU"
```

### Overlap vs new

| | Overlap | New |
|---|---|---|
| How | `srun --overlap` step inside a running job | `sbatch`, fresh allocation |
| Node state | Reuses the running dockerd and containers | Fresh node: the plan starts dockerd and builds/pulls |
| Limits | The parent job's time left and GPUs | Partition limits (`compute_dense` is chosen automatically above 24 h) |
| ID | `<job>.<step>`, e.g. `682119.2` | `<job>` |

`auto` (the default) overlaps when one of your running jobs has enough time left for the task, otherwise
it submits a new job, and prints which it chose and why. The planner is told the mode, so plans for a new
node include container setup and plans for an overlap reuse what is already running.

---

## Models

| Role | Model | When |
|---|---|---|
| Initial plan | `opus` | Every submit, with read-only tools on the working directory |
| Healing | `haiku` | First responder for every failed step |
| Healing escalation | `sonnet` | After 2 failed haiku attempts, a repeated fix, or haiku asking for help |
| Goal verification | `haiku` | End of job (`sonnet` if haiku can't give a verdict) |
| Lessons | `haiku` | After a verified success |
| Backup | Ollama `qwen2.5-coder:7b` | Whenever Claude is unreachable or over its usage limit; Claude is retried after a backoff |

Change them in `CLAUDE_CODE["models"]` in `config.py`.

## Self-healing

There is **no attempt limit**. A failed step is retried with new fixes until it works. The only stops are
the SLURM wall time and `ultron --cancel`. Each attempt sees the history of earlier attempts; a fix
identical to one already tried is rejected without running. After 6 failed attempts the healer may rewrite
the rest of the plan. Discord gets an update every 5 attempts so you can cancel a loop that isn't going anywhere.

A task that can never succeed will therefore hold its allocation until the wall time runs out.

## Safety blocklist

The executor refuses these, in plan steps and in fixes, and tells the healer to find another way:
recursive `rm` outside the job's working directory (and `/tmp/ultron*`), `docker ... prune` and
`docker volume rm`, disk formatting and raw device writes, `scancel`, git force pushes, anything touching
`.claude/`, modifying a `.env` file, powering off the node. Edit `SAFETY` in `config.py` to change it —
for example add `/workspace` to `rm_safe_prefixes` to allow cleanup inside the container's workspace.

## Learning

After a job is verified successful, Ultron writes what worked to `information/learned/` so later runs
(and other Ultron instances on the same filesystem) reuse it. See `information/README.md`.

---

## File layout

```
ultron/
├── ultron.py        ← CLI entrypoint (this is what `ultron` runs): mode choice, planning, submit
├── agent.py         ← Compute-side agent: executes the plan, self-heals, verifies, learns
├── backends.py      ← Claude Code (headless) + Ollama backup
├── planning.py      ← Planner prompt and plan parsing
├── knowledge.py     ← information/ library: reading, learned entries, secret scrubbing
├── safety.py        ← Destructive-command blocklist
├── jobgen.py        ← Resource resolution and job script generator
├── notifier.py      ← Discord webhook notifications
├── envfile.py       ← Loads .env
├── config.py        ← SLURM defaults, models, healing, safety
├── install.sh       ← Installs `~/bin/ultron`
├── .env.template    ← Copy to .env
├── information/     ← Reference guides + learned/ (auto-written)
├── logs/            ← Full audit logs per job
└── jobs/
    ├── index.json       ← All submitted jobs
    ├── *.sh             ← Generated job scripts (no secrets)
    ├── *.json           ← Job specs
    ├── *.plan.json      ← Claude's plans
    └── archive/         ← Retired pre-Ultron scripts (secrets redacted)
```

---

## Discord Notifications

Every job sends embeds for:
- 🚀 **Job queued** (on `ultron` submit)
- 🚀 **Job started** (when the compute node picks it up)
- ⚙️  **Execution plan** (as JSON)
- ⚙️  **Each step** (name, shell output, success/fail)
- ⚠️  **Still self-healing** (every 5 attempts) and **plan rewritten**
- ✅ / ❌ **Job complete/failed** (summary)
- 📄 **Full log dump** (chunked if large)

Set `ULTRON_DISCORD_DISABLE=1` in your shell to silence them for a test run.
