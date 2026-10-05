# TestVLA Policy — Run it in Isaac Lab

Quick guide to running the trained SmolVLA policy in simulation. No headaches with Python version conflicts (we handle the 3.10 vs 3.12 thing for you).

## TL;DR — One Command

From the humanoid repo root:

```bash
../run_testvla_sim.sh 3          # Run 3 episodes
../run_testvla_sim.sh             # Default: 1 episode
```

Done. Sit back, watch the robot try to pick up a vial and put it in a rack. Results print at the end.

## What You Need

- Docker + the `isaac-lab-ros2` container running
  ```bash
  cd ~/IsaacLab && ./docker/container.py start ros2
  ```
- HF_TOKEN in `Ultron/ultron/.env` (for the private model repo)
- ~30 min on the first run (RTX shader compilation), then ~2 min per episode

## How It Works (The Boring Details)

**The Problem:** The trained policy (SmolVLA, needs Python 3.12) can't live inside Isaac Sim (Python 3.10). They hate each other.

**The Solution:** We run the policy in its own process (Python 3.12 venv) and Isaac talks to it over a socket. Pickle + struct for serialization. Dead simple.

```
                   Container
    ┌─────────────────────────────────────┐
    │                                     │
    │  Isaac Sim (Python 3.10)            │  Sends observations
    │  lerobot_eval_rtc_remote.py         │─────────────────┐
    │                                     │                 │
    │  rtc_policy_server.py (Py 3.12) ←──┘ Gets actions   │
    │  - Loads TestVLA checkpoint                           │
    │  - Runs real-time chunking (RTC)                      │
    │  - Sends action back                                  │
    │                                     │
    └─────────────────────────────────────┘
```

## What We Tested

**Date:** Oct 4, 2026  
**Model:** `Ultrox9504/TestVLA` (SmolVLA, trained 6 hours on SO101 vial task)  
**Dataset:** CursedRock17/so101_teleop_vials_sim_and_real  
**Run:** 3 episodes, RTC enabled

**Results:**

```
[GRASP] Vial grasped in env(s): [0]
[RACK] vial_2 placed in rack in env(s): [0]
[RELEASE] Vial released in env(s): [0]
...
[INFO]: Evaluated 3 episodes
[INFO]: Success Rate: 1/3 (33.3%)
```

**What that means:**
- ✅ Episode 1: Grasped, placed, released. **Success.**
- ❌ Episode 2: Grasped, couldn't place it. Failed.
- ❌ Episode 3: Grasped, couldn't place it. Failed.

So the grasping works pretty reliably. The placement is hit-or-miss. That's fine for a 6-hour training run on simulated data — not production-ready, but good enough to verify the pipeline works end-to-end.

## Inside the Container

If you're already in the container, just run the two steps separately:

**Terminal 1:**
```bash
cd /workspace/humanoid/src/simulation/so101_vial_task/scripts
PYTHONPATH=/workspace/isaaclab/lerobot/src:/workspace/humanoid/src/il \
  /opt/vla_env/bin/python rtc_policy_server.py \
  --policy_path /workspace/humanoid/outputs/TestVLA \
  --execution_horizon 10
```

**Terminal 2:**
```bash
cd /workspace/humanoid/src/simulation/so101_vial_task
DISPLAY=:1 PYTHONPATH=/workspace/humanoid/src/il:/workspace/humanoid/src/simulation/so101_vial_task \
  /workspace/isaaclab/isaaclab.sh -p scripts/lerobot_eval_rtc_remote.py \
  --task Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval \
  --num_episodes 1 --execution_horizon 10 --server_port 5556
```

## Run via Ultron — Full Guide

**Ultron** is the autonomous AI agent that orchestrates everything. It reads your task description, builds a SLURM job script, submits it, and self-heals if anything breaks. For VLA policy evaluation, here are all the ways to use it:

### Setup (one time)

```bash
cd ~/IsaacLab/FallRepo/Ultron/ultron
bash install.sh
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
claude        # Then type: /login   (to authorize Claude Code)
vim .env      # Add HF_TOKEN and (optional) Discord webhook
```

### Basic Usage

```bash
# Default: plans and auto-chooses overlap or new allocation
ultron "run VLA policy eval for 3 episodes"

# See the plan without submitting
ultron --dry-run "run VLA policy eval for 3 episodes"
```

### Where to Run

```bash
# Overlap inside your current SLURM job (fastest, reuses dockerd)
ultron "run VLA policy eval for 5 episodes" -overlap

# Overlap inside a specific running job
ultron "run VLA policy eval for 5 episodes" -overlap 682119

# Fresh allocation (sbatch, new node)
ultron "run VLA policy eval for 5 episodes" -new

# Auto mode (default): overlap if time allows, else new
ultron --mode auto "run VLA policy eval for 5 episodes"
```

### Custom Resources (new allocation only)

```bash
# Specific GPU and memory
ultron -new --gpus 2 --gpu-type rtx_3090 --mem 64G --time 2:00:00 \
  "run VLA policy eval for 10 episodes"

# Pin to a specific node
ultron -new --node trpro-slurm1 "run VLA policy eval for 5 episodes"

# Use a GPU shard instead of a whole GPU (24 GB VRAM)
ultron -new --shard 24000 "run VLA policy eval for 3 episodes"
```

### Manage Jobs

```bash
ultron --list                 # Show all submitted jobs
ultron --status 682185        # Check status of a job
ultron --logs                 # Show latest run's full output
ultron --logs 682119.2 -f     # Follow latest step of run 682119.2
ultron --cancel 682119.2      # Cancel a specific run
```

### Advanced

```bash
# Use local Ollama backup if Claude is down
ultron --backend ollama "run VLA policy eval for 2 episodes"

# See the generated job script and plan before submit
ultron --dry-run -new "run VLA policy eval for 3 episodes"
```

### What Ultron Does

1. **Claude reads your repo** (`Ultron/ultron/README.md`, training logs, scripts) and **builds a plan**
2. **Generates a SLURM script** (no secrets baked in; reads from `.env` at runtime)
3. **Submits** (via `srun --overlap` or `sbatch`)
4. **Executes step by step** on the compute node
5. **Self-heals** if anything fails — reruns with fixes, no manual intervention
6. **Verifies success** at the end (checks for the `Success Rate:` line)
7. **Saves learned patterns** to `Ultron/ultron/information/learned/` for future runs
8. **Sends Discord updates** (job queued, started, each step, completion)

### How It Works (Overlap vs New)

| | Overlap | New |
|---|---|---|
| **How** | Runs as `srun --overlap` inside your current job | Submits new `sbatch` job |
| **Speed** | Instant, reuses dockerd + containers | Waits in queue, ~2 min setup (shader compilation) |
| **Time limit** | Parent job's remaining time | You specify `--time` (or defaults from config) |
| **Best for** | Quick tests, debugging, iterating | Long runs, want a fresh node guarantee |
| **Job ID** | `<job>.<step>`, e.g. `682119.2` | `<job>` |

### Tips

- **First run is slow** — RTX shader compilation takes ~20 min. Overlap into a running job to reuse the container.
- **Prompts are flexible** — Ultron understands natural language: `"test the VLA on 10 episodes"`, `"quick eval of the policy"`, etc.
- **Healing is automatic** — If the Docker pull fails or the model download times out, Ultron retries and fixes it.
- **Discord notifications** — Every step update and job status goes to your webhook (if configured).
- **Dry-run first** — Always run `--dry-run` to review the plan before submitting big jobs.

### Examples

```bash
# Quick sanity check (overlap, 1 episode, expected to finish in 5 min)
ultron "test VLA policy on 1 episode" -overlap

# Batch eval on a fresh node (many episodes, takes ~30 min including setup)
ultron -new --time 1:00:00 "comprehensive VLA evaluation: 50 episodes"

# Debug if something breaks (dry-run to see the plan)
ultron --dry-run -new "run VLA policy eval for 2 episodes"
```

For the full Ultron guide, see `Ultron/ultron/README.md`.

## Logs

- **Server:** `/tmp/testvla_server.log` — policy loading, pings, errors
- **Client:** `/tmp/testvla_sim.log` — full Isaac output, shader compilation, sim warnings

## GUI?

Yes! The sim runs on the container's `DISPLAY=:1` (Xvfb). To watch:

```bash
ssh -L 5900:localhost:5900 user@watcloud
vncviewer localhost:5900        # from your machine
```

Password is usually in the `.container.cfg` or just empty.

## Files Added

- `Ultron/` — The autonomous agent that orchestrates all this
- `src/simulation/so101_vial_task/scripts/rtc_policy_server.py` — The policy server
- `src/simulation/so101_vial_task/scripts/lerobot_eval_rtc_remote.py` — Remote eval that calls the server
