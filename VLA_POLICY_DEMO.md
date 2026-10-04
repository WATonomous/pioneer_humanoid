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
