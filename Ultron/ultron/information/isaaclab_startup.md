# Steps to Start IsaacLab

## Quick Flow

1. Git Bash → start interactive session
2. Terminal → SSH into dev session
3. Edit IsaacLab session config (safety)
4. Start SLURM + dockerd
5. Start container
6. Enter container
7. Fix/restart display (if VNC breaks)
8. Run IsaacLab + connect with VNC

---

## Git Bash

### 1. Start interactive session

- Go to `wato_asd_tooling`
- Run:

```bash
bash start_interactive_session.sh
```

> A "Pseudo-terminal will not be allocated" message is normal. The WATonomous logo should appear.

---

## Terminal

### 2. SSH into dev session

- Open a **new terminal**
- Run:

```bash
ssh -L 5900:localhost:5900 asd-dev-session
```

- Then:

```bash
cd IsaacLab && export DISPLAY=:1
```

### 3. Edit session config (safety)

- Edit the IsaacLab session config file
- Reference: https://chatgpt.com/s/t_6865bb0b4d7c8191b8523b4f49a0f79e

### 4. Start SLURM session + Docker

**a. Start SLURM** (pick one):

- New session:

```bash
srun --cpus-per-task 8 --mem 64G --gres shard:24084,tmpdisk:102400 --time 6:00:00 --pty bash
```

- Rejoin existing session:

```bash
srun --pty --overlap --jobid 222127 bash
```

**b. Start dockerd:**

```bash
export DOCKER_DATA_ROOT="/mnt/wato-drive2/rijul_chaddha/docker_data"
slurm-start-dockerd.sh
```

- Wait for: `Dockerd started successfully!`
- Optional test:

```bash
docker run --rm hello-world
```

### 5. Start the container

- **NOTE:** `container.py` does NOT have a `build` subcommand (valid commands are `start`, `enter`, `config`, `copy`, `stop`). Running `./docker/container.py start ros2` automatically builds the image (if not built) and runs the container in detached mode.
- Takes about **800–1200 seconds** for first-time builds

```bash
./docker/container.py start ros2
```

### 6. Executing Commands Inside Docker (Automated & SLURM Jobs)

> **Important for automated scripts:** Do NOT run `./docker/container.py enter` in automated SLURM jobs (it requires an interactive TTY). Instead, use `docker exec`:

- **Run Python / PyTorch / CUDA inside container:**
  Always use `/workspace/isaaclab/isaaclab.sh -p` (which invokes Isaac Sim's Python environment):
  ```bash
  docker exec isaac-lab-ros2 /workspace/isaaclab/isaaclab.sh -p -c "import torch; print('CUDA:', torch.cuda.is_available())"
  ```
  *(Do NOT use bare `python3` inside the container because PyTorch and Isaac Sim are installed in IsaacLab's internal Python environment, not the system Python).*

- **Run IsaacLab helper / scripts:**
  ```bash
  docker exec isaac-lab-ros2 /workspace/isaaclab/isaaclab.sh -p scripts/tutorials/04_sensors/run_frame_transformer.py --headless
  ```

For interactive human sessions:
```bash
./docker/container.py enter ros2
```

### 7. Restart display (if VNC breaks)

- Reference: https://chatgpt.com/c/68884ca7-af28-8004-9cfc-755f84e7d490

```bash
# 1. Kill old X, VNC, LXDE processes
pkill -f Xvfb
pkill -f x11vnc
pkill -f startlxde

# 2. Clean leftover lock files and sockets
rm -f /tmp/.X1-lock
rm -rf /tmp/.X11-unix/X1
rm -f /tmp/.Xauthority

# 3. Create fresh Xauthority + magic cookie for display :1
touch /tmp/.Xauthority
xauth add :1 . $(mcookie)

# 4. Start Xvfb on display :1 (auth disabled for simplicity)
Xvfb :1 -screen 0 1280x1024x24 -auth /tmp/.Xauthority -ac &

# 5. Set DISPLAY
export DISPLAY=:1

# 6. Start VNC server on display :1
x11vnc -display :1 -auth /tmp/.Xauthority -nopw -forever -shared &

# 7. Start lightweight desktop
startlxde &

# 8. (Optional) Let things settle
sleep 5

# 9. Test with a simple X app
xclock &
```

### 8. Run IsaacLab

```bash
./isaaclab.sh -s
```

- Download [VNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)
- Connect to `localhost:5900`

---

## Other Commands

| # | Purpose | Command |
|---|---------|---------|
| 1 | Check process | `squeue -u $USER` |
| 2 | Activate conda shell | `eval "$(/home/rijul_chaddha/miniconda3/bin/conda shell.bash hook)"` |
| 3 | Activate env | `conda activate env_isaaclab` |
| 4 | Start VNC server | `x11vnc -display :1 -auth /tmp/.Xauthority -nopw -forever -shared` |
| 5 | Fix numpy | `pip install numpy==1.26.4` |
| 6 | Install usd-core | `pip install usd-core==23.11` |
| 7 | Install lxml | `pip install lxml==4.9.2` |
| 8 | Install nano | `apt update && apt install nano -y` |
| 9 | Source ROS2 | `source /opt/ros/humble/setup.bash` |
| 10 | Frame transformer demo | `./isaaclab.sh -p scripts/tutorials/04_sensors/run_frame_transformer.py` |
| 11 | Imitation learning test | `python3 ImitationLearning/task_space_test.py --headless` |

- Numpy fix reference: https://gemini.google.com/app/805719bf83aea6af

---

## Kill a Process

1. Find the process:

```bash
ps aux | grep python
```

2. Read the PID (2nd column), then kill it:

```bash
kill -9 <PID>
```

---

## Run Scripts

- Leatherback robot (1 env, cameras on):

```bash
./isaaclab.sh -p AutonomousVehicle/LeatherBack_Robot.py --num_envs 1 --enable_cameras
```

- Arms demo (test):

```bash
./isaaclab.sh -p scripts/demos/arms.py
```