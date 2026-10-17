# Troubleshooting & Failure Recovery Steps

This document outlines common failure modes and exact recovery actions to take when executing tasks.

## 0. Autonomous Debugging & Resilience Mandate
- **NEVER GIVE UP ON A SINGLE FAILURE:** Errors are diagnostic feedback. When a shell command fails with a non-zero exit code, the agent MUST NOT abort.
- **Inspect Terminal Stderr / Outputs:** Read the exact error message (e.g., invalid command choice, wrong directory, missing package, port busy).
- **Formulate Corrective Actions:**
  1. If a command fails due to invalid syntax/flags (e.g., `container.py build`), immediately switch to the valid flag (`container.py start ros2`) or fall back to native tools (`docker exec`, `docker ps`).
  2. If a file or executable is not found, verify the path using `ls`, `which`, or check the current working directory.
  3. If a container is not running, inspect `docker logs <container>` to see why it exited before trying to enter or execute into it.
- **Iterative Recovery:** The agent will automatically re-plan 1-3 corrective steps and execute them to reach the desired goal.

---

## 1. Failure Modes & Ordered Fixes

### Error 1: `Cannot connect to the Docker daemon at unix:///tmp/run/docker.sock`
- **Cause:** `dockerd` was not launched on the compute node or hasn't finished booting.
- **Fail Steps / Fix:**
  1. Check if `slurm-start-dockerd.sh` is available: `which slurm-start-dockerd.sh`.
  2. Start dockerd in background:
     ```bash
     slurm-start-dockerd.sh &
     sleep 10
     ```
  3. Verify socket exists: `ls -la /tmp/run/docker.sock`.
  4. Ensure `export DOCKER_HOST=unix:///tmp/run/docker.sock` is set in the current subshell.

---

### Error 2: `Container isaac-lab-ros2 is not running`
- **Cause:** Container failed to start, was stopped, or image is missing.
- **Fail Steps / Fix:**
  1. Inspect container status:
     ```bash
     docker ps -a --filter name=isaac-lab-ros2
     ```
  2. If exited, inspect crash logs:
     ```bash
     docker logs --tail 100 isaac-lab-ros2
     ```
  3. Start container cleanly:
     ```bash
     cd /home/rijul_chaddha/IsaacLab
     ./docker/container.py start ros2
     ```
  4. Verify it is running before launching commands into it.

---

### Error 2b: `ModuleNotFoundError: No module named 'torch'` inside container
- **Cause:** Command ran `python3` instead of the IsaacLab Python wrapper. PyTorch is installed inside Isaac Sim's virtual Python environment (`/workspace/isaaclab/_isaac_sim/python.sh`), not the container's base system python.
- **Fail Steps / Fix:**
  Run the command using IsaacLab's `-p` flag:
  ```bash
  docker exec isaac-lab-ros2 /workspace/isaaclab/isaaclab.sh -p -c "import torch; print('CUDA:', torch.cuda.is_available())"
  ```
  Or for scripts:
  ```bash
  docker exec isaac-lab-ros2 /workspace/isaaclab/isaaclab.sh -p <path_to_script.py>
  ```

---

### Error 3: `CUDA out of memory` (OOM)
- **Cause:** PyTorch memory fragmentation or batch size too large for GPU VRAM.
- **Fail Steps / Fix:**
  1. Export allocator memory management before script run:
     ```bash
     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     ```
  2. Reduce `--global-batch-size` or `--training-batch-size` by 50% (e.g., from 64 to 32 or 16).
  3. Compensate effective batch size by doubling `--gradient-accumulation-steps`.
  4. Check GPU memory usage: `nvidia-smi`.

---

### Error 4: `NCCL error: unhandled system error / P2P failure`
- **Cause:** GeForce RTX 3090/4090 GPUs do not support standard NVLink P2P without specialized bridging, causing NCCL distributed runs to freeze or error.
- **Fail Steps / Fix:**
  1. Explicitly disable P2P and SHM in NCCL:
     ```bash
     export NCCL_P2P_DISABLE=1
     export NCCL_SHM_DISABLE=1
     ```
  2. Re-run `torch.distributed.run`.

---

### Error 5: `FileNotFoundError: /usr/local/cuda/bin/nvcc`
- **Cause:** DeepSpeed or PyTorch tries to compile custom C++/CUDA ops when no full CUDA Toolkit compiler is in PATH inside container.
- **Fail Steps / Fix:**
  1. Disable DeepSpeed op compilation:
     ```bash
     export DS_BUILD_OPS=0
     export DS_SKIP_CUDA_CHECK=1
     export DEEPSPEED_SKIP_CUDA_CHECK=1
     ```
  2. Create mock nvcc script:
     ```bash
     mkdir -p /usr/local/cuda/bin
     printf '#!/bin/bash\necho nvcc: NVIDIA R Cuda compiler driver\necho Cuda compilation tools, release 12.2, V12.2.91\n' > /usr/local/cuda/bin/nvcc
     chmod +x /usr/local/cuda/bin/nvcc
     ```

---

### Error 6: `Ollama connection refused at http://localhost:11434`
- **Cause:** Ollama daemon was killed, failed to bind port, or did not start.
- **Fail Steps / Fix:**
  1. Check if process is running: `pgrep -l ollama`.
  2. Start server in background with logging:
     ```bash
     ollama serve > /tmp/ollama.log 2>&1 &
     sleep 5
     ```
  3. Check health endpoint:
     ```bash
     curl -s http://localhost:11434/api/tags
     ```
  4. If model is missing: `ollama pull qwen2.5-coder:32b`.

---

## 2. General Rule for Ordered Step Execution

When Ultron runs multi-step pipelines:
1. Every critical preparatory step must check its exit code `$? == 0`.
2. Do not proceed to training if dataset path does not exist.
3. If a step fails, write the stderr into the job log, notify Discord, and execute cleanup (stopping child processes, container unmounts).
