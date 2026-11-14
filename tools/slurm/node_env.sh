# Build a job's Python environment on a WATcloud node. Sourced by the job scripts here from the extracted code
# snapshot (sbatch runs a spooled copy of the job script, so this file is found at $REPO/tools/slurm/), after they
# set PERSIST (NAS root), WORK (node-local scratch) and REPO (the snapshot, extracted under $WORK).
#
# Sets and exports: PY (venv python), plus the caches and render settings below.
# Overrides: LEROBOT_SPEC (pip spec of LeRobot, with pi0.5), TORCH_INDEX (torch's CUDA build).

mkdir -p "$WORK/tmp"
# Wheels (~3 GB of torch/CUDA) and the managed Python are cached on NAS: a flaky download is paid for once, not per job.
export TMPDIR="$WORK/tmp" UV_CACHE_DIR="$PERSIST/cache/uv" UV_PYTHON_INSTALL_DIR="$PERSIST/cache/uv-python"
export HF_HOME="$PERSIST/cache/huggingface"   # pi05_base (~14 GB) is downloaded once
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# The nodes' PyPI downloads time out intermittently; retry instead of failing the job at install.
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4
LEROBOT_SPEC=${LEROBOT_SPEC:-"lerobot[pi,training]==0.7.0"}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

retry() {  # the NAS cache keeps what already arrived, so a retry only fetches the rest
  for attempt in 1 2 3 4 5; do
    "$@" && return 0
    printf 'retry %s failed: %s\n' "$attempt" "$*"
    sleep $((attempt * 30))
  done
  return 1
}

retry python3 -m pip install --no-cache-dir --target "$WORK/bootstrap" uv
UV="$WORK/bootstrap/bin/uv"
retry "$UV" venv --python 3.12 "$WORK/venv"   # LeRobot 0.7 needs Python >= 3.12; uv fetches it if the node lacks it
PY="$WORK/venv/bin/python"
retry "$UV" pip install --python "$PY" 'torch==2.7.1' 'torchvision==0.22.1' --index-url "$TORCH_INDEX"
retry "$UV" pip install --python "$PY" 'torchcodec==0.5' "$LEROBOT_SPEC" 'mujoco==3.15.0' pillow imageio imageio-ffmpeg psutil \
  -e "$REPO/src/pioneer_humanoid" -e "$REPO/src/simulation/mujoco_scenes" -e "$REPO/src/robot_learning"
export PY
