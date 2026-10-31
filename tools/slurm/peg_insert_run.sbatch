#!/bin/bash
# Run peg-insert pipeline commands on a WATcloud GPU node, in a fresh venv built from a code snapshot.
#
#   sbatch --job-name=peg-eval tools/slurm/peg_insert_run.sbatch RUN_NAME COMMANDS_FILE
#
# RUN_NAME       output directory name under $PERSIST/runs/ (the job id is appended)
# COMMANDS_FILE  bash script (on the login node) run with $PY, $RUN, $PERSIST, $REPO set;
#                e.g. `$PY -m humanoid_mujoco_scenes.peg_insert.eval --checkpoint ... --report $RUN/eval.json`
# Environment overrides: CODE_TGZ (repo snapshot, default $LOGIN/artifacts/code.tgz), PERSIST, LOGIN.
# Choose the GPU at submission, e.g. --nodelist=trpro-slurm2 --gres=gpu:rtx_4090:1,tmpdisk:65536.
#SBATCH --partition=compute
#SBATCH --nodelist=trpro-slurm1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_3090:1,tmpdisk:65536
#SBATCH --time=24:00:00
#SBATCH --output=/home/law_zou/humanoid-peg-insert/logs/%x-%j.out
set -euo pipefail
RUN_NAME=$1
COMMANDS=$2
PERSIST=${PERSIST:-/mnt/wato-drive/law_zou/humanoid-peg-insert}
LOGIN=${LOGIN:-/home/law_zou/humanoid-peg-insert}
CODE_TGZ=${CODE_TGZ:-$LOGIN/artifacts/code.tgz}
WORK=/tmp/peg_${SLURM_JOB_ID}
RUN="$PERSIST/runs/${RUN_NAME}_${SLURM_JOB_ID}"
REPO="$WORK/repo"
mkdir -p "$WORK/tmp" "$RUN/logs" "$REPO"
# Arrow materializes combined image datasets; keep the HF cache on NAS, not limited /tmp.
# Wheels (~3 GB of torch/CUDA) are cached on NAS so a flaky download is paid for once, not per job.
export TMPDIR="$WORK/tmp" UV_CACHE_DIR="$PERSIST/cache/uv" HF_HOME="$PERSIST/cache/huggingface"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
# The nodes' PyPI downloads time out intermittently; retry instead of failing the job at install.
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4
exec > >(tee -a "$RUN/logs/pipeline.log") 2>&1
trap 'status=$?; printf "exit %s\n" "$status" > "$RUN/logs/exit_status"; exit $status' EXIT
trap 'exit 143' TERM
# The exact code and commands that produced this run are kept next to its outputs.
cp "$CODE_TGZ" "$RUN/code.tgz"
cp "$COMMANDS" "$RUN/commands.sh"
tar -xzf "$RUN/code.tgz" -C "$REPO"
printf 'Job %s on %s: %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(nvidia-smi --query-gpu=name --format=csv,noheader)"
retry() {  # the nodes' downloads fail intermittently; the NAS cache keeps what already arrived
  for attempt in 1 2 3 4 5; do "$@" && return 0; printf 'retry %s failed: %s\n' "$attempt" "$*"; sleep $((attempt * 30)); done
  return 1
}
retry python3 -m pip install --no-cache-dir --target "$WORK/bootstrap" uv
UV="$WORK/bootstrap/bin/uv"
"$UV" venv --python /usr/bin/python3 "$WORK/venv"
PY="$WORK/venv/bin/python"
retry "$UV" pip install --python "$PY" 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu126
retry "$UV" pip install --python "$PY" 'torchcodec==0.5' 'lerobot @ git+https://github.com/huggingface/lerobot.git@e670ac5daf9b76' \
  'mujoco==3.15.0' pillow imageio imageio-ffmpeg \
  -e "$REPO/src/pioneer_humanoid" -e "$REPO/src/simulation/mujoco_scenes" -e "$REPO/src/robot_learning"
export PY RUN PERSIST REPO
cd "$RUN"
bash -euo pipefail "$RUN/commands.sh"
printf 'PEG_RUN_COMPLETE %s\n' "$RUN"
