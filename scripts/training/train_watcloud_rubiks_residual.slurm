#!/usr/bin/env bash
# From the repository root: sbatch scripts/training/train_watcloud_rubiks_residual.slurm <pretrained_model>
#SBATCH --job-name=rubiks-residual
#SBATCH --output=slurm-rubiks-residual-%j.out
#SBATCH --error=slurm-rubiks-residual-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:rtx_3090:1
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ $# != 1 || ! -s "$1/model.safetensors" ]]; then
  echo 'Pass the path to the validated ACT pretrained_model directory.' >&2
  exit 1
fi
POLICY="$(realpath "$1")"
TRAIN_STORAGE="${TRAIN_STORAGE:-/mnt/wato-drive2/$USER/pioneer-training}"
VENV_DIR="${VENV_DIR:-$TRAIN_STORAGE/venv-lerobot043}"
PYTHON="$VENV_DIR/bin/python"
RESULTS="$TRAIN_STORAGE/runs/rubiks-residual-$SLURM_JOB_ID"
RL_DEPS="$RESULTS/rl-deps"
SIM_REVISION=66b5961d07b0c0706441f3f6ce5952d3fc1862f5
if ! git cat-file -e "$SIM_REVISION^{commit}" 2>/dev/null; then
  printf 'Fetch the recording-era simulator before submitting: git fetch --no-tags origin %s\n' "$SIM_REVISION" >&2
  exit 1
fi
export TMPDIR="$RESULTS/tmp"
export PIP_CACHE_DIR="$TRAIN_STORAGE/pip-cache"
export HF_HOME="$TRAIN_STORAGE/huggingface"
export TORCH_HOME="$TRAIN_STORAGE/torch"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"

if [[ -n "${SIM_RUNTIME:-}" ]]; then
  # Optionally reuse the simulator and dependencies from a verified evaluation job.
  SOURCE="$SIM_RUNTIME/sim-source"
  SIM_DEPS="$SIM_RUNTIME/python-deps"
  scene_path=src/simulation/mujoco_scenes/humanoid_mujoco_scenes/rubiks_cube/scene.py
  git show "$SIM_REVISION:$scene_path" | cmp - "$SOURCE/$scene_path"
  test -d "$SIM_DEPS/mujoco-3.17.0.dist-info"
else
  SOURCE="$RESULTS/sim-source"
  SIM_DEPS="$RESULTS/python-deps"
  mkdir -p "$SOURCE"
  git archive "$SIM_REVISION" src/robot_learning src/pioneer_humanoid \
    src/simulation/mujoco_scenes assets/pioneer_bimanual_arm/meshes \
    assets/pioneer_bimanual_arm/urdf assets/props/lightbox_mujoco | tar -x -C "$SOURCE"
  "$PYTHON" -m pip install --no-compile --target "$SIM_DEPS" \
    --constraint "$TRAIN_STORAGE/requirements-resolved.txt" \
    mujoco==3.17.0 transformers==4.57.1
fi

# Keep PPO's Gymnasium requirement local and reuse LeRobot's installed CUDA build.
"$PYTHON" -m pip install --no-compile --target "$RL_DEPS" \
  --constraint "$TRAIN_STORAGE/requirements-resolved.txt" matplotlib==3.10.8
"$PYTHON" -m pip install --no-compile --no-deps --target "$RL_DEPS" \
  stable-baselines3==2.7.0 gymnasium==1.2.3
export PYTHONPATH="$SOURCE/src/robot_learning:$SOURCE/src/pioneer_humanoid:$SOURCE/src/simulation/mujoco_scenes:$RL_DEPS:$SIM_DEPS"
echo "Frozen ACT: $POLICY"
echo "Simulator revision: $SIM_REVISION"
"$PYTHON" -u scripts/training/train_rubiks_residual.py --policy "$POLICY" --output "$RESULTS/smoke" --smoke
"$PYTHON" -u scripts/training/train_rubiks_residual.py --policy "$POLICY" --output "$RESULTS" \
  --steps "${RL_STEPS:-24576}" --save-every "${RL_SAVE_EVERY:-8192}"
echo "Residual models and evaluation: $RESULTS"
