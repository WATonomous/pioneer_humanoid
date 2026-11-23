#!/usr/bin/env bash
# From the repository root: sbatch scripts/training/eval_watcloud_rubiks.slurm <pretrained_model> [<pretrained_model> ...]
#SBATCH --job-name=pioneer-rubiks-eval
#SBATCH --output=slurm-rubiks-eval-%j.out
#SBATCH --error=slurm-rubiks-eval-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:rtx_3090:1
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ $# == 0 ]]; then
  echo 'Pass at least one LeRobot pretrained_model directory.' >&2
  exit 1
fi
POLICIES=()
for checkpoint in "$@"; do
  if [[ ! -s "$checkpoint/model.safetensors" ]]; then
    echo "Missing model weights in $checkpoint" >&2
    exit 1
  fi
  POLICIES+=("$(realpath "$checkpoint")")
done
TRAIN_STORAGE="${TRAIN_STORAGE:-/mnt/wato-drive2/$USER/pioneer-training}"
VENV_DIR="${VENV_DIR:-$TRAIN_STORAGE/venv-lerobot043}"
PYTHON="$VENV_DIR/bin/python"
RESULTS="$TRAIN_STORAGE/eval/rubiks-$SLURM_JOB_ID"
SOURCE="$RESULTS/sim-source"
DEPS="$RESULTS/python-deps"
# Use the original left-arm scene, camera and robot assets together.
SIM_REVISION=66b5961d07b0c0706441f3f6ce5952d3fc1862f5
if ! git cat-file -e "$SIM_REVISION^{commit}" 2>/dev/null; then
  printf 'Fetch the recording-era simulator before submitting: git fetch --no-tags origin %s\n' "$SIM_REVISION" >&2
  exit 1
fi
mkdir -p "$SOURCE" "$RESULTS/tmp"
git archive "$SIM_REVISION" src/robot_learning src/pioneer_humanoid \
  src/simulation/mujoco_scenes assets/pioneer_bimanual_arm/meshes \
  assets/pioneer_bimanual_arm/urdf assets/props/lightbox_mujoco | tar -x -C "$SOURCE"
export TMPDIR="$RESULTS/tmp"
export PIP_CACHE_DIR="$TRAIN_STORAGE/pip-cache"
export HF_HOME="$TRAIN_STORAGE/huggingface"
export TORCH_HOME="$TRAIN_STORAGE/torch"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# Keep simulator-only dependencies out of the training environment.
"$PYTHON" -m pip install --no-compile --target "$DEPS" \
  --constraint "$TRAIN_STORAGE/requirements-resolved.txt" \
  mujoco==3.17.0 transformers==4.57.1
export PYTHONPATH="$SOURCE/src/robot_learning:$SOURCE/src/pioneer_humanoid:$SOURCE/src/simulation/mujoco_scenes:$DEPS"
echo "Simulation revision: $SIM_REVISION"
for index in "${!POLICIES[@]}"; do
  policy="${POLICIES[index]}"
  video_dir="$RESULTS/videos"
  if (( ${#POLICIES[@]} > 1 )); then
    step_name="$(basename "$(dirname "$policy")")"
    video_dir="$video_dir/$((index + 1))-$step_name"
  fi
  echo "Policy: $policy"
  if [[ -n "${POSE_EPISODES:-}" ]]; then
    pose_args=(--policy "$policy" --episodes "$POSE_EPISODES" --seed "${POSE_SEED:-2026}" --perturb "${POSE_PERTURB:-both}")
    pose_output="$RESULTS/pose-results-$((index + 1))"
    if [[ -n "${POSE_ENSEMBLE_COEFF:-}" ]]; then
      if [[ "${POSE_INCLUDE_BASELINE:-0}" == 1 ]]; then
        "$PYTHON" scripts/training/eval_rubiks_poses.py "${pose_args[@]}" --output "$pose_output.json"
      fi
      "$PYTHON" scripts/training/eval_rubiks_poses.py "${pose_args[@]}" \
        --ensemble-coeff "$POSE_ENSEMBLE_COEFF" --output "$pose_output-ensemble.json"
    elif [[ -n "${POSE_ACTION_STEPS:-}" ]]; then
      read -ra action_steps <<< "$POSE_ACTION_STEPS"
      for steps in "${action_steps[@]}"; do
        echo "Replan every $steps actions"
        "$PYTHON" scripts/training/eval_rubiks_poses.py "${pose_args[@]}" \
          --action-steps "$steps" --output "$pose_output-actions-$steps.json"
      done
    else
      "$PYTHON" scripts/training/eval_rubiks_poses.py "${pose_args[@]}" --output "$pose_output.json"
    fi
    continue
  fi
  "$PYTHON" -m humanoid_robot_learning.sim.evaluate --scene rubiks_cube \
    --policy "$policy" --device cuda --task 'sim leader teleop demonstration' \
    --episodes 1 --max_s 8 --fps 25 --video_dir "$video_dir"
done
echo "Rollout results directory: $RESULTS"
