#!/bin/bash
# Train a LeRobot policy (pi0.5, ACT, humanoid_dit, ...) on a WATcloud GPU node, from a repo snapshot.
#
#   sbatch --job-name=pi05-drawer tools/slurm/train.sbatch RUN_NAME DATASET [lerobot-train args ...]
#
# RUN_NAME  output folder name under $PERSIST/runs/ (the job id is appended); training lands in <run>/train
# DATASET   LeRobot dataset folder on NAS (meta/, data/, videos/)
# Extra args go to `python -m humanoid_robot_learning.train` (= lerobot-train with our policies registered);
# --dataset.root, --dataset.repo_id and --output_dir are set here.
#
# The dataset is copied to node-local disk first (video decoding from NAS is slow). For
# --policy.use_relative_actions=true its action stats must be relative (lerobot-edit-dataset recompute_stats).
# Environment: CODE_TGZ, PERSIST, LOGIN   snapshot and folders (defaults below); LEROBOT_SPEC, TORCH_INDEX: node_env.sh
# Pick the GPU at submission, e.g. --nodelist=trpro-slurm2 --gres=gpu:rtx_4090:1,tmpdisk:131072.
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_3090:1,tmpdisk:131072
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/humanoid/logs/%x-%j.out
set -euo pipefail
RUN_NAME=$1
DATASET=$2
shift 2
PERSIST=${PERSIST:-/mnt/wato-drive/$USER/humanoid}
LOGIN=${LOGIN:-/home/$USER/humanoid}
CODE_TGZ=${CODE_TGZ:-$LOGIN/artifacts/code.tgz}
RUN="$PERSIST/runs/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN/logs"
exec > >(tee -a "$RUN/logs/pipeline.log") 2>&1
trap 'status=$?; printf "exit %s\n" "$status" > "$RUN/logs/exit_status"; exit $status' EXIT
trap 'exit 143' TERM
cp "$CODE_TGZ" "$RUN/code.tgz"   # the exact code and command that produced this run sit next to its outputs
printf '%q ' sbatch train.sbatch "$RUN_NAME" "$DATASET" "$@" > "$RUN/command.txt"
WORK=/tmp/humanoid_${SLURM_JOB_ID}
REPO="$WORK/repo"
mkdir -p "$REPO"
tar -xzf "$RUN/code.tgz" -C "$REPO"
source "$REPO/tools/slurm/node_env.sh"
printf 'Job %s on %s: %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(nvidia-smi --query-gpu=name --format=csv,noheader || true)"

TRAIN_DATA="$WORK/dataset"
cp -r "$DATASET" "$TRAIN_DATA"
cd "$RUN"
# A local dataset: the repo id only names it.
"$PY" -m humanoid_robot_learning.train --dataset.repo_id="humanoid/$RUN_NAME" --dataset.root="$TRAIN_DATA" \
  --output_dir="$RUN/train" "$@"
printf 'TRAIN_RUN_COMPLETE %s\n' "$RUN"
