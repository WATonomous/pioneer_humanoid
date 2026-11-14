#!/bin/bash
# Evaluate every checkpoint of a training run in sim (humanoid_robot_learning.sim.evaluate), recording the rollouts.
#
#   sbatch --job-name=eval-pi05 --dependency=afterok:<train job> tools/slurm/eval.sbatch RUN_NAME CHECKPOINTS_DIR \
#     --scene drawer_stow --episodes 40 --seed 1000 --max_s 90
#
# RUN_NAME         output folder name under $PERSIST/runs/ (the job id is appended)
# CHECKPOINTS_DIR  a LeRobot run's train/checkpoints (one numbered folder per checkpoint, each with pretrained_model/)
# Extra args go to evaluate.py for every checkpoint; --policy, --record and --device are set here. Per checkpoint:
# <run>/eval_<step>.log and the rollouts (LeRobot dataset + RECAP sidecar) in <run>/rollouts_<step>; summary.txt
# collects the success lines. Environment: STEPS="006000 010000" evaluates only those checkpoints (default: all);
# CODE_TGZ, PERSIST, LOGIN as in train.sbatch.
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_3090:1,tmpdisk:65536
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/humanoid/logs/%x-%j.out
set -euo pipefail
RUN_NAME=$1
CHECKPOINTS=$2
shift 2
PERSIST=${PERSIST:-/mnt/wato-drive/$USER/humanoid}
LOGIN=${LOGIN:-/home/$USER/humanoid}
CODE_TGZ=${CODE_TGZ:-$LOGIN/artifacts/code.tgz}
RUN="$PERSIST/runs/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN/logs"
exec > >(tee -a "$RUN/logs/pipeline.log") 2>&1
trap 'status=$?; printf "exit %s\n" "$status" > "$RUN/logs/exit_status"; exit $status' EXIT
trap 'exit 143' TERM
cp "$CODE_TGZ" "$RUN/code.tgz"
printf '%q ' sbatch eval.sbatch "$RUN_NAME" "$CHECKPOINTS" "$@" > "$RUN/command.txt"
WORK=/tmp/humanoid_${SLURM_JOB_ID}
REPO="$WORK/repo"
mkdir -p "$REPO"
tar -xzf "$RUN/code.tgz" -C "$REPO"
source "$REPO/tools/slurm/node_env.sh"
printf 'Job %s on %s: %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(nvidia-smi --query-gpu=name --format=csv,noheader || true)"

cd "$RUN"
steps=${STEPS:-$(ls "$CHECKPOINTS" | grep -E '^[0-9]+$' | sort)}   # numbered folders only (not the "last" link)
for step in $steps; do
  ckpt="$CHECKPOINTS/$step"
  "$PY" -m humanoid_robot_learning.sim.evaluate --policy "$ckpt/pretrained_model" --device cuda \
    --record "$RUN/rollouts_$step" "$@" 2>&1 | tee "$RUN/eval_$step.log"
  printf '%s %s\n' "$step" "$(grep -E '^\[EVAL\] .*: success' "$RUN/eval_$step.log" | tail -1)" >> "$RUN/summary.txt"
done
cat "$RUN/summary.txt"
printf 'EVAL_RUN_COMPLETE %s\n' "$RUN"
