#!/bin/bash
# RECAP policy rollouts in parallel: a Slurm job array, one seed range per task, each recorded as its own dataset
# with its RECAP sidecar (sim/evaluate.py --record). Merge the shards afterwards with humanoid-merge-takes.
#
#   sbatch --array=0-7 --job-name=rollouts-drawer tools/slurm/rollouts.sbatch RUN_NAME EPISODES_PER_TASK SEED ROUND \
#     --scene drawer_stow --policy /mnt/wato-drive/$USER/humanoid/runs/<train run>/train/checkpoints/last/pretrained_model
#
# RUN_NAME           output folder under $PERSIST/runs/ (the array's job id is appended), shared by all tasks
# EPISODES_PER_TASK  episodes per task; task i plays layout seeds SEED + i * EPISODES_PER_TASK, ...
# SEED, ROUND        layout seed of task 0's first episode; RECAP round stored with the rollouts
# Extra args go to `python -m humanoid_robot_learning.sim.evaluate` (--scene, --policy, --max_s, --advantage, ...);
# --episodes, --seed, --record and --round are set here.
#
# The policy runs on the task's GPU, MuJoCo and its rendering (EGL) on the node. Each task records to node-local disk
# and copies its shard to <run>/shard_<i> when done; logs: <run>/logs/task_<i>.log, exit status: <run>/logs/exit_<i>.
# Environment: CODE_TGZ, PERSIST, LOGIN (as train.sbatch); LEROBOT_SPEC, TORCH_INDEX: node_env.sh
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1,tmpdisk:65536
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/humanoid/logs/%x-%A_%a.out
set -euo pipefail
RUN_NAME=$1
PER_TASK=$2
SEED0=$3
ROUND=$4
shift 4
PERSIST=${PERSIST:-/mnt/wato-drive/$USER/humanoid}
LOGIN=${LOGIN:-/home/$USER/humanoid}
CODE_TGZ=${CODE_TGZ:-$LOGIN/artifacts/code.tgz}
TASK=${SLURM_ARRAY_TASK_ID:?submit with --array}
RUN="$PERSIST/runs/${RUN_NAME}_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN/logs"
exec > >(tee -a "$RUN/logs/task_${TASK}.log") 2>&1
trap 'status=$?; printf "exit %s\n" "$status" > "$RUN/logs/exit_${TASK}"; exit $status' EXIT
trap 'exit 143' TERM
WORK=/tmp/humanoid_${SLURM_JOB_ID}
REPO="$WORK/repo"
mkdir -p "$REPO"
tar -xzf "$CODE_TGZ" -C "$REPO"
if [ "$TASK" = "${SLURM_ARRAY_TASK_MIN:-0}" ]; then   # one copy of the code and command per array
  cp "$CODE_TGZ" "$RUN/code.tgz"
  printf '%q ' sbatch "--array=${SLURM_ARRAY_TASK_MIN:-0}-${SLURM_ARRAY_TASK_MAX:-0}" rollouts.sbatch \
    "$RUN_NAME" "$PER_TASK" "$SEED0" "$ROUND" "$@" > "$RUN/command.txt"
fi
source "$REPO/tools/slurm/node_env.sh"
SEED=$((SEED0 + TASK * PER_TASK))
printf 'Task %s of job %s on %s: seeds %s..%s, %s\n' "$TASK" "$SLURM_ARRAY_JOB_ID" "$(hostname)" "$SEED" \
  "$((SEED + PER_TASK - 1))" "$(nvidia-smi --query-gpu=name --format=csv,noheader || true)"

SHARD="$WORK/shard_${TASK}"
"$PY" -m humanoid_robot_learning.sim.evaluate "$@" --episodes "$PER_TASK" --seed "$SEED" --record "$SHARD" \
  --round "$ROUND"
cp -r "$SHARD" "$RUN/shard_${TASK}.tmp"
mv "$RUN/shard_${TASK}.tmp" "$RUN/shard_${TASK}"   # a complete shard folder only
rm -rf "$WORK"
printf 'ROLLOUT_SHARD_COMPLETE %s/shard_%s\n' "$RUN" "$TASK"
