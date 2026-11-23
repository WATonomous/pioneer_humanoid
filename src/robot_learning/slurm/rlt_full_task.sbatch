#!/bin/bash
# Evaluate a finished RLT run (humanoid_robot_learning/rlt/full_task.py): full peg_insert task with the actor taking
# over at the handoff vs the base alone, paired on the same layouts; plus the critical phase again from the run's eval
# start states. From the repo root:  sbatch src/robot_learning/slurm/rlt_full_task.sbatch   (BASE=, RLT= as below)
#SBATCH --job-name=rlt_full_task
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gres=shard:20480,tmpdisk:10240
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j-%x.out
set -euo pipefail
source ~/venvs/pioneer/bin/activate
export MUJOCO_GL=egl
cd src/robot_learning
BASE=${BASE:-$HOME/outputs/pioneer1_peg_100}
RLT=${RLT:-$HOME/outputs/rlt_peg_100}
python -m humanoid_robot_learning.rlt.full_task --policy "$BASE/checkpoints/last/pretrained_model" --rlt "$RLT" \
  --episodes ${EPISODES:-100} --videos ${VIDEOS:-6}
