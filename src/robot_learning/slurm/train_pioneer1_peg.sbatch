#!/bin/bash
# Phase 0: train Pioneer-1 (pioneer1) on the 40 synthetic peg_insert demos. From the repo root:  sbatch src/robot_learning/slurm/train_pioneer1_peg.sbatch
# 10k steps, not the README's 40k: 40 demos, not 600. The learning rate decays over --steps, so set it up front.
#SBATCH --job-name=pioneer1_peg
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:16384,tmpdisk:20480
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j-%x.out
set -e
source ~/venvs/pioneer/bin/activate
cd src/robot_learning
OUT=${OUT:-$HOME/outputs/pioneer1_peg_40}
python -m humanoid_robot_learning.train \
  --policy.type=pioneer1 --dataset.video_backend=pyav \
  --dataset.repo_id=zabsy/pioneer_8dof_peg_insert_synthetic \
  --steps=${STEPS:-10000} --batch_size=${BATCH:-32} --save_freq=${SAVE_FREQ:-2500} --policy.scheduler_warmup_steps=$(( ${STEPS:-10000} / 10 )) \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=$OUT
