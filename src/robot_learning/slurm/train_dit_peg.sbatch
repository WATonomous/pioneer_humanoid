#!/bin/bash
# Phase 0: train humanoid_dit on the 40 synthetic peg_insert demos. From the repo root:  sbatch src/robot_learning/slurm/train_dit_peg.sbatch
# 25k steps, not the README's 70k: 40 demos overfit at 70k. The learning rate decays over --steps, so set it up front.
#SBATCH --job-name=dit_peg
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:16384,tmpdisk:20480
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j-%x.out
set -e
source ~/venvs/pioneer/bin/activate
cd src/robot_learning
OUT=${OUT:-$HOME/outputs/dit_peg_40}
python -m humanoid_robot_learning.train \
  --policy.type=humanoid_dit \
  --dataset.repo_id=zabsy/pioneer_8dof_peg_insert_synthetic \
  --steps=${STEPS:-25000} --batch_size=6 \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=$OUT
