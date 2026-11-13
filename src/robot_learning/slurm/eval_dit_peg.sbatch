#!/bin/bash
# Phase 0 eval: success rate of the trained DiT on 40 held-out layouts (seeds 1000+, never in the demos), with videos.
# From the repo root:  sbatch src/robot_learning/slurm/eval_dit_peg.sbatch   (OUT= the training output dir)
#SBATCH --job-name=dit_peg_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=shard:8192,tmpdisk:10240
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j-%x.out
set -e
source ~/venvs/pioneer/bin/activate
cd src/robot_learning
OUT=${OUT:-$HOME/outputs/dit_peg_40}
MUJOCO_GL=egl python -m humanoid_robot_learning.sim.evaluate --scene peg_insert \
  --policy $OUT/checkpoints/last/pretrained_model \
  --episodes ${EPISODES:-40} --seed 1000 --video_dir $OUT/eval_videos
