#!/bin/bash
# Phase 0: train Pioneer-1 (pioneer1) on the 40 synthetic peg_insert demos. From the repo root:  sbatch src/robot_learning/slurm/train_pioneer1_peg.sbatch
# 10k steps, not the README's 40k: 40 demos, not 600. The learning rate decays over --steps, so set it up front.
# Another dataset: DATASET=<repo id>, plus DATASET_ROOT=<dir> for a local copy (e.g. the 100 takes from
# record_peg_demos.sbatch). Batch 32 peaks at 15 GB of GPU memory, so the shard is 20 GB.
#SBATCH --job-name=pioneer1_peg
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:20480,tmpdisk:20480
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j-%x.out
set -e
source ~/venvs/pioneer/bin/activate
cd src/robot_learning
OUT=${OUT:-$HOME/outputs/pioneer1_peg_40}
# A checkpoint with optimizer state is about 2.4 GB and the WATcloud home quota is 20 GB, so keep SAVE_FREQ coarse.
# RESUME=1 continues OUT from its last checkpoint (same config; extra flags such as --save_freq still apply).
if [ "${RESUME:-0}" = 1 ]; then
  exec python -m humanoid_robot_learning.train --config_path=$OUT/checkpoints/last/pretrained_model/train_config.json \
    --resume=true ${SAVE_FREQ:+--save_freq=$SAVE_FREQ}
fi
python -m humanoid_robot_learning.train \
  --policy.type=pioneer1 --dataset.video_backend=pyav \
  --dataset.repo_id=${DATASET:-zabsy/pioneer_8dof_peg_insert_synthetic} ${DATASET_ROOT:+--dataset.root=$DATASET_ROOT} \
  --steps=${STEPS:-10000} --batch_size=${BATCH:-32} --save_freq=${SAVE_FREQ:-2500} --policy.scheduler_warmup_steps=$(( ${STEPS:-10000} / 10 )) \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=$OUT
