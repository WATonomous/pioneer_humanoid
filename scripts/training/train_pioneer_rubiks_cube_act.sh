#!/usr/bin/env bash
# ACT training for the Rubik's cube LeRobot dataset.
#
# Requires LeRobot 0.4.3 and CUDA. WATcloud uses a standalone virtualenv.
# Submit scripts/training/train_pioneer_rubiks_cube_act.slurm from the repository root.
# Dataset and evaluation details: src/robot_learning/rubiks_cube.md.
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-learning-law/pioneer_v1_left_arm_rubiks_cube}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/pioneer_rubiks_cube_act}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${PYTHON:-python3}}"

# LeRobot requires a fresh output directory. It creates it itself.
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output already exists: $OUTPUT_DIR. Set OUTPUT_DIR to a new run directory." >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"

# Baseline: 10,000 steps, batch 8, save every 5,000 steps.
# Use STEPS=10 SAVE_FREQ=10 for installation validation.
# train-policy is an interactive shell alias, unavailable in batch scripts.
exec "$TRAIN_PYTHON" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir="$OUTPUT_DIR" \
  --steps="${STEPS:-10000}" \
  --batch_size="${BATCH_SIZE:-8}" \
  --save_freq="${SAVE_FREQ:-5000}" \
  --seed=1000 \
  --wandb.enable=false \
  --job_name=pioneer_rubiks_cube_act_smoke \
  "$@"

# Longer run, after reviewing the baseline:
#   OUTPUT_DIR=outputs/train/pioneer_rubiks_cube_act_50k STEPS=50000 \
#     SAVE_FREQ=10000 bash scripts/training/train_pioneer_rubiks_cube_act.sh

# Checkpoint when done (smoke run):
#   $OUTPUT_DIR/checkpoints/last/pretrained_model
#
# For the recorded cube pick and place, see the WATcloud evaluation job in
# src/robot_learning/rubiks_cube.md.
