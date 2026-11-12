#!/usr/bin/env bash
# Run inside simulation_isaac. Environment overrides select a later aggregate/run.
set -euo pipefail
task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
task_data="${CARGO_DATASET_ROOT:-${task_repo}/datasets/nasa_cargo/synthetic_50}"
task_output="${CARGO_OUTPUT_DIR:-${task_repo}/outputs/train/nasa_cargo_act_v1}"
task_steps="${CARGO_STEPS:-100000}"
task_batch="${CARGO_BATCH_SIZE:-4}"
task_weights="${CARGO_BACKBONE_WEIGHTS:-ResNet18_Weights.IMAGENET1K_V1}"
task_lab="${ISAACLAB:-/workspace/isaaclab}"
export PYTHONPATH="${task_repo}/src/robot_learning:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-${task_repo}/outputs/cache/torch}"

"${task_lab}/isaaclab.sh" -p -m humanoid_robot_learning.sim.cargo_data validate "${task_data}"
exec "${task_lab}/isaaclab.sh" -p -m humanoid_robot_learning.train \
  --dataset.repo_id=humanoid/nasa_cargo_left_arm \
  --dataset.root="${task_data}" \
  --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=50 --policy.n_action_steps=25 \
  --policy.pretrained_backbone_weights="${task_weights}" \
  --policy.use_amp=true \
  --batch_size="${task_batch}" --steps="${task_steps}" \
  --num_workers=2 --save_freq=5000 --log_freq=100 --eval_freq=0 \
  --wandb.enable=false --output_dir="${task_output}" "$@"
