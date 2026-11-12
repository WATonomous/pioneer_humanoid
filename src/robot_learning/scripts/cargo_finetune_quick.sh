#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
task_data="${CARGO_DATASET_ROOT:-${task_repo}/datasets/nasa_cargo/randomized_v2/aggregate}"
task_output="${CARGO_OUTPUT_DIR:-${task_repo}/outputs/train/nasa_cargo_act_randomized_v2}"
task_policy="${CARGO_POLICY_PATH:-${task_repo}/outputs/train/nasa_cargo_act_v1/checkpoints/100000/pretrained_model}"
task_lab="${ISAACLAB:-/workspace/isaaclab}"
export PYTHONPATH="${task_repo}/src/robot_learning:${PYTHONPATH:-}"
"${task_lab}/isaaclab.sh" -p -m humanoid_robot_learning.sim.cargo_data validate "${task_data}"
exec "${task_lab}/isaaclab.sh" -p -m humanoid_robot_learning.sim.cargo_finetune \
  --dataset.repo_id=humanoid/nasa_cargo_left_arm --dataset.root="${task_data}" \
  --dataset.video_backend=pyav --policy.path="${task_policy}" \
  --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=true \
  --policy.pretrained_backbone_weights=null \
  --policy.optimizer_lr=0.000005 --policy.optimizer_lr_backbone=0.000005 \
  --steps="${CARGO_STEPS:-5000}" --batch_size=4 --num_workers=2 \
  --save_freq=2000 --log_freq=100 --eval_freq=0 --wandb.enable=false \
  --output_dir="${task_output}" "$@"
