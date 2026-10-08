#!/usr/bin/env bash
# Train ACT (LeRobot) on a tidy_table dataset recorded by the leader teleop or scripted_demos.py.
#
#   train_act.sh <dataset root> <output dir> [extra lerobot-train flags...]
#
# Inputs the policy sees: observation.state, observation.images.* (top, wrist_left) and
# observation.environment_state (which object is next). leader_* and subtask_index are not observation.*,
# so they are never inputs. Defaults are for one GPU; on CPU add --policy.device=cpu --batch_size=2.
# No internet to download.pytorch.org (for the ImageNet ResNet18)? add --policy.pretrained_backbone_weights=null.
set -euo pipefail
ROOT=${1:?dataset root (the folder with meta/ data/ videos/)}
OUT=${2:?output dir (must not exist yet)}
shift 2
REPO_ID=$(python -c "import json,sys; print(json.load(open(sys.argv[1]+'/meta/info.json')).get('robot_type','local/tidy_table'))" "$ROOT")
exec lerobot-train \
  --dataset.repo_id="local/${REPO_ID}" \
  --dataset.root="$ROOT" \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --output_dir="$OUT" \
  --job_name=act_tidy_table \
  --steps=100000 \
  --batch_size=8 \
  --save_freq=10000 \
  --log_freq=200 \
  --wandb.enable=false \
  "$@"
