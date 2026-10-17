#!/usr/bin/env bash
# Train Diffusion Policy (LeRobot) on a tidy_table dataset, the same inputs as train_act.sh.
#
#   train_diffusion.sh <dataset root> <output dir> [extra lerobot-train flags...]
#
# Unlike ACT it can commit to one of several valid grasps (a box's four sides) instead of averaging them.
# Timing at 25 fps: sees the last 2 frames, predicts 32 actions (1.3 s), runs 12 (0.5 s) before re-planning.
# Images are resized to 240x320 and randomly cropped to 95% while training. DDIM with 10 denoising steps at
# inference keeps eval_sim.py / live_view.py near real time. UNet down_dims 256/512/1024 (the default 512/1024/2048
# is 278M params, ~3.3 GB per checkpoint with optimizer state). Defaults are for one GPU.
set -euo pipefail
ROOT=${1:?dataset root (the folder with meta/ data/ videos/)}
OUT=${2:?output dir (must not exist yet)}
shift 2
REPO_ID=$(python -c "import json,sys; print(json.load(open(sys.argv[1]+'/meta/info.json')).get('robot_type','local/tidy_table'))" "$ROOT")
exec lerobot-train \
  --dataset.repo_id="local/${REPO_ID}" \
  --dataset.root="$ROOT" \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.horizon=32 \
  --policy.n_action_steps=12 \
  --policy.drop_n_last_frames=19 \
  --policy.resize_shape="[240,320]" \
  --policy.crop_ratio=0.95 \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=10 \
  --policy.down_dims="[256,512,1024]" \
  --output_dir="$OUT" \
  --job_name=diffusion_tidy_table \
  --steps=150000 \
  --batch_size=32 \
  --save_freq=25000 \
  --log_freq=200 \
  --wandb.enable=false \
  "$@"
