#!/usr/bin/env bash
# Phase 0 of the RL-token experiment on a single-GPU cloud box (e.g. an NVIDIA Brev L4 instance, Ubuntu):
# install, smoke test, train Pioneer-1 (pioneer1) on the synthetic peg_insert demos, evaluate it, upload the checkpoint
# and results to a private Hugging Face model repo, then (optionally) power the machine off so it stops billing.
#
#   hf auth login     # once, so the private dataset, the gated DINOv3 encoder and the upload work
#   AUTO_SHUTDOWN=1 bash run_phase0.sh              # or run it inside tmux; logs to ~/phase0/phase0.log
#
# Env vars: STEPS (default 10000: 40 demos, so far fewer than the README's 40k on 600), BATCH (default 32), SAVE_FREQ
# (checkpoint every N steps, default 2500), EVAL_STEPS (earlier checkpoints to eval besides the last, default 5000),
# EPISODES (eval, default 40), GRAD_CKPT=true (about half the GPU memory, if the smoke test runs out),
# HF_MODEL_REPO (default zabsy/pioneer1_peg_40),
# EVAL_STEPS=none evals only the last checkpoint. AUTO_SHUTDOWN=1 powers off at the end, success or failure (a stopped
# box may still bill a little for disk); leave it off when someone will stop the box by hand after more work.
# Rerunnable: a finished training run is reused, so a failed eval does not retrain.
set -euo pipefail

STEPS=${STEPS:-10000}
EPISODES=${EPISODES:-40}
HF_MODEL_REPO=${HF_MODEL_REPO:-zabsy/pioneer1_peg_40}
DATASET=zabsy/pioneer_8dof_peg_insert_synthetic
BRANCH=rlt-peg-insert
WORK=$HOME/phase0
REPO=$WORK/pioneer_humanoid
VENV=$WORK/venv
OUT=$WORK/outputs/pioneer1_peg_40
mkdir -p "$WORK"
exec > >(tee -a "$WORK/phase0.log") 2>&1

finish() {
  rc=$?
  [ -n "${GPU_LOG_PID:-}" ] && kill "$GPU_LOG_PID" 2>/dev/null
  echo "[phase0] $(date +%T) exit code $rc"
  if [ "${AUTO_SHUTDOWN:-0}" = 1 ]; then
    echo "[phase0] powering off in 60 s (Ctrl-C to cancel)"; sleep 60; sudo shutdown -h now
  fi
}
trap finish EXIT
step() { echo; echo "[phase0] $(date +%T) === $* ==="; }

step "system packages"
nvidia-smi --query-gpu=name,memory.total --format=csv
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-venv python3-dev libegl1 libgl1 libosmesa6 ffmpeg >/dev/null
# GPU rendering: Brev's GCP images ship only the compute half of the NVIDIA driver (libnvidia-compute-*), so EGL falls
# back to Mesa on the CPU and an eval episode takes minutes instead of seconds. Add the GL/EGL half at the same version.
if [ ! -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  compute=$(dpkg-query -W -f '${Package}=${Version}\n' 'libnvidia-compute-*' | head -1)
  [ -n "$compute" ] || { echo "[phase0] no libnvidia-compute package: install the NVIDIA GL libraries by hand"; exit 1; }
  sudo apt-get install -y -qq "${compute/libnvidia-compute-/libnvidia-gl-}" >/dev/null
fi

step "code ($BRANCH)"
if [ -d "$REPO/.git" ]; then git -C "$REPO" fetch -q origin "$BRANCH" && git -C "$REPO" checkout -q "$BRANCH" && git -C "$REPO" pull -q
else git clone -q -b "$BRANCH" https://github.com/WATonomous/humanoid.git "$REPO"; fi

step "python env"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q -U pip
pip install -q -e "$REPO/src/robot_learning[pioneer1,sim]" -e "$REPO/src/simulation/mujoco_scenes" -e "$REPO/src/pioneer_humanoid" mujoco scipy
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
hf auth whoami || { echo "[phase0] not logged in to Hugging Face: run 'hf auth login' and rerun"; exit 1; }
# GPU memory and load every 30 s for the whole run, to size the batch for the RL phases.
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 30 > "$WORK/gpu.log" &
GPU_LOG_PID=$!

train() {  # train <steps> <output dir>
  python -m humanoid_robot_learning.train \
    --policy.type=pioneer1 --dataset.repo_id="$DATASET" --dataset.video_backend=pyav \
    --steps="$1" --batch_size="${BATCH:-32}" --save_freq="${SAVE_FREQ:-2500}" \
    --policy.scheduler_warmup_steps=$(( $1 / 10 )) --policy.grad_checkpoint="${GRAD_CKPT:-false}" \
    --policy.device=cuda --policy.push_to_hub=false --output_dir="$2"
}

cd "$REPO/src/robot_learning"
if [ -d "$OUT/checkpoints/last/pretrained_model" ]; then
  step "training already done: reusing $OUT"
else
  step "smoke test (100 steps)"
  rm -rf "$WORK/outputs/smoke"; train 100 "$WORK/outputs/smoke"; rm -rf "$WORK/outputs/smoke"
  echo "[phase0] peak GPU memory so far: $(cut -d, -f2 "$WORK/gpu.log" | sort -n | tail -1) MiB"
  step "train $STEPS steps"
  rm -rf "$OUT"; time train "$STEPS" "$OUT"
fi

# Eval the final checkpoint and an earlier one (EVAL_STEPS), to see whether more training helped or overfit.
for s in ${EVAL_STEPS:-5000} last; do
  if [ "$s" = last ]; then ckpt="$OUT/checkpoints/last"; else ckpt=$(ls -d "$OUT"/checkpoints/*"$s" 2>/dev/null | head -1 || true); fi
  [ -n "$ckpt" ] && [ -d "$ckpt/pretrained_model" ] || { echo "[phase0] no checkpoint for step $s, skipping"; continue; }
  step "eval step $s: $EPISODES episodes on held-out seeds 1000+"
  MUJOCO_GL=egl python -m humanoid_robot_learning.sim.evaluate --scene peg_insert \
    --policy "$ckpt/pretrained_model" --episodes "$EPISODES" --seed 1000 --max_s 30 \
    --video_dir "$OUT/eval_videos_$s" 2>&1 | tee "$OUT/eval_$s.txt"
done

step "upload checkpoint + results to $HF_MODEL_REPO (private)"
cp "$WORK/phase0.log" "$WORK/gpu.log" "$OUT/"
python - "$OUT" "$HF_MODEL_REPO" <<'EOF'
import sys
from huggingface_hub import HfApi
out, repo = sys.argv[1:]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path=f"{out}/checkpoints/last/pretrained_model", path_in_repo="pretrained_model", repo_id=repo,
                  commit_message="Pioneer-1 on 40 synthetic peg_insert demos")
api.upload_folder(folder_path=out, repo_id=repo, allow_patterns=["eval_*.txt", "phase0.log", "gpu.log", "eval_videos_*/*"],
                  commit_message="Phase 0 eval results")
print(f"uploaded to https://huggingface.co/{repo}")
EOF

step "done"
for f in "$OUT"/eval_*.txt; do echo "--- $f"; tail -n 3 "$f"; done
