#!/usr/bin/env bash
# Phase 0 of the RL-token experiment on a single-GPU cloud box (e.g. an NVIDIA Brev L4 instance, Ubuntu):
# install, smoke test, train humanoid_dit on the synthetic peg_insert demos, evaluate it, upload the checkpoint and
# results to a private Hugging Face model repo, then (optionally) power the machine off so it stops billing.
#
#   hf auth login                                   # once, so the private dataset and the upload work
#   AUTO_SHUTDOWN=1 bash run_phase0.sh              # or run it inside tmux; logs to ~/phase0/phase0.log
#
# Env vars: STEPS (default 25000), EPISODES (eval, default 40), HF_MODEL_REPO (default zabsy/dit_peg_40),
# AUTO_SHUTDOWN=1 powers off at the end, success or failure (a stopped box may still bill a little for disk).
# Rerunnable: a finished training run is reused, so a failed eval does not retrain.
set -euo pipefail

STEPS=${STEPS:-25000}
EPISODES=${EPISODES:-40}
HF_MODEL_REPO=${HF_MODEL_REPO:-zabsy/dit_peg_40}
DATASET=zabsy/pioneer_8dof_peg_insert_synthetic
BRANCH=claude/lucid-fermi-fl708u
WORK=$HOME/phase0
REPO=$WORK/pioneer_humanoid
VENV=$WORK/venv
OUT=$WORK/outputs/dit_peg_40
mkdir -p "$WORK"
exec > >(tee -a "$WORK/phase0.log") 2>&1

finish() {
  rc=$?
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

step "code ($BRANCH)"
if [ -d "$REPO/.git" ]; then git -C "$REPO" fetch -q origin "$BRANCH" && git -C "$REPO" checkout -q "$BRANCH" && git -C "$REPO" pull -q
else git clone -q -b "$BRANCH" https://github.com/WATonomous/pioneer_humanoid.git "$REPO"; fi

step "python env"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q -U pip
pip install -q -e "$REPO/src/robot_learning[dit,sim]" -e "$REPO/src/simulation/mujoco_scenes" mujoco scipy
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
hf auth whoami || { echo "[phase0] not logged in to Hugging Face: run 'hf auth login' and rerun"; exit 1; }

train() {  # train <steps> <output dir>
  python -m humanoid_robot_learning.train \
    --policy.type=humanoid_dit --dataset.repo_id="$DATASET" \
    --steps="$1" --batch_size=6 --policy.device=cuda --policy.push_to_hub=false --output_dir="$2"
}

cd "$REPO/src/robot_learning"
if [ -d "$OUT/checkpoints/last/pretrained_model" ]; then
  step "training already done: reusing $OUT"
else
  step "smoke test (200 steps)"
  rm -rf "$WORK/outputs/dit_smoke"; train 200 "$WORK/outputs/dit_smoke"
  step "train $STEPS steps"
  rm -rf "$OUT"; time train "$STEPS" "$OUT"
fi

step "eval: $EPISODES episodes on held-out seeds 1000+"
MUJOCO_GL=egl python -m humanoid_robot_learning.sim.evaluate --scene peg_insert \
  --policy "$OUT/checkpoints/last/pretrained_model" --episodes "$EPISODES" --seed 1000 \
  --video_dir "$OUT/eval_videos" 2>&1 | tee "$OUT/eval.txt"

step "upload checkpoint + results to $HF_MODEL_REPO (private)"
cp "$WORK/phase0.log" "$OUT/phase0.log"
python - "$OUT" "$HF_MODEL_REPO" <<'EOF'
import sys
from huggingface_hub import HfApi
out, repo = sys.argv[1:]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path=f"{out}/checkpoints/last/pretrained_model", path_in_repo="pretrained_model", repo_id=repo,
                  commit_message="humanoid_dit on 40 synthetic peg_insert demos")
api.upload_folder(folder_path=out, repo_id=repo, allow_patterns=["eval.txt", "phase0.log", "eval_videos/*"],
                  commit_message="Phase 0 eval results")
print(f"uploaded to https://huggingface.co/{repo}")
EOF

step "done"
tail -n 5 "$OUT/eval.txt"
