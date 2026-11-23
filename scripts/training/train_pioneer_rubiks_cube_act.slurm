#!/usr/bin/env bash
# Submit from the repository root: sbatch scripts/training/train_pioneer_rubiks_cube_act.slurm
# Override TRAIN_STORAGE to choose another persistent shared storage directory.
#SBATCH --job-name=pioneer-rubiks-act
#SBATCH --output=slurm-act-%j.out
#SBATCH --error=slurm-act-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:rtx_3090:1
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TRAIN_STORAGE="${TRAIN_STORAGE:-/mnt/wato-drive2/$USER/pioneer-training}"
export VENV_DIR="${VENV_DIR:-$TRAIN_STORAGE/venv-lerobot043}"
export HF_HOME="$TRAIN_STORAGE/huggingface"
export TORCH_HOME="$TRAIN_STORAGE/torch"
export OMP_NUM_THREADS=8
export TRAIN_PYTHON="$VENV_DIR/bin/python"
export OUTPUT_DIR="${OUTPUT_DIR:-$TRAIN_STORAGE/runs/act-$SLURM_JOB_ID}"
mkdir -p "$TRAIN_STORAGE"
nvidia-smi
if [[ "${SETUP_ENV:-0}" == 1 ]]; then
  bash scripts/training/setup_watcloud_act.sh
fi
if [[ ! -x "$TRAIN_PYTHON" ]]; then
  echo 'Training environment missing. Submit once with SETUP_ENV=1.' >&2
  exit 1
fi
"$TRAIN_PYTHON" - <<'CHECK'
import torch
import torchvision
import av
from importlib.metadata import version
assert torch.cuda.is_available(), 'CUDA is unavailable in this allocation'
print('LeRobot', version('lerobot'), 'torch', torch.__version__, 'torchvision', torchvision.__version__)
print('GPU:', torch.cuda.get_device_name(0), 'PyAV:', av.__version__, flush=True)
CHECK
bash scripts/training/train_pioneer_rubiks_cube_act.sh --dataset.video_backend=pyav --log_freq="${LOG_FREQ:-100}"
test -s "$OUTPUT_DIR/checkpoints/last/pretrained_model/model.safetensors"
echo "Training completed; checkpoint: $OUTPUT_DIR/checkpoints/last/pretrained_model"
