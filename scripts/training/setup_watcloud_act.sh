#!/usr/bin/env bash
# Run inside a SLURM allocation; no Docker, Isaac, or administrator access.
set -euo pipefail
TRAIN_STORAGE="${TRAIN_STORAGE:-/mnt/wato-drive2/$USER/pioneer-training}"
VENV_DIR="${VENV_DIR:-$TRAIN_STORAGE/venv-lerobot043}"
export PIP_CACHE_DIR="$TRAIN_STORAGE/pip-cache"
# Compute nodes have a very small /tmp; pip stages large CUDA wheels here.
export TMPDIR="$TRAIN_STORAGE/tmp-${SLURM_JOB_ID:-setup}"
mkdir -p "$TRAIN_STORAGE" "$TMPDIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install --no-compile torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
"$VENV_DIR/bin/python" -m pip install --no-compile lerobot==0.4.3 torchcodec==0.5
"$VENV_DIR/bin/python" -m pip check
"$VENV_DIR/bin/python" -m pip freeze > "$TRAIN_STORAGE/requirements-resolved.txt"
