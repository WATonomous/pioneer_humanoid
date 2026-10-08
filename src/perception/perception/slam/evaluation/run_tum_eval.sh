#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash

set -euo pipefail

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
  FORCE=true
  shift
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 [--force] /path/to/tum_dataset" >&2
  exit 1
fi

DATASET_PATH="$1"
GROUND_TRUTH="$DATASET_PATH/groundtruth.txt"
ESTIMATED_POSES="$DATASET_PATH/rtabmap_poses.txt"

if [[ ! -f "$GROUND_TRUTH" ]]; then
  echo "Error: ground truth not found: $GROUND_TRUTH" >&2
  exit 1
fi

POSE_COUNT=0
if [[ -f "$ESTIMATED_POSES" ]]; then
  POSE_COUNT="$(awk '!/^#/ {count++} END {print count+0}' "$ESTIMATED_POSES")"
fi

if [[ "$FORCE" == true || "$POSE_COUNT" -lt 3 ]]; then
  echo "Running RTAB-Map SLAM..." >&2
  rtabmap-rgbd_dataset "$DATASET_PATH" >&2
else
  echo "Reusing existing trajectory ($POSE_COUNT poses): $ESTIMATED_POSES" >&2
fi

if command -v evo_ape >/dev/null 2>&1; then
  EVO_APE="$(command -v evo_ape)"
elif [[ -x /opt/evo-venv/bin/evo_ape ]]; then
  EVO_APE=/opt/evo-venv/bin/evo_ape
else
  echo "Error: evo_ape is not installed or available." >&2
  exit 1
fi

if ! POSITION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_POSES" -a 2>&1)"; then
  echo "$POSITION_OUTPUT" >&2
  exit 1
fi
POSITION_RMSE="$(printf '%s\n' "$POSITION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"

if ! ROTATION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_POSES" -a -r angle_deg 2>&1)"; then
  echo "$ROTATION_OUTPUT" >&2
  exit 1
fi
ROTATION_RMSE="$(printf '%s\n' "$ROTATION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"

if [[ -z "$POSITION_RMSE" || -z "$ROTATION_RMSE" ]]; then
  echo "Error: evo did not report both RMSE values." >&2
  exit 1
fi

printf '\n\nPositional RMSE: %s m\n' "$POSITION_RMSE"
printf 'Rotational RMSE: %s deg\n\n' "$ROTATION_RMSE"
