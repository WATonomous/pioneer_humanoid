#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash

set -euo pipefail

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
  FORCE=true
  shift
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 [--force] /path/to/rgbd_dataset" >&2
  exit 1
fi

DATASET_PATH="$(realpath "$1")"
GROUND_TRUTH="$DATASET_PATH/groundtruth.txt"
ESTIMATED_POSES="$DATASET_PATH/rtabmap_poses.txt"
DATABASE="$DATASET_PATH/rtabmap.db"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$GROUND_TRUTH" ]]; then
  echo "Error: ground truth not found: $GROUND_TRUTH" >&2
  exit 1
fi

if [[ ! -d "$DATASET_PATH/rgb_sync" || ! -d "$DATASET_PATH/depth_sync" ]]; then
  echo "Associating RGB and depth frames..." >&2
  python3 "$SCRIPT_DIR/associate_rgbd.py" "$DATASET_PATH" --force >&2
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

# The dataset runner can finish successfully but fail to export poses when the
# database contains disconnected map segments. Recover its connected graph.
if [[ ! -s "$ESTIMATED_POSES" ]]; then
  if [[ ! -f "$DATABASE" ]]; then
    echo "Error: RTAB-Map did not create a trajectory or database." >&2
    exit 1
  fi

  echo "Exporting the connected trajectory from rtabmap.db..." >&2
  rtabmap-export \
    --poses \
    --poses_format 1 \
    --opt 2 \
    --output rtabmap \
    --output_dir "$DATASET_PATH" \
    "$DATABASE" >&2
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
printf 'Rotational RMSE: %s deg\n' "$ROTATION_RMSE"

if [[ -f "$DATABASE" ]] && command -v rtabmap-info >/dev/null 2>&1; then
  if DATABASE_INFO="$(rtabmap-info "$DATABASE" 2>/dev/null)"; then
    TOTAL_NODES="$(printf '%s\n' "$DATABASE_INFO" | awk '$1 == "WM:" {print $2; exit}')"
    CONNECTED_POSES="$(printf '%s\n' "$DATABASE_INFO" | awk '$1 == "Optimized" && $2 == "graph:" {print $3; exit}')"
    MAP_COUNTS="$(printf '%s\n' "$DATABASE_INFO" | awk '$1 == "Maps" && $2 == "in" && $3 == "graph:" {print $4; exit}')"

    if [[ "$TOTAL_NODES" =~ ^[0-9]+$ && "$CONNECTED_POSES" =~ ^[0-9]+$ && "$TOTAL_NODES" -gt 0 ]]; then
      EXCLUDED_POSES=$((TOTAL_NODES - CONNECTED_POSES))
      CONNECTED_PERCENT="$(awk -v connected="$CONNECTED_POSES" -v total="$TOTAL_NODES" 'BEGIN {printf "%.1f", 100 * connected / total}')"
      EXCLUDED_PERCENT="$(awk -v excluded="$EXCLUDED_POSES" -v total="$TOTAL_NODES" 'BEGIN {printf "%.1f", 100 * excluded / total}')"

      printf 'Connected SLAM nodes: %s/%s (%s%%)\n' "$CONNECTED_POSES" "$TOTAL_NODES" "$CONNECTED_PERCENT"
      printf 'Excluded SLAM nodes: %s (%s%%)\n' "$EXCLUDED_POSES" "$EXCLUDED_PERCENT"
    fi

    if [[ "$MAP_COUNTS" =~ ^[0-9]+/[0-9]+$ ]]; then
      printf 'Connected map segments: %s\n' "$MAP_COUNTS"
    fi
  fi
fi

printf '\n'
