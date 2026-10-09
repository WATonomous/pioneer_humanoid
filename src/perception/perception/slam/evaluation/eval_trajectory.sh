#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GROUND_TRUTH.txt ESTIMATED_TRAJECTORY.txt" >&2
  exit 1
fi

GROUND_TRUTH="$1"
ESTIMATED_TRAJECTORY="$2"

for trajectory in "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY"; do
  if [[ ! -s "$trajectory" ]]; then
    echo "Error: trajectory file not found or empty: $trajectory" >&2
    exit 1
  fi
done

if command -v evo_ape >/dev/null 2>&1; then
  EVO_APE="$(command -v evo_ape)"
elif [[ -x /opt/evo-venv/bin/evo_ape ]]; then
  EVO_APE=/opt/evo-venv/bin/evo_ape
else
  echo "Error: evo_ape is not installed or available." >&2
  exit 1
fi

POSITION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY" -a 2>&1)" || {
  echo "$POSITION_OUTPUT" >&2
  exit 1
}

ROTATION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY" -a -r angle_deg 2>&1)" || {
  echo "$ROTATION_OUTPUT" >&2
  exit 1
}

POSITION_RMSE="$(printf '%s\n' "$POSITION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"
ROTATION_RMSE="$(printf '%s\n' "$ROTATION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"

printf 'Positional RMSE: %s m\n' "$POSITION_RMSE"
printf 'Rotational RMSE: %s deg\n' "$ROTATION_RMSE"
