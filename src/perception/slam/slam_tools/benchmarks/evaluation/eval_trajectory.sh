#!/usr/bin/env bash

set -euo pipefail

# checks if number of args is not 2
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GROUND_TRUTH.txt ESTIMATED_TRAJECTORY.txt" >&2
  exit 1
fi

# sets the files according to arguments
GROUND_TRUTH="$1"
ESTIMATED_TRAJECTORY="$2"


# Checks if file exist/non-zero
for trajectory in "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY"; do
  if [[ ! -s "$trajectory" ]]; then
    echo "Error: trajectory file not found or empty: $trajectory" >&2
    exit 1
  fi
done


# Checks if evo_ape is installed/finds it path
if command -v evo_ape >/dev/null 2>&1; then
  EVO_APE="$(command -v evo_ape)"
elif [[ -x /opt/evo-venv/bin/evo_ape ]]; then
  EVO_APE=/opt/evo-venv/bin/evo_ape
else
  echo "Error: evo_ape is not installed or available." >&2
  exit 1
fi

# Calcs position error
POSITION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY" -a 2>&1)" || {
  echo "$POSITION_OUTPUT" >&2
  exit 1
}

# Calcs rotation error
ROTATION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY" -a -r angle_deg 2>&1)" || {
  echo "$ROTATION_OUTPUT" >&2
  exit 1
}

# Calcs relative pose errors
RPE_TRANSLATION_OUTPUT="$(evo_rpe tum "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY" -a -r trans_part 2>&1)" || { echo "$RPE_TRANSLATION_OUTPUT" >&2; exit 1; }
RPE_ROTATION_OUTPUT="$(evo_rpe tum "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY" -a -r angle_deg 2>&1)" || { echo "$RPE_ROTATION_OUTPUT" >&2; exit 1; }

# Gets the values
POSITION_RMSE="$(printf '%s\n' "$POSITION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"
ROTATION_RMSE="$(printf '%s\n' "$ROTATION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"
RPE_TRANSLATION_RMSE="$(printf '%s\n' "$RPE_TRANSLATION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"
RPE_ROTATION_RMSE="$(printf '%s\n' "$RPE_ROTATION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"

# Prints
printf 'Positional RMSE: %s m\n' "$POSITION_RMSE"
printf 'Rotational RMSE: %s deg\n' "$ROTATION_RMSE"
printf 'RPE Translational RMSE: %s m\n' "$RPE_TRANSLATION_RMSE"
printf 'RPE Rotational RMSE: %s deg\n' "$RPE_ROTATION_RMSE"
