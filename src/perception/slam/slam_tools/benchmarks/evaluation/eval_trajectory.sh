#!/usr/bin/env bash

set -euo pipefail

# checks if number of args is less than 2
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GROUND_TRUTH.txt ESTIMATED_TRAJECTORY.txt [--align-scale] [--csv]" >&2
  exit 1
fi

# sets the files according to arguments
GROUND_TRUTH="$1"
ESTIMATED_TRAJECTORY="$2"

ALIGNMENT_ARGS=(-a)
CSV_OUTPUT=0
for option in "${@:3}"; do
  case "$option" in
    --align-scale) ALIGNMENT_ARGS=(-as) ;;
    --csv) CSV_OUTPUT=1 ;;
    *) echo "Error: unknown option: $option" >&2; exit 1 ;;
  esac
done


# Checks if file exist/non-zero
for trajectory in "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY"; do
  if [[ ! -s "$trajectory" ]]; then
    echo "Error: trajectory file not found or empty: $trajectory" >&2
    exit 1
  fi
done

# EVO requires strictly increasing timestamps. Keep the first row for each
# timestamp in a temporary copy without changing the original ground truth.
TEMP_GT="$(mktemp)"
trap 'rm -f "$TEMP_GT"' EXIT
awk '/^#/ {print; next} !seen[$1]++' "$GROUND_TRUTH" > "$TEMP_GT"
GROUND_TRUTH="$TEMP_GT"


# Checks if evo_ape is installed/finds it path
if command -v evo_ape >/dev/null 2>&1; then
  EVO_APE="$(command -v evo_ape)"
elif [[ -x /opt/evo-venv/bin/evo_ape ]]; then
  EVO_APE=/opt/evo-venv/bin/evo_ape
else
  echo "Error: evo_ape is not installed or available." >&2
  exit 1
fi

if command -v evo_rpe >/dev/null 2>&1; then
  EVO_RPE="$(command -v evo_rpe)"
elif [[ -x /opt/evo-venv/bin/evo_rpe ]]; then
  EVO_RPE=/opt/evo-venv/bin/evo_rpe
else
  echo "Error: evo_rpe is not installed or available." >&2
  exit 1
fi

# Calcs position error
POSITION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY" "${ALIGNMENT_ARGS[@]}" 2>&1)" || {
  echo "$POSITION_OUTPUT" >&2
  exit 1
}

# Calcs rotation error
ROTATION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY" "${ALIGNMENT_ARGS[@]}" -r angle_deg 2>&1)" || {
  echo "$ROTATION_OUTPUT" >&2
  exit 1
}

# Calcs relative pose errors over a fixed one-metre distance so RPE is
# comparable between frame-based and keyframe trajectories.
RPE_TRANSLATION_OUTPUT="$($EVO_RPE tum "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY" "${ALIGNMENT_ARGS[@]}" -r trans_part --delta 1 --delta_unit m 2>&1)" || { echo "$RPE_TRANSLATION_OUTPUT" >&2; exit 1; }
RPE_ROTATION_OUTPUT="$($EVO_RPE tum "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY" "${ALIGNMENT_ARGS[@]}" -r angle_deg --delta 1 --delta_unit m 2>&1)" || { echo "$RPE_ROTATION_OUTPUT" >&2; exit 1; }

# Gets the values
POSITION_RMSE="$(printf '%s\n' "$POSITION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"
ROTATION_RMSE="$(printf '%s\n' "$ROTATION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"
RPE_TRANSLATION_RMSE="$(printf '%s\n' "$RPE_TRANSLATION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"
RPE_ROTATION_RMSE="$(printf '%s\n' "$RPE_ROTATION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"

# Prints
if [[ "$CSV_OUTPUT" -eq 1 ]]; then
  printf '%s,%s,%s,%s\n' \
    "$POSITION_RMSE" "$ROTATION_RMSE" \
    "$RPE_TRANSLATION_RMSE" "$RPE_ROTATION_RMSE"
else
  printf 'Positional RMSE: %s m\n' "$POSITION_RMSE"
  printf 'Rotational RMSE: %s deg\n' "$ROTATION_RMSE"
  printf 'RPE Translational RMSE: %s m\n' "$RPE_TRANSLATION_RMSE"
  printf 'RPE Rotational RMSE: %s deg\n' "$RPE_ROTATION_RMSE"
fi
