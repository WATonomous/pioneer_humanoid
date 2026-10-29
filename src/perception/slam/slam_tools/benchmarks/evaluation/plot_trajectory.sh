#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GROUND_TRUTH.txt ESTIMATED_TRAJECTORY.txt" >&2
  exit 1
fi

GROUND_TRUTH="$(realpath "$1")"
ESTIMATED_TRAJECTORY="$(realpath "$2")"
for file in "$GROUND_TRUTH" "$ESTIMATED_TRAJECTORY"; do
  [[ -s "$file" ]] || { echo "Error: missing or empty trajectory: $file" >&2; exit 1; }
done

if command -v evo_traj >/dev/null 2>&1; then
  EVO_TRAJ="$(command -v evo_traj)"
elif [[ -x /opt/evo-venv/bin/evo_traj ]]; then
  EVO_TRAJ=/opt/evo-venv/bin/evo_traj
else
  echo "Error: evo_traj is not installed or available." >&2
  exit 1
fi

TEMP_GT="$(mktemp)"
trap 'rm -f "$TEMP_GT"' EXIT
awk '/^#/ {print; next} !seen[$1]++' "$GROUND_TRUTH" > "$TEMP_GT"

PLOT_DIR="$(dirname "$GROUND_TRUTH")/trajectory_plots"
mkdir -p "$PLOT_DIR"
"$EVO_TRAJ" tum "$TEMP_GT" "$ESTIMATED_TRAJECTORY" \
  --ref "$TEMP_GT" --sync -a --plot_mode=xyz \
  --save_plot "$PLOT_DIR/trajectory_comparison"

echo "Plots saved in: $PLOT_DIR"
