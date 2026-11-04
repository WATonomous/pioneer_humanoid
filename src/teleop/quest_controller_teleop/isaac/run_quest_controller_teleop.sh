#!/usr/bin/env bash
# Launch the Quest Touch controller scene inside the simulation_isaac container.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_LAB="${ISAAC_LAB:-/workspace/isaaclab}"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/../../keyboard_teleop:${PYTHONPATH:-}"
exec "${ISAAC_LAB}/isaaclab.sh" -p "${SCRIPT_DIR}/run_quest_controller_teleop.py" \
  --device cuda --enable_cameras "$@"
