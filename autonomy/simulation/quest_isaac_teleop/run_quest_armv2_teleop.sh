#!/usr/bin/env bash
# Launch Quest armWithStand (wato_arm_v2) teleop (both arms = Differential
# IK/DLS fingertip tracking) inside the simulation_isaac container.
#
# Run this from the repo root (on the host):
#   ./watod -t simulation_isaac_dev
#
# Then inside the container:
#   cd /workspace/humanoid/autonomy/simulation/quest_isaac_teleop
#   ./run_quest_armv2_teleop.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_LAB="${ISAAC_LAB:-/workspace/isaaclab}"

export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/../Teleop/keyboard_based_teleoperation:${SCRIPT_DIR}/../Humanoid_Wato:${PYTHONPATH}"


exec "${ISAAC_LAB}/isaaclab.sh" -p \
  "${SCRIPT_DIR}/run_quest_armv2_teleop.py" \
  --device cpu \
  "$@"
