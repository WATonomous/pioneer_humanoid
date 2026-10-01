#!/bin/bash
# Quest VR -> Isaac Sim teleop, single-script spin-up. Brings up the
# simulation_isaac container, sets up ADB port tunnels, and launches
# quest_teleop_node + webxr_server.py + the Isaac Sim IK script inside it
# (container_start_teleop.sh auto-detects which IK script exists on the
# current branch). See src/teleop/quest_isaac_teleop/README.md for
# the full manual step-by-step this consolidates, and for one-time setup
# (image build, SSL certs, Quest developer mode).
set -e

# Script lives in src/teleop/quest_isaac_teleop/ ; walk up to the repo root.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

# Exported so it applies to every ./watod call below, not just the first --
# ./watod otherwise falls back to whatever ACTIVE_MODULES is set to in
# watod-config.local.sh, which may not include simulation_isaac.
export ACTIVE_MODULES="simulation_isaac"

echo "=== Quest Teleop ==="
echo "[1/3] Starting simulation_isaac container..."
./watod up -d

echo "[2/3] Setting up ADB port tunnels..."
if ! adb devices | grep -q "device$"; then
    echo "  WARNING: no Quest detected by adb. Plug in the Quest over USB and"
    echo "  ensure developer mode is enabled, then re-run."
    exit 1
fi
adb reverse tcp:8443 tcp:8443
adb reverse tcp:9090 tcp:9090
echo "  Tunnels ready (8443 + 9090)."

echo "[3/3] Launching teleop pipeline inside container..."
echo "  Isaac Sim takes 1-3 min to load. Wait for:"
echo "  [Quest] Ready. Waiting for /quest_teleop messages."
echo ""
./watod exec simulation_isaac bash \
    /workspace/humanoid/src/teleop/quest_isaac_teleop/container_start_teleop.sh "$@"
