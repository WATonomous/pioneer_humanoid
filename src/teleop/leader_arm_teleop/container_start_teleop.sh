#!/usr/bin/env bash
# Runs inside the simulation_isaac container. Keep Isaac in the foreground so
# Ctrl+C reaches it and its finally block disables leader torque on exit.
set -euo pipefail

TELEOP_DIR="/workspace/humanoid/src/teleop/leader_arm_teleop"
ISAACLAB="${ISAACLAB:-/workspace/isaaclab}"
port="${TELEOP_PORT:-/dev/ttyACM0}"

args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
    case "${args[$index]}" in
        --port)
            ((index + 1 < ${#args[@]})) || {
                echo "[leader] ERROR: --port requires a device path" >&2
                exit 1
            }
            port="${args[$((index + 1))]}"
            ((index += 1))
            ;;
        --port=*)
            port="${args[$index]#--port=}"
            ;;
    esac
done

[[ -e "$port" ]] || {
    echo "[leader] ERROR: serial device is not visible inside the container: $port" >&2
    exit 1
}

if pgrep -f '[l]eader_arm_teleop.py' >/dev/null 2>&1; then
    echo "[leader] ERROR: leader_arm_teleop.py is already running." >&2
    echo "[leader] Stop the existing process before launching another." >&2
    exit 1
fi

cd "$TELEOP_DIR"
echo "[leader] Launching with serial port $port"
echo "[leader] Press Ctrl+C to stop. Leader torque remains OFF."

exec "$ISAACLAB/isaaclab.sh" -p leader_arm_teleop.py "$@"
