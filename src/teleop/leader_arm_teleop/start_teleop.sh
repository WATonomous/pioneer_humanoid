#!/usr/bin/env bash
# One-command host launcher for the five-servo leader arm.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONTAINER_LAUNCHER="/workspace/humanoid/src/teleop/leader_arm_teleop/container_start_teleop.sh"

usage() {
    cat <<'EOF'
Usage: ./src/teleop/leader_arm_teleop/start_teleop.sh [OPTIONS]

Starts the simulation_isaac container, detects the leader arm's USB serial
port, and launches five-joint teleoperation in Isaac Sim. Any unrecognized
options are forwarded to leader_arm_teleop.py.

Examples:
  ./src/teleop/leader_arm_teleop/start_teleop.sh
  ./src/teleop/leader_arm_teleop/start_teleop.sh --port /dev/ttyACM0
  ./src/teleop/leader_arm_teleop/start_teleop.sh --filter-alpha 0.5

Port selection priority:
  1. --port PATH
  2. TELEOP_PORT environment variable
  3. One matching device under /dev/serial/by-id
  4. One /dev/ttyACM* or /dev/ttyUSB* device

Press Ctrl+C to stop Isaac Sim and release the serial port.
EOF
}

die() {
    echo "[leader] ERROR: $*" >&2
    exit 1
}

detect_port() {
    local candidate target
    local -a stable_candidates=()
    local -a raw_candidates=()

    shopt -s nullglob
    for candidate in /dev/serial/by-id/*; do
        target="$(readlink -f "$candidate")"
        if [[ "$target" == /dev/ttyACM* || "$target" == /dev/ttyUSB* ]]; then
            stable_candidates+=("$candidate")
        fi
    done
    shopt -u nullglob

    if ((${#stable_candidates[@]} == 1)); then
        printf '%s\n' "${stable_candidates[0]}"
        return
    fi

    if ((${#stable_candidates[@]} > 1)); then
        echo "[leader] Multiple USB serial devices found:" >&2
        printf '  %s\n' "${stable_candidates[@]}" >&2
        die "re-run with --port PATH to choose the leader arm"
    fi

    shopt -s nullglob
    raw_candidates=(/dev/ttyACM* /dev/ttyUSB*)
    shopt -u nullglob

    if ((${#raw_candidates[@]} == 1)); then
        printf '%s\n' "${raw_candidates[0]}"
        return
    fi

    if ((${#raw_candidates[@]} == 0)); then
        die "no USB serial device found; plug in the leader arm and try again"
    fi

    echo "[leader] Multiple USB serial devices found:" >&2
    printf '  %s\n' "${raw_candidates[@]}" >&2
    die "re-run with --port PATH to choose the leader arm"
}

port="${TELEOP_PORT:-}"
port_was_forwarded=false
forwarded_args=()

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --port)
            (($# >= 2)) || die "--port requires a device path"
            port="$2"
            port_was_forwarded=true
            forwarded_args+=("$1" "$2")
            shift 2
            ;;
        --port=*)
            port="${1#--port=}"
            port_was_forwarded=true
            forwarded_args+=("$1")
            shift
            ;;
        *)
            forwarded_args+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$port" ]]; then
    port="$(detect_port)"
fi
[[ -e "$port" ]] || die "serial device does not exist: $port"

if [[ "$port_was_forwarded" == false ]]; then
    forwarded_args+=(--port "$port")
fi

cd "$REPO_ROOT"
export ACTIVE_MODULES="simulation_isaac"

echo "=== Leader Arm Teleop ==="
echo "[1/2] Leader detected at $port"
echo "[2/2] Starting Isaac Sim (the first launch can take a few minutes)..."
./watod up -d

exec ./watod exec simulation_isaac bash "$CONTAINER_LAUNCHER" "${forwarded_args[@]}"
