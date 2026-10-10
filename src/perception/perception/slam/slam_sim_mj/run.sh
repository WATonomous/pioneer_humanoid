#!/usr/bin/env bash
# One unattended SLAM run: start rtabmap, walk the loop, export the map, report.
#
#   ./run.sh [LAPS] [PERIOD_S]        e.g. ./run.sh 2 100
#   RECORD=1 ./run.sh                 also write the walkthrough to demo.mp4
#
# Recording is off by default: encoding runs inside the publish loop and costs about
# 1% of the frame rate, which is harmless for a demo and pointless noise otherwise.
#
# For interactive work run the two halves in separate terminals instead (see
# README.md) -- you get RViz and can watch odometry as it happens.
# NOT `set -u`: /opt/ros/jazzy/setup.bash reads unset variables (AMENT_TRACE_SETUP_FILES)
# and aborts the script on its first line under nounset.
set -o pipefail

LAPS="${1:-2}"
PERIOD="${2:-100}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Find the repo root by walking up to the .git marker, NOT by counting "../.." from
# here. A fixed depth silently resolves to the wrong directory the moment the package
# moves -- which has already happened twice: once when the repo renamed autonomy/ to
# src/, and again when this package was considered for a deeper home under perception/.
# The failure is quiet: outputs/ and .venv are simply looked for somewhere else.
ROOT="$HERE"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/.git" ]; do ROOT="$(dirname "$ROOT")"; done
if [ ! -d "$ROOT/.git" ]; then
    echo "could not find the repo root (no .git above $HERE)" >&2
    exit 1
fi
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs/slam_mj"

source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# SHM for small messages, TCP for large ones. Plain SHM has been seen to abort
# rtabmap on the much larger map-cloud messages; UDPv4 drops image frames.
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-LARGE_DATA}"

mkdir -p "$OUT"

# The scene is generated, not committed (see .gitignore). Build it on first use so a
# fresh checkout works without a documented extra step.
if [ ! -f "$HERE/scene/room.xml" ]; then
    echo "== generating scene (first run) =="
    "$PY" "$HERE/scene/build_room.py"
fi

LAUNCH_PID=""
cleanup() {
    # Kill the launch process first and by PID. Matching only the node names left the
    # `ros2 launch` parent alive on every run -- seven of them accumulated across one
    # session, each holding its nodes' DDS discovery open.
    [ -n "$LAUNCH_PID" ] && kill -TERM "$LAUNCH_PID" 2>/dev/null
    sleep 2
    # Then sweep any node that outlived its parent, matching the INSTALLED BINARY
    # PATHS only. Matching loose names instead ("slam_sim_mj", "rtabmap") also matches
    # the command line of whatever invoked this script -- `./run.sh` mentions
    # slam_sim_mj in its own path -- so the sweep killed its own caller's shell.
    for p in $(pgrep -f "/opt/ros/[a-z]*/lib/rtabmap|/opt/ros/[a-z]*/lib/imu_filter_madgwick"); do
        [ "$p" = "$$" ] || [ "$p" = "$PPID" ] && continue
        kill -TERM "$p" 2>/dev/null
    done
}
trap cleanup EXIT
cleanup; sleep 2

echo "== starting rtabmap =="
ros2 launch "$HERE/launch/slam_sim_mj.launch.py" rviz:=false > "$OUT/rtabmap.log" 2>&1 &
LAUNCH_PID=$!
sleep 10

echo "== walking $LAPS lap(s) at ${PERIOD}s each =="
RECORD_ARGS=()
[ -n "${RECORD:-}" ] && RECORD_ARGS=(--record "$OUT/demo.mp4")

"$PY" "$HERE/mj_slam_publisher.py" --path corridor --laps "$LAPS" --period "$PERIOD" \
    "${RECORD_ARGS[@]}" \
    2>&1 | grep -Ev "Exception ignored|EGLError|^  |Traceback|File \""

sleep 5
echo "== exporting =="
"$PY" "$HERE/export_map.py" "$OUT/map_2d.png"
"$PY" "$HERE/check_map.py" "$OUT/rtabmap.db"
