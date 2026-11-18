# Send the LegPose test poses to joint_command (legpose_test_script.py, run inside the joint_command
# container so you do not have to enter it or remember its paths), record telemetry while it runs,
# then plot it into the run folder.
#
#   tools/legpose_test.sh --check    # verify the setup. Publishes NOTHING.
#   tools/legpose_test.sh            # run the script; record; plot
#
# Output: outputs/gl40_bench/<stamp>_legpose-test/ with angle.png, velocity.png, tracking.png,
# torque.png, temperature.png, summary.md, telemetry.csv and run.json.
#
# The recorder (telemetry_record.py --limb leg) only LISTENS: your script's LegPose messages, the
# MotorCmd frames joint_command sends, and the motor's feedback. It stops 3 s after the last pose.
# The plots run on the HOST (the container has no matplotlib).
#
# Needs joint_command_node running with `legs` set (it creates /leg/<side>/joint_targets), and the
# E-stop in reach. Ctrl-C stops sending. Env: JC_SERVICE (default joint_command), LEGPOSE_SIDE
# (default left; must match the side your script publishes), LEGPOSE_NO_RECORD=1 to skip recording
# and plotting.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="${JC_SERVICE:-joint_command}"
SIDE="${LEGPOSE_SIDE:-left}"
CONTAINER="$(docker ps --filter "name=-${SERVICE}-" --format '{{.Names}}' | head -1)"

if [[ -z "$CONTAINER" ]]; then
  echo "No running '${SERVICE}' container found. Start it with:  ./watod up -d" >&2
  exit 1
fi

# /opt/humanoid_scripts, not /root/...: this container runs as the host user and /root is root-only.
SCRIPT=/opt/humanoid_scripts/legpose_test_script.py
RECORDER=/opt/humanoid_scripts/telemetry_record.py
RUN_ROOT="${REPO}/outputs/gl40_bench"

# Plot a run folder on the host with uv if present, else the host's own python3 + matplotlib.
plot_run() {
  local run="$1"
  if command -v uv >/dev/null 2>&1; then
    uv run --with matplotlib --with numpy "${REPO}/tools/gl40_telemetry_plot.py" "$run"
  elif python3 -c "import matplotlib, numpy" >/dev/null 2>&1; then
    python3 "${REPO}/tools/gl40_telemetry_plot.py" "$run"
  else
    echo "Cannot plot: the host has neither uv nor python3 with matplotlib + numpy." >&2
    echo "  Install one (e.g. 'pip install matplotlib numpy'), then run:" >&2
    echo "  python3 tools/gl40_telemetry_plot.py ${run}" >&2
    return 1
  fi
}

if [[ "${1:-}" == "--check" ]]; then
  docker exec "$CONTAINER" bash -lc "source /opt/watonomous/setup.bash
    test -r $SCRIPT || { echo 'FAIL: $SCRIPT is not readable in the container'; exit 1; }
    test -r $RECORDER || { echo 'FAIL: $RECORDER is not readable in the container'; exit 1; }
    python3 -c \"compile(open('$SCRIPT').read(), '$SCRIPT', 'exec'); import rclpy, common_msgs.msg\" \
      || { echo 'FAIL: the script does not compile, or rclpy/common_msgs cannot be imported'; exit 1; }
    echo 'ok: script and recorder are visible, compile, and their imports work'
    OUT=\"\$(python3 $RECORDER --limb leg --side ${SIDE} --no-log --duration 2 2>&1)\" \
      || { echo 'FAIL: the recorder cannot start (it could not read the leg config, or its imports are broken):'; echo \"\$OUT\" | tail -6; exit 1; }
    echo 'ok: the recorder found the leg calibration + actuators and started (passive, wrote nothing)'
    ros2 node list 2>/dev/null | grep -q '^/joint_command_node\$' \
      && echo 'ok: /joint_command_node is running' \
      || { echo 'FAIL: /joint_command_node is not running (start it: ros2 launch joint_command joint_command.launch.py)'; exit 1; }
    INFO=\"\$(ros2 topic info /leg/${SIDE}/joint_targets 2>&1)\"
    echo \"\$INFO\" | grep -q 'Subscription count: [1-9]' \
      && echo 'ok: joint_command is listening on /leg/${SIDE}/joint_targets' \
      || { echo 'FAIL: nothing listens on /leg/${SIDE}/joint_targets: is legs: [\"${SIDE}\"] set, and is a ${SIDE}-leg joint active: true?'; exit 1; }
    echo 'ok: output folder is writable'; test -w /outputs || { echo 'FAIL: /outputs is not writable in the container'; exit 1; }"
  if command -v uv >/dev/null 2>&1 || python3 -c "import matplotlib, numpy" >/dev/null 2>&1; then
    echo "ok: the host can plot (uv or python3 + matplotlib + numpy)"
  else
    echo "WARNING: the host cannot plot (no uv, no python3 matplotlib/numpy); the run will record but not plot"
  fi
  echo 'ALL GOOD: nothing was published.'
  exit 0
fi

# -it so Ctrl-C reaches the script -- only from a terminal.
TTY_FLAGS=()
[[ -t 0 && -t 1 ]] && TTY_FLAGS=(-it)
ARGS="$(printf '%q ' "$@")"

if [[ "${LEGPOSE_NO_RECORD:-0}" == "1" ]]; then
  exec docker exec "${TTY_FLAGS[@]}" "$CONTAINER" bash -lc "source /opt/watonomous/setup.bash; \
    exec python3 $SCRIPT ${ARGS}"
fi

# 1. Start the recorder (passive) and wait until it is subscribed, so no pose is missed.
mkdir -p "$RUN_ROOT"
REC_LOG="$(mktemp)"
docker exec "$CONTAINER" bash -lc "source /opt/watonomous/setup.bash; \
  exec python3 $RECORDER --limb leg --side ${SIDE} --label legpose-test \
    --idle-stop 3 --start-timeout 60 --duration 600" >"$REC_LOG" 2>&1 &
REC_PID=$!
for _ in $(seq 1 60); do
  grep -q '^recording' "$REC_LOG" 2>/dev/null && break
  kill -0 "$REC_PID" 2>/dev/null || break
  sleep 0.25
done
if ! grep -q '^recording' "$REC_LOG" 2>/dev/null; then
  echo "The recorder did not start; NOT sending any pose. Its output:" >&2
  cat "$REC_LOG" >&2
  kill "$REC_PID" 2>/dev/null || true
  exit 1
fi

# 2. Run your script (not exec: we still have to plot afterwards).
RC=0
docker exec "${TTY_FLAGS[@]}" "$CONTAINER" bash -lc "source /opt/watonomous/setup.bash; \
  exec python3 $SCRIPT ${ARGS}" || RC=$?

# 3. The recorder stops by itself 3 s after the last pose.
wait "$REC_PID" || true
echo
echo "--- recorder"
grep -v '^recording' "$REC_LOG" | sed 's/^/  /' || true

# 4. Plot into the run folder.
RUN_NAME="$(sed -n 's|^Telemetry: .*/||p' "$REC_LOG" | tail -1)"
rm -f "$REC_LOG"
if [[ -z "$RUN_NAME" || ! -d "${RUN_ROOT}/${RUN_NAME}" ]]; then
  echo "No run folder was written (nothing was recorded)." >&2
  exit "${RC:-1}"
fi
RUN="${RUN_ROOT}/${RUN_NAME}"
echo
echo "=> plotting ${RUN}"
plot_run "$RUN" || true
echo
echo "Output folder: ${RUN}"
ls -1 "$RUN" | sed 's/^/  /'
exit "$RC"
