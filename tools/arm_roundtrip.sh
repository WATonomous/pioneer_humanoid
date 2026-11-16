# Angle benchmark through the teleop path (arm_roundtrip.py in the joint_command container),
# then plot. See that script for the sequence.
#
#   tools/arm_roundtrip.sh --joints elbow.yaw --offset "0,0,0,0,5,0"
#   tools/arm_roundtrip.sh --pose "0,0,10,20,0,0" --label reach-a
#
# Needs can_node and joint_command_node running. Env: JC_SERVICE (default joint_command),
# GL40_NO_PLOT=1 to skip plotting.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="${JC_SERVICE:-joint_command}"
CONTAINER="$(docker ps --filter "name=-${SERVICE}-" --format '{{.Names}}' | head -1)"

if [[ -z "$CONTAINER" ]]; then
  echo "No running '${SERVICE}' container found. Start it with:  ./watod up -d" >&2
  exit 1
fi

mkdir -p "${REPO}/outputs/gl40_bench"
BEFORE="$(ls -1 "${REPO}/outputs/gl40_bench" 2>/dev/null | wc -l)"

# -it so Ctrl-C reaches the script (its handler ramps the arm home) -- only from a terminal.
TTY_FLAGS=()
[[ -t 0 && -t 1 ]] && TTY_FLAGS=(-it)
ARGS="$(printf '%q ' "$@")"
# /opt/humanoid_scripts, not /root/...: this container runs as the host user.
docker exec "${TTY_FLAGS[@]}" "$CONTAINER" bash -lc "source /opt/watonomous/setup.bash; \
  exec python3 /opt/humanoid_scripts/arm_roundtrip.py ${ARGS}" || RC=$?

AFTER="$(ls -1 "${REPO}/outputs/gl40_bench" 2>/dev/null | wc -l)"
if [[ "${GL40_NO_PLOT:-0}" == "1" || "$AFTER" == "$BEFORE" ]]; then
  exit "${RC:-0}"
fi

RUN="${REPO}/outputs/gl40_bench/$(ls -1t "${REPO}/outputs/gl40_bench" | head -1)"
echo
echo "=> plotting ${RUN}"
uv run --with matplotlib --with numpy "${REPO}/tools/gl40_telemetry_plot.py" "$RUN"
exit "${RC:-0}"
