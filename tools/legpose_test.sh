# Send the LegPose test poses to joint_command (legpose_test_script.py, run inside the joint_command
# container so you do not have to enter it or remember its paths).
#
#   tools/legpose_test.sh --check    # verify the setup. Publishes NOTHING.
#   tools/legpose_test.sh            # run it: publishes the POSES list in the script
#
# Needs joint_command_node running with `legs` set (it creates /leg/<side>/joint_targets), and the
# E-stop in reach. Ctrl-C stops sending. Env: JC_SERVICE (default joint_command).
set -euo pipefail

SERVICE="${JC_SERVICE:-joint_command}"
CONTAINER="$(docker ps --filter "name=-${SERVICE}-" --format '{{.Names}}' | head -1)"

if [[ -z "$CONTAINER" ]]; then
  echo "No running '${SERVICE}' container found. Start it with:  ./watod up -d" >&2
  exit 1
fi

# /opt/humanoid_scripts, not /root/...: this container runs as the host user and /root is root-only.
SCRIPT=/opt/humanoid_scripts/legpose_test_script.py

if [[ "${1:-}" == "--check" ]]; then
  docker exec "$CONTAINER" bash -lc "source /opt/watonomous/setup.bash
    test -r $SCRIPT || { echo 'FAIL: $SCRIPT is not readable in the container'; exit 1; }
    python3 -c \"compile(open('$SCRIPT').read(), '$SCRIPT', 'exec'); import rclpy, common_msgs.msg\" \
      || { echo 'FAIL: the script does not compile, or rclpy/common_msgs cannot be imported'; exit 1; }
    echo 'ok: script is visible, compiles, and its imports work'
    ros2 node list 2>/dev/null | grep -q '^/joint_command_node\$' \
      && echo 'ok: /joint_command_node is running' \
      || { echo 'FAIL: /joint_command_node is not running (start it: ros2 launch joint_command joint_command.launch.py)'; exit 1; }
    INFO=\"\$(ros2 topic info /leg/left/joint_targets 2>&1)\"
    echo \"\$INFO\" | grep -q 'Subscription count: [1-9]' \
      && echo 'ok: joint_command is listening on /leg/left/joint_targets' \
      || { echo 'FAIL: nothing listens on /leg/left/joint_targets: is legs: [\"left\"] set, and is a left-leg joint active: true?'; exit 1; }
    echo 'ALL GOOD: nothing was published.'"
  exit 0
fi

# -it so Ctrl-C reaches the script -- only from a terminal.
TTY_FLAGS=()
[[ -t 0 && -t 1 ]] && TTY_FLAGS=(-it)
ARGS="$(printf '%q ' "$@")"
exec docker exec "${TTY_FLAGS[@]}" "$CONTAINER" bash -lc "source /opt/watonomous/setup.bash; \
  exec python3 $SCRIPT ${ARGS}"
