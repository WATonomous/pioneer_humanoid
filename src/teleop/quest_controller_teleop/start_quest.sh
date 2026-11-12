#!/usr/bin/env bash
# One command, sim to headset in VR: forwards the port, leaves exactly one teleop tab in Meta Browser, starts the
# MuJoCo sim and clicks into VR (the browser needs a user gesture; index.html takes a click anywhere).
# Usage: start_quest.sh [mujoco_sim.py flags, e.g. --scene bimanual_sorting]   PY=python for non-macOS
set -euo pipefail
cd "$(dirname "$0")/../../.."
PORT=8444
PY=${PY:-.venv/bin/mjpython}

adb wait-for-device
adb reverse tcp:$PORT tcp:$PORT >/dev/null
adb forward tcp:9222 localabstract:chrome_devtools_remote >/dev/null
# Each `am start` opens another tab; stale tabs on this page only confuse which one you are looking at.
for id in $(curl -s localhost:9222/json/list | python3 -c "import json, sys
for t in json.load(sys.stdin):
    if t['type'] == 'page' and ':$PORT' in t['url']: print(t['id'])"); do
  curl -s "localhost:9222/json/close/$id" >/dev/null
done

"$PY" src/teleop/quest_controller_teleop/mujoco/mujoco_sim.py "$@" &
SIM=$!
trap 'kill $SIM 2>/dev/null' INT TERM
until curl -sf "localhost:$PORT/" >/dev/null; do sleep .5; done
# Closing the last tab quits the browser, and an `am start` sent while it is quitting is dropped: retry until the
# page is up (the devtools socket comes back with the browser, so forward it again each time).
for try in 1 2 3 4 5; do
  adb shell am start -a android.intent.action.VIEW -d "http://localhost:$PORT/" com.oculus.browser >/dev/null
  sleep 3
  adb forward tcp:9222 localabstract:chrome_devtools_remote >/dev/null 2>&1 || true
  curl -s localhost:9222/json/list | grep -q "localhost:$PORT" && break
done
adb shell input tap 500 400
echo "[Quest] Headset should be in VR. If it shows the page instead, pull the trigger anywhere on it."
wait $SIM
