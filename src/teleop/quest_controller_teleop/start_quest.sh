#!/usr/bin/env bash
# One command, sim to headset in VR: forwards the port, leaves exactly one teleop tab in Meta Browser, starts the
# MuJoCo sim and clicks into VR (the browser needs a user gesture; index.html takes a click anywhere). The click is
# retried until the headset streams poses, so the VR controllers never have to touch the page.
# Usage: start_quest.sh [mujoco_sim.py flags, e.g. --scene bimanual_sorting]   PY=<python> to override
set -euo pipefail
cd "$(dirname "$0")/../../.."
PORT=8444
PY=${PY:-.venv/bin/python}

adb wait-for-device
adb reverse tcp:$PORT tcp:$PORT >/dev/null
# Each `am start` opens another tab; close stale teleop tabs. Best effort: the browser's DevTools socket is not always
# up (it goes away while the browser restarts), and a missed cleanup is harmless (the tab in VR owns control).
if adb forward tcp:9222 localabstract:chrome_devtools_remote >/dev/null 2>&1; then
  for id in $(curl -s --max-time 3 localhost:9222/json/list | python3 -c "import json, sys
try:
    for t in json.load(sys.stdin):
        if t['type'] == 'page' and ':$PORT' in t['url']: print(t['id'])
except ValueError:
    pass" || true); do
    curl -s --max-time 3 "localhost:9222/json/close/$id" >/dev/null || true
  done
fi

"$PY" src/teleop/quest_controller_teleop/mujoco/mujoco_sim.py "$@" &
SIM=$!
trap 'kill $SIM 2>/dev/null' INT TERM
until curl -sf "localhost:$PORT/" >/dev/null; do sleep .5; done
sleep 1   # the first control ticks replace the startup status with "Tracking paused"
streaming() { ! curl -s --max-time 2 "localhost:$PORT/status" | grep -q "Tracking paused"; }

# Closing the last tab quits the browser, and an `am start` sent while it is quitting is dropped: open, then click
# until poses arrive. Taps go to a few spots in case the browser panel sits elsewhere.
for try in 1 2 3 4 5 6; do
  streaming && break
  if [ $try = 1 ] || [ $try = 4 ]; then   # reopen sparingly: every open is another tab
    adb shell am start -a android.intent.action.VIEW -d "http://localhost:$PORT/" com.oculus.browser >/dev/null
    sleep 3
  fi
  for xy in "500 400" "960 600" "1300 500"; do
    adb shell input tap $xy
    sleep 2
    streaming && break 2
  done
done
if streaming; then
  echo "[Quest] Headset in VR and streaming."
else
  echo "[Quest] Could not click into VR (headset asleep?). Wake it and rerun, or pull the trigger on the page."
fi
wait $SIM
