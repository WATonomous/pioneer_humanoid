#!/usr/bin/env bash
# Headset to collection in one command, as quest_controller_teleop/start_quest.sh: forward the port over USB, open the
# headset page in Meta Browser, start collect.py and tap into VR until poses arrive.
# Usage: src/umi_gripper/start_umi.sh --task "..." [collect.py flags]      PY=<python> to override
set -euo pipefail
cd "$(dirname "$0")"
PORT=${PORT:-8444}
PY=${PY:-../../.venv/bin/python}

adb wait-for-device
adb reverse tcp:$PORT tcp:$PORT >/dev/null
"$PY" -m umi_gripper.collect "$@" &
COLLECT=$!
trap 'kill -TERM $COLLECT 2>/dev/null' INT TERM
until curl -sf "localhost:$PORT/" >/dev/null; do sleep .5; done
streaming() { ! curl -s --max-time 2 "localhost:$PORT/status" | grep -q "no poses"; }
sleep 1.5
for try in 1 2 3 4 5 6; do
  streaming && break
  if [ $try = 1 ] || [ $try = 4 ]; then
    adb shell am start -a android.intent.action.VIEW -d "http://localhost:$PORT/" com.oculus.browser >/dev/null
    sleep 3
  fi
  for xy in "500 400" "960 600" "1300 500"; do
    adb shell input tap $xy
    sleep 2
    streaming && break 2
  done
done
streaming && echo "[UMI] Headset in VR and streaming." \
  || echo "[UMI] Could not click into VR (headset asleep?). Wake it, or pull the trigger on the page."
wait $COLLECT
