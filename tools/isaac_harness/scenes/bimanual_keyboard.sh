#!/usr/bin/env bash
# Bimanual arm + table + QWERTY keyboard (37 keys), built through the session
# daemon (isaac_session.sh must already be started).
#
# Usage:
#   tools/isaac_harness/isaac_session.sh start
#   tools/isaac_harness/scenes/bimanual_keyboard.sh
#   KEY_SCALE=1.0 tools/isaac_harness/scenes/bimanual_keyboard.sh   # real-size keys
#
# KEY_SCALE defaults to 2.0: a real 19 mm key pitch is narrower than the
# gripper's closed fingertip gap (~74 mm), so the arm would hit several keys at
# once. Drop to 1.0 once a pointing/stylus fingertip exists.
#
# Layout: robot at origin facing +x; rows run along y, successive rows step
# away from the robot along +x. Keys are static (kinematic) -- no press
# physics yet. Table geometry matches bimanual_vial_rack.sh (top at z=-0.25).

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TOOLS_DIR
export KEY_SCALE="${KEY_SCALE:-2.0}"
C="python3 $TOOLS_DIR/isaac_session_client.py"

echo "[scene] table"
$C spawn_primitive --name Table --shape cuboid --size 0.9,1.2,0.05 \
  --pos 0.63,-0.2,-0.275 --color 0.35,0.25,0.15 --static

echo "[scene] robot (pioneer_bimanual_arm, real actuator config)"
$C spawn_bimanual_arm --name Robot --pos 0,0,0

echo "[scene] keyboard (KEY_SCALE=$KEY_SCALE)"
python3 - <<'PY'
import os, subprocess, sys

tools = os.environ["TOOLS_DIR"]
s = float(os.environ["KEY_SCALE"])
client = ["python3", os.path.join(tools, "isaac_session_client.py")]

TABLE_TOP_Z = -0.25
PITCH = 0.019 * s            # key center-to-center
KEY = 0.015 * s              # key footprint
KEY_H = 0.012 * s            # key height above base
BASE_H = 0.015 * s           # base plate thickness
# (row letters, stagger in key pitches) -- x = row index away from robot
ROWS = [("1234567890", 0.0), ("QWERTYUIOP", 0.5), ("ASDFGHJKL", 0.75), ("ZXCVBNM", 1.25)]
CENTER_X, CENTER_Y = 0.50, 0.0   # keyboard center on the table
n_rows = len(ROWS) + 1           # +1 for the space bar row
width = 11 * PITCH + 0.01 * s    # +1 pitch: the staggered Q row overhangs 10 pitches
depth = n_rows * PITCH + 0.01 * s
base_z = TABLE_TOP_Z + BASE_H / 2
key_z = TABLE_TOP_Z + BASE_H + KEY_H / 2 + 0.002   # 2 mm clearance, kinematic anyway


def run(*args):
    subprocess.run(client + [str(a) for a in args], check=True, stdout=subprocess.DEVNULL)


def spawn(name, size, x, y, z, color):
    run("spawn_primitive", "--name", name, "--shape", "cuboid",
        "--size", ",".join(f"{v:.4f}" for v in size),
        "--pos", f"{x:.4f},{y:.4f},{z:.4f}", "--color", color, "--static")


spawn("KeyboardBase", (depth, width, BASE_H), CENTER_X, CENTER_Y, base_z, "0.08,0.08,0.09")

x0 = CENTER_X - (n_rows - 1) * PITCH / 2
for r, (letters, stagger) in enumerate(ROWS):
    x = x0 + r * PITCH
    for i, ch in enumerate(letters):
        y = CENTER_Y - 4.5 * PITCH + stagger * PITCH + i * PITCH
        spawn(f"Key_{ch}", (KEY, KEY, KEY_H), x, y, key_z, "0.85,0.85,0.88")
        print(f"  key {ch}", file=sys.stderr, end="\r")

x = x0 + len(ROWS) * PITCH
spawn("Key_SPACE", (KEY, 5 * PITCH - (PITCH - KEY), KEY_H), x, CENTER_Y, key_z, "0.85,0.85,0.88")
PY

echo "[scene] settling physics"
$C step --n 30

echo "[scene] ready -- Table, Robot, KeyboardBase, Key_* (37 keys)"
echo "[scene] shot: $C screenshot --eye 0.1,-0.9,0.5 --target 0.5,0,-0.2 --out keyboard.png"
