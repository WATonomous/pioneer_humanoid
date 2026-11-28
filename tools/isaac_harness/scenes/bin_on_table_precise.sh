#!/usr/bin/env bash
# Bimanual arm + packing table + a KLT bin, seated with verify_loop's
# `stack` check instead of pure drop-and-settle — dogfoods verify_loop.py
# (tools/isaac_harness/scenes/ycb_pick_place.sh spawns the same bin but
# never verifies it landed centered, just drops it and hopes).
#
# This scene exists as the SECOND attempt at dogfooding verify_loop — the
# first was a 3-box YCB stack (cracker box -> sugar box -> gelatin box).
# It converged the z-gap and x/y-centering exactly as designed every time,
# but the STACK ITSELF still wasn't stable under continued physics
# stepping: a supporting box that settles with even a few degrees of tilt
# gives gravity a persistent sideways component, so anything placed on it
# slides/tips off eventually regardless of how precisely it was initially
# centered — a real physical limitation, not a loop bug. verify_loop only
# corrects the placed object's own translation, not the supporting
# object's rotation/level. A bin on a big flat table has no such tilted-
# support problem, which is why this scene (not the box stack) is the one
# kept as the reference example.
#
# Usage:
#   tools/isaac_harness/isaac_session.sh start       # if not already running
#   tools/isaac_harness/scenes/bin_on_table_precise.sh

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="python3 $TOOLS_DIR/isaac_session_client.py"
V="python3 $TOOLS_DIR/verify_loop.py"

ISAAC_NUCLEUS_DIR="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"
TABLE="$ISAAC_NUCLEUS_DIR/Props/PackingTable/packing_table.usd"
KLT="$ISAAC_NUCLEUS_DIR/Props/KLT_Bin/small_KLT.usd"

echo "[scene] robot (pioneer_bimanual_arm, real actuator config)"
$C spawn_bimanual_arm --name Robot --pos=0,0,0

echo "[scene] packing table (scaled 0.6x uniform, rotated so its wide side faces the robot)"
$C spawn_usd --name Table --usd-path "$TABLE" --pos=0.7,0,-1.18 --scale=0.6,0.6,0.6 --rot=0.70710678,0,0,0.70710678
$C step --n 30

echo "[scene] bin, dropped roughly then precisely seated via verify_loop (z-gap + xy-centering)"
$C spawn_usd --name Bin --usd-path "$KLT" --pos=0.7,0,0.3
$C step --n 60
$V stack --name Bin --on Table --xy-tol 0.02

echo "[scene] settling physics further to confirm the precise seat holds"
$C step --n 150

echo "[scene] ready — spawned: Robot, Table, Bin"
