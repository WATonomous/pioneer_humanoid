#!/usr/bin/env bash
# Bimanual arm + table + vial rack + 3 vials scene, built through the
# session daemon (isaac_session.sh must already be started). Saved here so
# it can be rebuilt identically later instead of re-typing the spawn
# commands each time.
#
# Usage:
#   tools/isaac_harness/isaac_session.sh start       # if not already running
#   tools/isaac_harness/scenes/bimanual_vial_rack.sh
#
# Table geometry from the archived pick_place_bimanual task
# (TABLE_TOP_Z=0.05, TABLE_DIMS=(0.9,1.2,0.05), TABLE_X_MIN=0.18) and the so101_vial_task's
# rack/vial assets (assets/lerobot/so101_vial_task/usd/).
#
# Vial spawn Z is 0.12, not the vials' resting height (~0.1) — spawning
# exactly at rest height risks a small initial interpenetration with the
# table collision mesh, which can trigger a violent PhysX correction impulse
# and send the vial flying/tipping instead of settling normally (observed:
# one vial out of three tipped over within 5 physics steps when spawned at
# 0.1; the extra clearance avoids it).

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="python3 $TOOLS_DIR/isaac_session_client.py"

VIAL_ASSETS="/workspace/humanoid/assets/lerobot/so101_vial_task/usd"

# Table top at the lightbox workcell's height below the arm base (humanoid_rl_tasks.workcell:
# TABLE_TOP_Z - ROBOT_STAND_LIFT_Z = -0.425): higher, the arms' home pose (forearms forward
# at elbow height, ~-0.24) sweeps through it. This is a demo-scene-only change; the real
# task's task_geometry.py is untouched.
TABLE_TOP_Z=-0.425

echo "[scene] table"
$C spawn_primitive --name Table --shape cuboid --size 0.9,1.2,0.05 \
  --pos 0.63,-0.2,-0.45 --color 0.35,0.25,0.15 --static

echo "[scene] robot (pioneer_bimanual_arm_8dof, real actuator config)"
$C spawn_bimanual_arm --name Robot --pos 0,0,0

echo "[scene] vial rack"
$C spawn_usd --name VialRack --usd-path "$VIAL_ASSETS/Vial_rack_simple.usda" --pos 0.5,-0.35,$TABLE_TOP_Z

echo "[scene] vials"
$C spawn_usd --name Vial1 --usd-path "$VIAL_ASSETS/Vial_opaque.usda" --pos 0.55,-0.05,-0.355
$C spawn_usd --name Vial2 --usd-path "$VIAL_ASSETS/Vial_opaque.usda" --pos 0.55,-0.15,-0.355
$C spawn_usd --name Vial3 --usd-path "$VIAL_ASSETS/Vial_opaque.usda" --pos 0.55,-0.25,-0.355

echo "[scene] settling physics"
$C step --n 60

echo "[scene] ready — spawned: Table, Robot, VialRack, Vial1, Vial2, Vial3"
