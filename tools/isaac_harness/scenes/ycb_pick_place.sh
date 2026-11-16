#!/usr/bin/env bash
# Bimanual arm + packing table + 4 YCB grasp targets + a KLT placement bin,
# built through the session daemon (isaac_session.sh must already be
# started). Saved so it can be rebuilt identically instead of re-deriving
# the geometry each time — several of these numbers cost real debugging
# time (see notes below), not just placement guesses.
#
# Usage:
#   tools/isaac_harness/isaac_session.sh start       # if not already running
#   tools/isaac_harness/scenes/ycb_pick_place.sh
#
# Assets are all real Nucleus assets, found via `list_dir` rather than
# guessed filenames (isaacsim.core's ISAAC_NUCLEUS_DIR base):
#   Props/PackingTable/packing_table.usd
#   Props/KLT_Bin/small_KLT.usd
#   Props/YCB/Axis_Aligned/{003_cracker_box,006_mustard_bottle,025_mug,011_banana}.usd
#
# GOTCHAS THAT SHAPED THE NUMBERS BELOW:
#
# 1. packing_table.usd is ~1.08m tall unscaled (industrial height) — too
#    tall for this robot's reach. Scaled down 0.6x uniformly. Uniform, not
#    per-axis: an earlier attempt at anisotropic scale (0.5,0.6,0.7) caused
#    a ~6cm mismatch between the table's VISUAL bbox top and its actual
#    PhysX collision surface (objects dropped on top sank ~6cm below the
#    visual surface, consistently, across every object tested) — looked
#    exactly like a physics bug but was really non-uniform scaling breaking
#    the collision-mesh approximation. Uniform scale avoids it.
# 2. The table's own query()/root_pos_w reports a position that does NOT
#    match where it was spawned (a fixed, large offset — same every run,
#    independent of physics) — this asset's internal rigid-body prim has a
#    locally-authored offset baked into the file, unlike the YCB props
#    (which don't have this). For this asset, trust bbox() (raw USD,
#    world-composed), not query(), for "where did I actually put it."
# 3. The table's visual bbox top and its real physics-contact height still
#    don't match exactly even after fixing (1) — rather than trust either
#    number, every object here is spawned well ABOVE the table (z=0.3,
#    generous clearance) and left to fall and settle under real gravity;
#    the resting position is read back with query(), never assumed.
# 4. Spawning a rigid body with near-zero clearance at its expected resting
#    contact point risks a violent PhysX correction impulse (this scene's
#    table was actually flung sideways this way during iteration, when
#    first spawned flush against the ground plane with zero clearance).
#    Always leave real clearance for anything with rigid_props applied.
# 5. Table position kept clear of the robot's own bbox in x (checked with
#    `overlap`, not eyeballed) — an earlier table placement's footprint
#    overlapped the robot base by ~9cm before this was caught.
# 6. The table is rotated 90 deg (rot below) so its WIDE side faces the robot and
#    its NARROW side is the approach depth. `overlap`-clean placement isn't the
#    same as reachable placement: the first version of this scene had the table's
#    long (1.48m) axis running straight out from the robot as depth, with only its
#    short (0.47m) axis facing it — no bbox overlap, but most of the table (and the
#    bin on it) was ~1.5m+ from the robot base, well past any plausible reach.
#    Rotating 90 deg puts the wide 1.48m side across the robot's front and the
#    narrow 0.47m side as depth, so everything spawned on it stays close.

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="python3 $TOOLS_DIR/isaac_session_client.py"

ISAAC_NUCLEUS_DIR="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"
TABLE="$ISAAC_NUCLEUS_DIR/Props/PackingTable/packing_table.usd"
KLT="$ISAAC_NUCLEUS_DIR/Props/KLT_Bin/small_KLT.usd"
YCB="$ISAAC_NUCLEUS_DIR/Props/YCB/Axis_Aligned"

echo "[scene] robot (pioneer_bimanual_arm_8dof, real actuator config)"
$C spawn_bimanual_arm --name Robot --pos=0,0,0

echo "[scene] packing table (scaled 0.6x uniform, rotated 90deg so its wide side faces the robot)"
$C spawn_usd --name Table --usd-path "$TABLE" --pos=0.7,0,-1.18 --scale=0.6,0.6,0.6 --rot=0.70710678,0,0,0.70710678
$C step --n 30

echo "[scene] 4 YCB grasp targets + KLT bin, dropped from clearance onto the table"
$C spawn_usd --name CrackerBox --usd-path "$YCB/003_cracker_box.usd" --pos=0.7,-0.55,0.3 --apply-physics
$C spawn_usd --name MustardBottle --usd-path "$YCB/006_mustard_bottle.usd" --pos=0.7,-0.25,0.3 --apply-physics
$C spawn_usd --name Mug --usd-path "$YCB/025_mug.usd" --pos=0.7,0.25,0.3 --apply-physics
$C spawn_usd --name Banana --usd-path "$YCB/011_banana.usd" --pos=0.7,0.55,0.3 --apply-physics
$C spawn_usd --name Bin --usd-path "$KLT" --pos=0.7,0.0,0.3

echo "[scene] settling physics"
$C step --n 150

echo "[scene] ready — spawned: Robot, Table, CrackerBox, MustardBottle, Mug, Banana, Bin"
