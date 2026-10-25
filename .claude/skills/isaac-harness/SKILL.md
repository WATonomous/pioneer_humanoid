---
name: isaac-harness
description: Use whenever driving IsaacLab/Isaac Sim scenes in this repo remotely — no physical access to the monitor (Claude Code Remote Control, SSH, headless box) — for tasks like building/checking a scene, spawning objects or this repo's robots, taking screenshots to show the user, or verifying scene geometry (positions, overlaps, clearances) programmatically. Also read before touching tools/isaac_harness/ itself. Triggers on "show me the scene", "take a screenshot of isaac sim", "build a scene with X", "spawn the robot", "check if X is on the table", or any remote IsaacLab iteration.
---

# isaac_harness

`tools/isaac_harness/` is this repo's toolkit for driving IsaacLab/Isaac Sim scenes when
there's no physical access to the monitor. Built and hardened over one long session —
every gotcha below cost real time to discover; don't rediscover them.

## Pick the right tool first

| Need | Use |
|---|---|
| One screenshot, single check | `isaac_screenshot.sh` |
| Several camera angles of one build, single check | `isaac_multiview_screenshot.sh` (GUI/xwd) or `isaac_headless_multiview.sh` (faster, no display) |
| Building a scene incrementally, checking positions, iterating on placement, anything with >1-2 round trips | **`isaac_session.sh` + `isaac_session_client.py`** — a persistent daemon. Relaunching Isaac Sim costs ~15-40s every time; against a running daemon each command costs well under 2s. |

Full command reference and usage examples: `tools/isaac_harness/README.md`. This skill is
the "how to not get burned" layer on top of that, not a replacement for it.

## The daemon workflow (the common case)

```bash
tools/isaac_harness/isaac_session.sh start          # ~20-30s, once
python3 tools/isaac_harness/isaac_session_client.py spawn_primitive --name X ...
python3 tools/isaac_harness/isaac_session_client.py spawn_bimanual_arm --name Robot ...
python3 tools/isaac_harness/isaac_session_client.py screenshot --eye ... --target ... --out shot.png
tools/isaac_harness/isaac_session.sh stop           # see shutdown timing note below
```

There are saved reference scenes under `tools/isaac_harness/scenes/` — reuse or extend
rather than re-typing spawn commands from scratch:
- `bimanual_vial_rack.sh` — table + this repo's robot + vial rack + 3 vials
- `ycb_pick_place.sh` — robot + a real Nucleus packing table + 4 YCB grasp objects +
  a KLT bin, all dropped and settled under real physics. Read its header comment before
  copying its numbers elsewhere — every one of them exists because a naive first attempt
  broke in a specific, now-documented way (see the new sections below).

## Hard rules

- **Never `SIGKILL` (`kill -9`) an Isaac Sim process without the user's explicit
  go-ahead for that specific process, every time.** SIGKILL gives it no chance to
  release its GPU/graphics context. The blast radius has ranged from "next launch
  segfaults instantly" to "needs a full machine reboot to recover." Always try
  `kill -SIGINT` (same as Ctrl+C) or plain `kill` (SIGTERM) first, or the daemon's own
  `shutdown` command. `isaac_session.sh` already does this before relaunching.
- **Never pass `--experience isaacsim.exp.full.kit`.** That's for an unrelated
  cloud/pip-only Isaac Sim setup and causes a deterministic native segfault
  (`omni.graph.core.plugin.so`) on this repo's local Docker install. Omit the flag —
  `isaaclab.sh` picks the correct local experience on its own.
- **Check the host's actual `$DISPLAY` before any GUI-mode launch.** It is NOT
  reliably `:0` — this box can boot into Wayland (`:0`) or a plain X11 session (`:1`),
  and it changes across reboots. Verify with `echo $DISPLAY` / `w` first. Headless mode
  sidesteps this entirely — prefer it when a live window isn't specifically needed.
- **Graceful shutdown can legitimately take minutes on a heavy, long-lived daemon
  session** (many spawned objects, an articulation, a long command history) — this is
  confirmed, not a guess: the daemon's shutdown *logic* was code-reviewed and found
  correct, and `simulation_app.close()` was still observed pumping ~65% GPU load 3+
  minutes after accepting the shutdown command.
  **Standing exception to the SIGKILL rule above, for this specific daemon only:**
  the user has explicitly authorized SIGKILLing `isaac_session_daemon.py` (not any
  other Isaac Sim process) after a ~15-30s wait when graceful shutdown hangs like
  this — no need to ask each time. It's been repeated several times and always
  relaunches cleanly afterward. Always verify with a fresh `isaac_session.sh start`
  that it does come back up clean; if a launch ever crashes after this, stop and go
  back to asking before further force-kills, since that would mean the pattern
  changed. This exception does NOT extend to one-shot scripts or GUI launches — those
  still need per-instance authorization.

## Use `bbox`/`query`/`distance` to answer questions, not screenshots + guessing

The single biggest workflow mistake made building this: repeatedly guessing a
parameter (e.g. "is the ground plane in the right place?"), taking a screenshot,
eyeballing it, guessing again. This wastes many round trips and the guesses can still
be wrong in ways that aren't obvious from a render angle.

Instead: **query the actual geometry and compute the answer directly.**

Example — placing a ground plane flush under a robot's base instead of trial-and-error:
```bash
python3 isaac_session_client.py bbox --name Robot
# {"min": [-0.30, -0.38, -1.1997], "max": [0.35, 0.38, 0.30], ...}
# → ground Z should be -1.1997, not a guessed round number
```
This found a real bug: a previously "looks fine" screenshot was actually hiding the
ground plane cutting 0.2m into the robot's base — invisible at that camera angle,
obvious as soon as the actual number was checked.

**Known limitation to work around, not ignore:** `bbox` reads raw USD stage
transforms, which do NOT reliably sync from the physics tensor/Fabric layer every
step. For an object that's actively moving/falling under physics, `bbox` can silently
return a stale, spawn-time position — `query()`'s tensor-sourced pose is authoritative
for a moving tracked object instead. The daemon flags this itself: any `bbox` response
for a tracked *dynamic* (non-kinematic) object includes a `"warning"` field saying
exactly this — check for it, and prefer `query()` when it's present and the object may
have moved. Static/kinematic objects (tables, ground, anything that hasn't been
stepped) don't carry this risk.

`overlap` (AABB overlap, boolean) and `distance` (center-to-center) exist to answer
"is X actually on Y" / "are these interpenetrating" programmatically — use them instead
of a screenshot + eyeball judgment call when the question is that concrete.

## Finding & placing external Nucleus assets (YCB, packing tables, bins, etc.)

- **Don't guess an asset's filename — use `list_dir`.** It runs `omni.client.list()`
  against a real Nucleus (or local) directory and returns actual entries. Browse down
  to the file before spawning it:
  ```bash
  python3 isaac_session_client.py list_dir --path "$ISAAC_NUCLEUS_DIR/Props/YCB/"
  python3 isaac_session_client.py list_dir --path "$ISAAC_NUCLEUS_DIR/Props/YCB/Axis_Aligned/"
  # -> 003_cracker_box.usd, 006_mustard_bottle.usd, 025_mug.usd, ...
  ```
  `ISAAC_NUCLEUS_DIR` is
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac`
  (IsaacLab's own `isaaclab.utils.assets.ISAAC_NUCLEUS_DIR` constant — check that file if
  this ever needs updating for a version bump). Other useful categories found this way:
  `Props/KLT_Bin/`, `Props/PackingTable/`, `Props/Pallet/`, `Props/Sektion_Cabinet/`.
- **If an object only needs to be *looked at* (no physics interaction), use
  `spawn_visual`, not `spawn_usd`.** It places the USD reference directly with no
  `RigidObject`/`Articulation` wrapper at all, so it can't hit the RigidBodyAPI problem
  below — there's nothing to apply a physics schema to in the first place. Used for a
  quick "what does this asset even look like" screenshot.
- **A bare-mesh USD asset with no physics baked in (YCB props included) needs
  `spawn_usd --apply-physics` — and the reason it needs a *flag* at all, rather than
  working automatically, is worth understanding, not just invoking.** IsaacLab's
  `UsdFileCfg(rigid_props=..., collision_props=..., mass_props=...)` routes through
  `isaaclab.sim.schemas.modify_rigid_body_properties()` (and the collision/mass
  equivalents) — and those `modify_*` functions only **modify an already-applied**
  API schema. If the asset has no `RigidBodyAPI` baked in, the check fails and the
  function just `return`s `False`, no exception, no warning. The actual failure only
  shows up later and confusingly, inside `RigidObject._initialize_impl()` at the next
  `sim.reset()`: `"Failed to find a rigid body... Please ensure RigidBodyAPI is
  applied."` IsaacLab ships a separate `define_rigid_body_properties()` /
  `define_collision_properties()` / `define_mass_properties()` family that applies the
  schema *first*, then delegates to the same `modify_*()` — that's the step
  `--apply-physics` performs explicitly, directly on the spawned prim, instead of
  routing through `UsdFileCfg`'s cfg fields. Confirmed fixed by dropping a cracker box
  from height and watching it actually fall and settle under real gravity (not just
  "no exception at spawn time").
- **`scale` on `spawn_usd`/`spawn_visual` should be uniform, not per-axis, for a
  physics-enabled asset.** An anisotropic scale (e.g. `0.5,0.6,0.7`) caused a
  reproducible ~6cm mismatch between an asset's *visual* bbox top and its actual PhysX
  collision surface — every object dropped onto a non-uniformly-scaled table sank a
  consistent few cm below the visible tabletop. Looked exactly like a physics bug;
  was actually non-uniform scaling breaking the collision-mesh approximation for that
  asset. A uniform scale factor (`0.6,0.6,0.6`) avoided it entirely on the same asset.
  If a non-uniform footprint is genuinely needed, verify with a drop-test (below)
  before trusting it, don't assume it scales cleanly.
- **Don't precompute an exact resting/contact height for a physics object — drop it
  from real clearance and read back where it actually landed with `query()`.** Even
  after fixing the scale issue above, a scaled prefab asset's *visual* bbox top and its
  *actual* physics-contact height still didn't line up closely enough to trust by
  calculation. The robust pattern: spawn well above the target surface (e.g.
  `table_top_z + 0.3`, not `table_top_z + 0.02`), `step` enough times to fully settle
  (~60-150 depending on object count), then read the real resting pose with `query()`.
  This is strictly more reliable than computing "table top + half the object's own
  height" by hand.
- **Some assets' `query()` reports a pose that doesn't match where you spawned them —
  by a large, constant offset — even when the object never moved.** One packing-table
  asset consistently reported a `root_pos_w` offset by (+0.3, -0.06, ...) from its
  actual spawn translation, identically across multiple rebuilds, regardless of
  physics settling. This isn't the usual bbox-staleness issue (see below) — it's the
  opposite: the asset's *actual* PhysX rigid-body prim (wherever it's authored inside
  the referenced file) has a local transform offset baked in, and `query()`'s
  `root_pos_w` reports that inner prim's frame, not the outer reference Xform's. For
  an asset like this, trust `bbox()` (raw USD, world-composed from the outer
  transform you actually set) over `query()` for "is this where I put it" — this is
  the mirror image of `bbox`'s own staleness problem below (there, `query()` is
  trustworthy and `bbox()` is stale; here it's reversed, and only some assets do it).
  Don't assume either source is universally authoritative — sanity-check a spawned
  asset's `query()` against its known spawn translation once, and treat a large
  constant mismatch as this offset, not as "the object moved."

## Scene layout: check orientation and reach, not just AABB overlap

`overlap` only answers "do these two bounding boxes intersect" — it does NOT answer
"is this table oriented so its reachable surface actually faces the robot." A table
placed with zero overlap against the robot can still be effectively unusable if its
*long* axis runs away from the robot (depth) instead of across it (width) — everything
past the first ~0.5m becomes unreachable even though nothing is technically colliding.
Before finalizing a multi-object scene layout: `bbox` the table/surface and the robot,
compare which world axis carries the surface's long dimension against which axis the
robot is offset along, and check that the *short* dimension is the one between the
robot and the surface (i.e. the wide side faces the robot, the narrow side is the
approach depth) — not just that the two boxes don't intersect.

## Robot-specific

- **This repo's robot is `pioneer_bimanual_arm.usd`.** If it looks like a single
  6-DOF arm rather than two arms, or looks like an old/wrong model, `git pull` — the
  canonical asset and its config (`Teleop/keyboard_based_teleoperation/bimanual_arm_cfg.py`)
  have moved before (deleted+replaced in commit `950b86b8`, "Adding the new URDF and
  USD files... Removing unused USD/URDF files").
- **Spawn it with `spawn_bimanual_arm`, not generic `spawn_usd --articulation`.**
  The generic command builds a bare `ArticulationCfg` with no actuators and throws
  `TypeError: Missing values detected ... - actuators`. `spawn_bimanual_arm` uses the
  real, actuator-complete `BIMANUAL_ARM_CFG` instead.

## Object-spawning gotchas

- `spawn_primitive --static` means *kinematic* (still a rigid body, just not
  physics-driven), not "no rigid body at all" — `RigidObject` requires
  `USD RigidBodyAPI` on the prim regardless.
- Spawning a rigid object exactly at its expected resting height risks a small
  initial interpenetration with whatever it's resting on, which can trigger a violent
  PhysX correction impulse (observed: a vial spawned at table-top+0.05 tipped over and
  flew within 5 physics steps; +0.02 extra clearance fixed it). Spawn slightly above
  resting height and let physics settle, don't spawn exactly at it.
- **`set_pose` on a kinematic object (`spawn_primitive --static`) used to crash the
  whole process** — a genuine PhysX/CUDA "illegal memory access" abort (core dump),
  not a Python exception, from calling `write_root_pose_to_sim()` on a kinematic body
  (it isn't driven through that tensor-write path). **Fixed**: `set_pose` now writes a
  kinematic object's new pose directly to its USD prim transform instead of through
  PhysX, and dynamic objects still use `write_root_pose_to_sim()` as before.
  Object dynamic/kinematic-ness is tracked automatically (same flag `bbox`'s staleness
  warning already uses) — nothing to configure when calling `set_pose`.
- **`set_pose` changes surviving a later `spawn_*`/`remove` — fixed, real root cause
  found.** A real (non-soft) `sim.reset()` calls PhysX `stop()` then `play()`. On
  PLAY, a rigid body's pose is re-initialized from the USD prim's **authored xformOp
  transform on the stage** — not from `cfg.init_state`, not from
  `data.default_root_state`. Two earlier fixes both targeted the wrong source and
  both failed silently for that reason (mutating `cfg.init_state`, writing directly
  into `data.default_root_state` — neither is what PLAY actually reads).
  `write_root_pose_to_sim()` only ever touches the live physics *tensor*; it never
  writes back to the USD prim's actual attributes (confirmed: `query()` reflected a
  new pose instantly, but a `bbox()` read of the same prim right after showed the
  authored transform unchanged) — so PLAY kept re-reading the stale, original spawn
  transform regardless. **Fixed** by writing the new pose to the prim's xformOps for
  every object now, dynamic or kinematic (previously only the kinematic branch did
  this) — `write_root_pose_to_sim()` still runs too, so `query()` reflects the move
  immediately without needing a `step()` first. Verified: spawn, `set_pose`, spawn a
  second throwaway object (triggers `sim.reset()`), re-`query()` the first — pose
  held.
- **After `set_pose`, `query()` is now reliable for both dynamic and kinematic
  objects** (the USD-prim write happens for both) — no need to special-case which
  command to check with anymore.
- **`remove` used to corrupt every OTHER tracked Articulation's physics view — fixed,
  real root cause found.** Symptom was `ReferenceError: weakly-referenced object no
  longer exists` / `Failed to get DOF velocities from backend` on a completely
  untouched object's next `step`/`query`, right after removing something else
  entirely. Cause: `del objects[name]` alone doesn't immediately garbage-collect the
  removed object — `_create_buffers()` gives every asset a `WrenchComposer(self)`, a
  reference cycle, so CPython's refcounting can't free it right away. Until an actual
  GC pass happens, the removed object's PLAY/STOP timeline callbacks (registered in
  `AssetBase.__init__`, only unsubscribed in `__del__`) stay live — and the very next
  `sim.reset()` fires them against a prim `stage.RemovePrim()` already deleted,
  throwing mid-callback in a way that corrupted shared PhysX/tensor-view state for
  every other tracked Articulation. Plain resets from ordinary spawns never showed
  this; only removes did, because only removes delete the underlying prim out from
  under a still-subscribed callback. **Fixed** by forcing `gc.collect()` between
  `stage.RemovePrim()` and `sim.reset()`, so the removed object's `__del__` runs and
  unsubscribes it before the reset cycle can fire its now-orphaned callback. An
  earlier, different fix attempt here (forcing every OTHER object to manually
  reinitialize) caused a genuine CUDA crash and was reverted — this fix doesn't touch
  other objects at all. Verified: spawn an Articulation + a throwaway object, remove
  the throwaway, then `step`/`joint_state`/`query` the untouched Articulation — no
  error, repeatable across multiple removes in a row.
- **After any camera repositioning, check the shot actually shows what you think it
  does before drawing a conclusion from it** — a "nothing changed" screenshot can mean
  the change didn't happen (see above) OR it can mean you're looking at the wrong side
  of the object entirely (a 180° rotation viewed from the original camera angle shows
  the back, not "no visible change"). Confirm with `query()`/`bbox()` and, if
  orientation is in question, a shot from more than one angle before concluding
  anything is broken or unchanged.

## verify_loop.py — check/adjust convergence for placement that can't be computed up front

`tools/isaac_harness/verify_loop.py` is a client-side (host-side) library, NOT a
daemon command — it composes existing atomic commands (`bbox`/`overlap`/`query`/
`set_pose`) into a check-adjust-recheck loop. Only reach for it when the target
can't just be computed and spawned at directly (a scaled prefab asset's real
physics-contact height not matching its visual bbox, or genuinely trial-and-error
placement) — for anything with a known target, spawn there directly, a loop is
pure overhead.

Use `is_stacked_on`/`settle_stack` (or the CLI's `stack` subcommand), not
`is_resting_on`/`close_gap_along_axis` (`rest`), for anything that needs to stay
put under continued physics — `is_resting_on` only checks the z-gap, and an
object placed with only the z-gap corrected can converge cleanly and STILL
drift/topple over further stepping, because nothing corrected x/y centering.
Confirmed by testing, not assumed.

**Known limitation, confirmed by testing, not yet fixed:** even with x/y
centering, `verify_loop` only corrects the *placed* object's own translation —
it never corrects the *supporting* object's rotation. A 3-box YCB stack
(cracker/sugar/gelatin box) converged the z-gap and x/y-centering perfectly at
every step, but still wasn't stable under continued physics stepping, because
the supporting box itself settled with a few degrees of residual tilt — a
tilted support surface gives gravity a persistent sideways component regardless
of how precisely the object on top was centered. This is a real physical
limitation of the current tool, not a bug to route around with more damping or
tighter tolerances. Prefer flat, large, well-settled support surfaces (a table,
not another small irregular object) for anything built with `verify_loop` until
a leveling check/adjustment exists. `scenes/bin_on_table_precise.sh` is the
verified-stable reference example (positions bit-identical across 300 extra
physics steps after convergence); the 3-box stack was tried and abandoned for
this reason.

## When to ask vs. proceed

Ask before forcing a stuck shutdown (see hard rules). Otherwise, act — spawn, step,
query, screenshot freely; none of that is destructive or hard to undo. Multi-view
screenshots from several angles are usually worth the extra couple of calls when
judging spatial correctness (a single angle can hide a problem, as the ground-plane
bug above shows) — don't stop at one shot if the question is "is this actually right."
