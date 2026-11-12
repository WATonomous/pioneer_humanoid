# NASA Cargo Transfer Bag pick-and-place

Scene `nasa_cargo` places a lightweight **0.1 kg (100 g)** NASA Cargo Transfer Bag (CTB) on a
gray pickup station beside a green receiving tray, in front of Pioneer's
bimanual arms. Pick up the red handle, lift clear, transfer to the green
tray and set the bag down upright. The bag moves only through physics.

This adapts the [NASA CLR cargo pick-and-place demo](https://github.com/NASA-JSC-Robotics/clr_ws/blob/jazzy-devel/src/clr_demos/clr_behavior_demos/trees/roboplan_pick_and_place_ctb.xml)
to a fixed arm stand. NASA's rail/lift traversal and bench destination are
replaced with two nearby stations; their planner is not included. A scripted
privileged-state expert generates bootstrap demonstrations through robot
contact, and a separate runner executes your learned policy and records human
corrections. See [the training/HG-DAgger workflow](HG_DAGGER.md).

For fixed cargo and a randomized green destination, see [moving-tray playback and HG-DAgger](MOVING_TRAY.md).

## Open the scene

On the host, from your repo root (skip startup if already running):

```bash
ACTIVE_MODULES="simulation_isaac" ./watod up -d
./watod -t simulation_isaac
```

Inside the container:

```bash
cd /workspace/humanoid
/workspace/isaaclab/isaaclab.sh -p tools/isaac_harness/view_nasa_cargo.py --device cuda
```

Use normal Isaac viewport navigation. Arms hold their resting pose. `R`
resets the bag and task history; closing the window exits. No VR, ROS or
recording is started. The viewer does not pick up the bag automatically.

## Test the task physics

```bash
/workspace/isaaclab/isaaclab.sh -p tools/isaac_harness/view_nasa_cargo.py \
  --headless --device cuda --check
```

Look for `NASA_CARGO_CHECK_PASS`. Checks cover:

- Stable gravity-driven rest on the pickup station.
- Contact/friction lifting of the actual handle by a parallel-jaw test fixture,
  with no attachment joint, bag pose updates or externally commanded bag forces.
- A drop above the target followed by stable placement and task success.
- Rejection of partial overhang, hovering, moving and tipped poses; task reset.

The fixture is added only for `--check`. The drop check explicitly repositions
the bag above the target; it is not an autonomous transfer demonstration.
Neither check proves that the humanoid can grasp it with its own grippers.

The physics check covers the current mass. Earlier 1.8 kg and 1.0 kg versions
also passed the fixture lift, target drop, negative-placement and reset checks.
The asset has no external dependencies. The current 100 g setting is a
lightweight training prop: its payload alone contributes about 0.10 N m at
a 10 cm horizontal offset. This does not include gripper weight or dynamic
loads, and does not establish a hardware payload rating. The hardware wrist
and gripper must still be commissioned before attempting physical pickup.

## Connect your policy

For synthetic data generation, overnight ACT training, keyboard correction
collection and dataset aggregation, see [the HG-DAgger workflow](HG_DAGGER.md).

After your usual Isaac AppLauncher:

```python
from isaaclab.scene import InteractiveScene
from humanoid_isaac_scenes import make_scene_cfg
from humanoid_isaac_scenes.nasa_cargo.task import CargoTask
from pioneer_humanoid.bimanual_arm import BIMANUAL_ARM_CFG

scene = InteractiveScene(make_scene_cfg("nasa_cargo", BIMANUAL_ARM_CFG))
sim.reset()
task = CargoTask(scene)
# Your observations -> ML policy -> robot joint targets -> physics step.
# scene.update(dt)
status = task.update(dt)
# status['ever_lifted'], status['inside_target'], status['success']: tensors per environment.
```

Success requires a previous lift of at least 6 cm, the entire conservative
bag bounding box inside the tray, upright within 15 degrees, its bottom
within 1.5 cm of the tray floor, and low velocity continuously for one second.
This detects settled placement; it does not independently detect gripper
release or prohibit a gripper from touching a placed bag. Call `task.reset()`
whenever you reset the bag. Add whichever cameras your model needs; the
scene-only viewer has no observation cameras by default.

If you later want to collect demonstrations, the existing Quest runner also
accepts this scene:

```bash
src/teleop/quest_teleop/sim/run_quest_bimanual_teleop.sh \
  --scene nasa_cargo --record --record-arms bimanual
```

Its usual bridge setup is required. Local stereo pinhole cameras are the
default for this scene. Robot state/action ordering remains left 7 values,
then right 7, using absolute joint targets and normalized gripper closure.

## Geometry and provenance

The bag is rigid, not simulated cloth. Body dimensions are approximately
40.6 × 22.9 × 53.4 cm, with the red handle above it. Source inches are converted
using NASA's 0.0254 mesh scale and exact source origin rotations. Mass and
diagonal inertia are based on the original Xacro: mass is reduced from NASA's
1.8 kg to 0.1 kg and inertia is multiplied by 0.1/1.8, preserving geometry and
relative mass distribution. This is a simulation task setting, not a verified
hardware payload rating. The bag starts without yaw;
the pickup handle is roughly at x=0.50 m, y=0.22 m, z=1.05 m. Your arm's
grasp and load capacity still need a real control/VR trial.

Station positions, tray interior and heights live in
[scene.py](../../src/simulation/isaac_scenes/humanoid_isaac_scenes/nasa_cargo/scene.py).
The stations are 48 cm high to keep the full-size bag's handle around 1.05 m
without scaling NASA's model. The supports and tray are our task geometry. Body collision uses a convex
hull; handle collision uses convex decomposition to retain the grasp opening.

`ctb.usdc` is self-contained and included, so running the scene needs no
downloads. Rebuild with `python assets/nasa_cargo/build_usd.py` in an environment
with `numpy` and USD Python bindings (e.g. `usd-core`).

`source/` contains the unmodified CTB subset from
[NASA's mockups repository](https://github.com/NASA-JSC-Robotics/surface_robotics_mockups_description/tree/934f1129085450e99ac6c1cf3e7f9b20ad076469/ctb)
at the revision referenced by CLR, `934f1129085450e99ac6c1cf3e7f9b20ad076469`,
plus the [Apache 2.0 license](source/LICENSE) and [citation](source/CITATION.cff).
[source_manifest.json](source_manifest.json) records source SHA-256 hashes.
Please retain attribution and cite NASA's iMETRO facility paper in published work.
