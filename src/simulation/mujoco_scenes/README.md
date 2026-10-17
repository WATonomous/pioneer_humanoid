# humanoid_mujoco_scenes

Plain-MuJoCo scenes for the Pioneer arm — CPU only, no Isaac. One folder per scene, auto-discovered.
The arm comes from `pioneer_humanoid.mujoco_bimanual_arm` (the URDF plus the same joint limits and gains as
the Isaac config).

```bash
pip install mujoco pillow
pip install -e src/pioneer_humanoid -e src/simulation/mujoco_scenes
```

## Use a scene

```bash
# leader-arm teleop (needs a display; on macOS use mjpython)
python src/teleop/pioneer_leader_arm_teleop/pioneer_leader_arm_teleop.py --target mujoco --scene peg_insert

# headless check: render the scene with the arm at home
MUJOCO_GL=egl python -m humanoid_mujoco_scenes.preview --scene peg_insert --png peg.png
```

## Add a scene

```
humanoid_mujoco_scenes/my_scene/
├── __init__.py          # empty
└── scene.py
```

Every scene builds on the **lightbox workcell** (`workcell.py`), the same rig as Isaac's
`humanoid_rl_tasks.workcell.LightboxWorkcellCfg`: floor, the CAD lightbox (visual only, lit by its two fill
lights) and the 30.5-inch table, whose top collides as a box. `add_workcell(spec)` adds it; put your objects on
`TABLE_TOP_Z` (0.780 m: the lightbox's bottom panel on the 30.5-inch table), inside `TABLE_X` (0.235–0.845 m in front of the arm base) and `TABLE_HALF_Y`.

```python
# humanoid_mujoco_scenes/my_scene/scene.py
import mujoco
from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene


@scene("my_scene", camera=dict(lookat=[0.4, 0.3, TABLE_TOP_Z], distance=0.5, azimuth=200, elevation=-35))
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02] * 3, pos=[0.4, 0.3, TABLE_TOP_Z + 0.02])
```

- **`robot_pos`** — arm base placement; default `(0, 0, 1.1997)` puts the stand's feet on the floor (z=0).
- **`camera`** — optional MuJoCo free-camera fields for the initial view. From behind the table, keep the camera
  inside the lightbox (x < 0.8 m, z < 1.4 m) or its back panel hides the scene; from the front, the arm is in the way.
- **`step`** — optional `step(model, data)`, called by the teleop each control step before the physics, for
  mechanics a static model can't express (`zip_tie`'s one-way ratchet). Keep its state in `data` so a reset clears it.
- **`reset`** — optional `reset(model, data, rng)`, called at startup and after every reset to randomise
  the new episode (object placement, task order).
- **`progress`** — optional `progress(model, data) -> (index, total, instruction)` for multi-step tasks.
  The teleop prints each new step, and `--record` stores the instruction as each frame's `task` (LeRobot's
  per-frame task) plus a `subtask_index` feature; `index == total` once everything is done.
- **`condition`** + **`condition_names`** — optional `condition(model, data) -> np.ndarray`: the current instruction as
  numbers for a policy with no language input (`tidy_table`: just `done`). `--record`
  stores it as `observation.environment_state`, which LeRobot's ACT takes as an input.

## Scenes

| name | notes |
|------|-------|
| `bare` | Empty lightbox workcell + arm. |
| `drawer_stow` | Long horizon, 5 steps: open the drawer (pinch the tab on its front, pull), put the red, green and blue 4 cm blocks in it in the order announced, close it. Every reset shuffles the blocks' places (a strip beside the cabinet) and the order. Each step is checked and latched in order; `progress` gives the current step and its instruction. Laid out for the gripper pointing down: the jaws are 61 mm long and ~16 cm across fully open, so half-close it before reaching for a block. |
| `matcha` | Long horizon, 4 steps: spoon the matcha into the cup, pour the water in, whisk it, serve the cup on the tray. Matcha and water are 5 mm balls (green in a pre-filled ladle, blue in a pitcher). Tools have chunky 25 mm square handles sticking up, for the gripper pointing down; the ladle and pitcher handles are posts on the side away from the cup, so held at the top and tipped toward the cup (wrist pitch) they empty over the far lip. Checks: ≥5/8 matcha and ≥12/24 water in the cup, whisk tines moved 20 cm inside the cup (circles or zig-zag), cup upright on the tray with most of the drink. Reset shifts the cup and pitcher. ~0.35 s physics per simulated second (32 balls; the slowest scene). |
| `duplo` | Long horizon, 3 steps: stack the red, blue and yellow Duplo 2x4 bricks (64 x 32 x 19 mm) on the green baseplate in the colour order announced (first on the plate, each next one on the previous). Studs, walls and tubes are real collision geometry; the studs taper, so a brick put down within ~4 mm of the grid slides onto it. A brick seated square on the grid, overlapping what it sits on by at least one stud, is welded there (MuJoCo can't do the press fit); pulling it off with more than 6 N, or twisting it, releases it. Bricks lie in a row beside the plate, long side along X, for the gripper pointing down pinching their 32 mm width. Reset shuffles the order and the bricks' places. |
| `zip_tie` | A pre-threaded 300 × 3.6 mm nylon zip tie around four loose rods (spring-mounted, a few mm apart). The strap is one continuous chain from the head's root, round the rods, back through the slot and out as the tail toward the robot. Grab the tail ≳60 mm from the head (the jaws are 89 mm long) and pull: strap feeds through a one-way, toothed ratchet and the loop gathers then squeezes the rods (about 1 N to cinch). Success: `zip_tie.scene.is_tight(model, data)` (loop ≤ 107 mm, all rods touching). |
| `tidy_table` | Clear the table: put every object into the trash bin, in any order ("put everything in the trash bin"). Every reset lays out a messy table: 2 objects (box, cylinder or ball, any mix; each its own colour), 36-46 mm and about the same in all three directions, random mass, friction and yaw, some cylinders lying on their side, scattered over the left arm's reach (~18 x 16 cm, never touching). The bin is fixed on the arm's outer side: 24 x 14 cm inside and only 3 cm high, so an object let go anywhere over it lands inside. `progress` counts the objects binned (once each); `scene.episode_status(model, data)` gives that, `tidiness` (share in the bin now), `success` (all of them) and, as notes only, objects `dropped` or `toppled`. `condition` is one number, `done`. Overhead RGB camera `top` (640 x 480, world-mounted above the far outer corner, so the arm at home hides nothing); record it with `--cameras top,wrist_left`. Pointing down, the arm has no wrist roll and reaches only part of the layout's far side. Still on its own 0.705 m table, not the lightbox workcell (its recorded datasets were made on that table). Tests: `pytest src/simulation/mujoco_scenes/tests/test_tidy_table.py`. |
| `peg_insert` | 4 cm square peg, block with a square hole `CLEARANCE` (4 mm, 2 mm a side) wider. Peg and block sit inside the left arm's gripper-down reach (x 0.30–0.45, y 0.22–0.38); each reset shifts the peg (±15 mm) and block (±10 mm). `is_inserted`: peg in the hole ≥ 2 cm deep. |

## Differences from Isaac

- The lightbox meshes (`assets/props/lightbox_mujoco/*.stl`, one per CAD material, colours in `colors.json`) are
  converted from `assets/props/lightbox.usd`, in the USD's frame. If the USD changes, convert it again (usd-core:
  its meshes sit under instanced prims, so traverse instance proxies, and flip the faces of mirrored parts).
- The arm base is at the world origin, not Isaac's x = 0.15 m: the workcell is shifted to match, so positions are
  relative to the arm in both.

- Finger collisions are boxes fitted to each finger mesh (MuJoCo's convex hulls are rounded and let a held object slip).
- Contact settings for grasping (`_register.py`): 1 ms step, elliptic friction cones with `impratio` 10, and
  stiffer contacts (`solref` 0.004 s, `solimp` 0.95–0.99) on every geom a scene leaves at MuJoCo's defaults.
  Arm joints carry 0.01 kg m² armature (`mujoco_bimanual_arm.ARM_ARMATURE`). Without these a grasped object
  spins and slides out of the jaws and sinks millimetres into them.
- Arm self-collisions are off, as in Isaac.
- Robot cameras (`make_model(name, cameras={...})`) use the same mounts and lenses as Isaac (`pioneer_humanoid/arm_params.py`).
