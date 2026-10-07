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

```python
# humanoid_mujoco_scenes/my_scene/scene.py
import mujoco
from humanoid_mujoco_scenes import add_floor, scene


@scene("my_scene", camera=dict(lookat=[0.4, 0.3, 0.75], distance=1.0, azimuth=200, elevation=-35))
def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.5, 0.35], pos=[0.5, 0, 0.35])
```

- **`robot_pos`** — arm base placement; default `(0, 0, 1.1997)` puts the stand's feet on the floor (z=0).
- **`camera`** — optional MuJoCo free-camera fields for the initial view.
- **`step`** — optional `step(model, data)`, called by the teleop each control step before the physics, for
  mechanics a static model can't express (`zip_tie`'s one-way ratchet). Keep its state in `data` so a reset clears it.
- **`reset`** — optional `reset(model, data, rng)`, called at startup and after every reset to randomise
  the new episode (object placement, task order).
- **`progress`** — optional `progress(model, data) -> (index, total, instruction)` for multi-step tasks.
  The teleop prints each new step, and `--record` stores the instruction as each frame's `task` (LeRobot's
  per-frame task) plus a `subtask_index` feature; `index == total` once everything is done.
- **`condition`** + **`condition_names`** — optional `condition(model, data) -> np.ndarray`: the current instruction as
  numbers for a policy with no language input (`tidy_table`: one-hot target colour and shape, plus `done`). `--record`
  stores it as `observation.environment_state`, which LeRobot's ACT takes as an input.

## Scenes

| name | notes |
|------|-------|
| `bare` | Floor + arm. |
| `drawer_stow` | Long horizon, 5 steps: open the drawer (pinch the tab on its front, pull), put the red, green and blue 4 cm blocks in it in the order announced, close it. Every reset shuffles the blocks' places (a strip beside the cabinet) and the order. Each step is checked and latched in order; `progress` gives the current step and its instruction. Laid out for the gripper pointing down: the jaws are 61 mm long and ~16 cm across fully open, so half-close it before reaching for a block. |
| `matcha` | Long horizon, 4 steps: spoon the matcha into the cup, pour the water in, whisk it, serve the cup on the tray. Matcha and water are 5 mm balls (green in a pre-filled ladle, blue in a pitcher). Tools have chunky 25 mm square handles sticking up, for the gripper pointing down; the ladle and pitcher handles are posts on the side away from the cup, so held at the top and tipped toward the cup (wrist pitch) they empty over the far lip. Checks: ≥5/8 matcha and ≥12/24 water in the cup, whisk tines moved 20 cm inside the cup (circles or zig-zag), cup upright on the tray with most of the drink. Reset shifts the cup and pitcher. ~0.35 s physics per simulated second (32 balls; the slowest scene). |
| `duplo` | Long horizon, 3 steps: stack the red, blue and yellow Duplo 2x4 bricks (64 x 32 x 19 mm) on the green baseplate in the colour order announced (first on the plate, each next one on the previous). Studs, walls and tubes are real collision geometry; the studs taper, so a brick put down within ~4 mm of the grid slides onto it. A brick seated square on the grid is welded there (MuJoCo can't do the press fit); pulling it off with more than 6 N, or twisting it, releases it. Bricks lie in a row beside the plate, long side along X, for the gripper pointing down pinching their 32 mm width. Reset shuffles the order and the bricks' places. |
| `zip_tie` | A pre-threaded 300 × 3.6 mm nylon zip tie around four loose rods (spring-mounted, a few mm apart). The strap is one continuous chain from the head's root, round the rods, back through the slot and out as the tail toward the robot. Grab the tail ≳60 mm from the head (the jaws are 89 mm long) and pull: strap feeds through a one-way, toothed ratchet and the loop gathers then squeezes the rods (about 1 N to cinch). Success: `zip_tie.scene.is_tight(model, data)` (loop ≤ 107 mm, all rods touching). |
| `tidy_table` | Long horizon, 3-4 steps (level 1 of a table-tidying task): put each object (box, cylinder, ball) into a bin, any of the three, in the order announced, e.g. "put the red ball in a bin". Every reset lays out a messy table: objects (≤2 per shape, each its own colour), sizes (25-50 mm across), mass, friction, any yaw, some cylinders lying on their side, scattered over the left arm's gripper-down reach (~18 x 16 cm, never touching, clear of the bins), and a new order; bins are fixed. Fails and keeps flags for an object resting on the table after being picked up (`dropped`), a box or cylinder tipped from how it was set out (`toppled`) and one binned before its turn (`early`); `scene.episode_status(model, data)` gives these, `tidiness` (share binned) and `success`. Overhead RGB camera `top` (640 x 480, world-mounted above the far outer corner, so the arm at home hides nothing); record it with `--cameras top,wrist_left`. The arm has no wrist roll: pointing down it turns only about -45..+30 deg about vertical. Tests: `pytest src/simulation/mujoco_scenes/tests/test_tidy_table.py`. |
| `peg_insert` | Table (top 0.705 m, as in Isaac `push`), 4 cm square peg, block with a square hole `CLEARANCE` (1 mm) wider. Peg and block sit inside the left arm's gripper-down reach (x 0.30–0.45, y 0.22–0.38). |

## Differences from Isaac

- Finger collisions are boxes fitted to each finger mesh (MuJoCo's convex hulls are rounded and let a held object slip).
- Contact settings for grasping (`_register.py`): 1 ms step, elliptic friction cones with `impratio` 10, and
  stiffer contacts (`solref` 0.004 s, `solimp` 0.95–0.99) on every geom a scene leaves at MuJoCo's defaults.
  Arm joints carry 0.01 kg m² armature (`mujoco_bimanual_arm.ARM_ARMATURE`). Without these a grasped object
  spins and slides out of the jaws and sinks millimetres into them.
- Arm self-collisions are off, as in Isaac.
- Robot cameras (`make_model(name, cameras={...})`) use the same mounts and lenses as Isaac (`pioneer_humanoid/arm_params.py`).
