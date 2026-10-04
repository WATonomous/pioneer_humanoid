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

## Scenes

| name | notes |
|------|-------|
| `bare` | Floor + arm. |
| `zip_tie` | A pre-threaded zip tie around four vertical rods (head fixed). Grab the tail (on edge, jaws pinch its faces) and pull it toward the robot: the tail slides out of the head against 1 N of tooth drag on a one-way ratchet, and the loop shrinks onto the rods. Success: `zip_tie.scene.is_tight(model, data)` (loop within 4 mm of snug). |
| `peg_insert` | Table (top 0.705 m, as in Isaac `push`), 4 cm square peg, block with a square hole `CLEARANCE` (1 mm) wider. Peg and block sit inside the left arm's gripper-down reach (x 0.30–0.45, y 0.22–0.38). |

## Differences from Isaac

- Finger collisions are boxes fitted to each finger mesh (MuJoCo's convex hulls are rounded and let a held object slip).
- Contact settings for grasping (`_register.py`): 1 ms step, elliptic friction cones with `impratio` 10, and
  stiffer contacts (`solref` 0.004 s, `solimp` 0.95–0.99) on every geom a scene leaves at MuJoCo's defaults.
  Arm joints carry 0.01 kg m² armature (`mujoco_bimanual_arm.ARM_ARMATURE`). Without these a grasped object
  spins and slides out of the jaws and sinks millimetres into them.
- Arm self-collisions are off, as in Isaac.
- Robot cameras (`make_model(name, cameras={...})`) use the same mounts and lenses as Isaac (`pioneer_humanoid/arm_params.py`).
