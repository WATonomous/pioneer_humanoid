# humanoid_isaac_scenes

Manipulation scenes for **pioneer bimanual-arm teleop data collection** — one
folder per scene, auto-discovered. Adding a scene requires touching *only* this
package.

## Add a scene

```
humanoid_isaac_scenes/my_scene/
├── __init__.py          # empty
└── scene.py
```

Every manipulation scene builds on the **lightbox workcell**
(`humanoid_rl_tasks.workcell.LightboxWorkcellCfg`): floor, light, the CAD
lightbox, the 30.5-inch table's collision, and the arm on its floor stand at
`ROBOT_BASE_POS`. Subclass it and add only your objects, on `TABLE_TOP_Z`:

```python
# humanoid_isaac_scenes/my_scene/scene.py
from dataclasses import MISSING

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from humanoid_isaac_scenes import scene
from humanoid_rl_tasks.workcell import ROBOT_BASE_POS, TABLE_TOP_Z, WORKCELL_CAMERA, LightboxWorkcellCfg


@scene("my_scene", robot_pos=ROBOT_BASE_POS, camera=WORKCELL_CAMERA)
@configclass
class MySceneCfg(LightboxWorkcellCfg):
    widget = RigidObjectCfg(...)  # init_state pos z = TABLE_TOP_Z (+ half-height if centre-origin)
```

- **`robot_pos`** — pass `ROBOT_BASE_POS` so the arm stands where the workcell
  expects it. The table front edge is at `TABLE_X_MIN` (~0.39 m); objects
  ~0.30 m in front of the arm base (x ≈ 0.45) are in reach.
- **`camera`** — optional `(eye, target)` for the teleop initial view.
- A scene that needs a different room (e.g. `garment_fold`'s apartment) can
  still subclass plain `InteractiveSceneCfg` instead.

That's it. `keyboard_teleop --scene my_scene` now works — no edits to
`keyboard_teleop`, or the Dockerfile.

## Use a scene

```
./watod -t simulation_isaac
# in the container:
keyboard_teleop --scene <name>      # pass an unknown name to list them
```

## What lives here vs. not

- **Here:** lightweight teleop/data-collection scenes — mostly a `scene.py`.
- **Not here:** full task packages with training/eval/recording infra
  (`so101_vial_task`) and the RL tasks (`humanoid_rl_tasks/` — `inhand`,
  `locomotion`, `push_block`). A scene that doubles as an
  RL task keeps its geometry in `humanoid_rl_tasks/<task>/scene.py`; the folder
  here is just a one-liner that registers it for teleop (see `push_block/`).

## Scenes

| name | notes |
|------|-------|
| `bare` | Empty lightbox workcell — the default `--scene`. |
| `push` | Ramp-box + block (the RL `push_block` task's scene, re-registered). |
| `garment_fold` | Own apartment / worksurface, not the lightbox. Needs `humanoid_garment_fold` installed. |
| `vial_rack` | Lightbox workcell + `Vial_rack_simple.usda` + 3 loose `Vial_opaque.usda`. Rack/vial xy is a first pass — reach-tune against the LEFT arm while driving it. Assets: `assets/lerobot/so101_vial_task/usd/`. |
