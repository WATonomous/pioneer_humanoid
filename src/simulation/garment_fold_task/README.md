# humanoid-garment-fold

Bimanual **garment-folding** Isaac Lab task for the WATonomous `pioneer_bimanual_arm_8dof`,
ported from the LeHome Challenge. Fold long/short tops and long/short pants
on a table; success is judged on the cloth's particle geometry.

## Status

Runs end-to-end: scene + garment load, a full episode runs clean, and
`keyboard_teleop.py --scene garment_fold` lets you drive the arm and touch
the garment live (see "Interactive teleop" below).

Known gaps:
* **Grasping/pinching untuned** -- the gripper hasn't been tuned against fabric.
* **Arm base pose is a trade-off**: `(0, -0.40, 0.95)` was picked for reach,
  which re-sinks the stand ~25cm into the floor -- see `garment_pioneer_cfg.py`.

## Layout

```
garment_fold_task/
├── NOTICE                       what was vendored/modified, §4 of the repo's own Apache-2.0 LICENSE
├── pyproject.toml
├── vendor_assets/
│   ├── garments/Release/       1 sample garment per category (committed, ~6 MB)
│   ├── scenes/Table038/        fallback table (committed, ~1.2 MB)
│   └── scenes/marble/          Scene_00_Apartment.usd — .gitignore'd, you copy it
└── humanoid_garment_fold/
    ├── tasks/
    │   ├── garment_env.py            VENDORED — the GarmentEnv base
    │   ├── garment_env_cfg.py        VENDORED — base cfg (robot fields removed)
    │   ├── challenge_garment_loader.py  VENDORED
    │   ├── garment_pioneer_cfg.py    NEW — pioneer config
    │   ├── garment_pioneer_env.py    NEW — pioneer env (subclass)
    │   ├── teleop_scene.py           NEW — keyboard_teleop.py --scene registration
    │   └── __init__.py               gym.register(...)
    ├── assets/
    │   ├── garment_object.py         VENDORED — PhysX particle-cloth garment
    │   └── scene.py                  adapted — scene USD paths
    ├── utils/
    │   ├── success_checker_garment.py  VENDORED — fold-quality check
    │   └── logger.py                 NEW — console shim
    └── config/particle_garment_cfg.yaml  VENDORED — particle solver params
```

## The retarget

Upstream drives two SO101 follower arms (`left_arm`/`right_arm`, 12-dim action).
This drives **one `pioneer_bimanual_arm_8dof`** instead: per arm 7 joints
`{side}_shoulder_pitch .. {side}_wrist_yaw` + `{side}_gripper`, EE `{side}_link7`.
Action is 16-dim, the `pioneer_8dof` dataset layout: `[left ×7, left gripper,
right ×7, right gripper]` joint-position targets, grippers 0 = open, 1 = closed;
`observation.state` the same. Names/cfg come from `pioneer_humanoid`
(`arm_8dof_params.py`, `bimanual_arm_8dof.py`).

The pioneer arm's front axis is `+X`; the garment sits at world `~(0,0,0.63)`,
so the base is rotated +90° about Z. `_build_worksurface()` tries, in order:
optional NuRec backdrop, `Scene_00_Apartment.usd` (default — gitignored,
~19MB, copy it in), ground + vendored `Table038.usd` (in front of the arm), ground only.

## Run it

Use the **`simulation_isaac_garment`** watod module, not `simulation_isaac` --
same published `isaac_lab` image plus two fixes (Isaac Lab 2.3.0 downgrade for
a `TiledCamera` freeze bug, `open3d` for the success checker) that aren't safe
to bake into the shared Dockerfile without broader testing. See
`docker/simulation/isaac_lab/isaac_lab_garment.Dockerfile` for exactly what
and why.

```bash
# Copy the whole Assets/scenes/marble/ folder from the LeHome challenge into
# vendor_assets/scenes/marble/ first -- the .usd references the .usdz next to
# it and won't load without both.
ACTIVE_MODULES="simulation_isaac_garment" ./watod up -d
./watod -t simulation_isaac_garment_dev
# inside the container:
pip install -e src/pioneer_humanoid src/simulation/garment_fold_task --no-deps --no-build-isolation
```

Verified against a real `docker build` of `isaac_lab_garment.Dockerfile`, not
just a hand-patched container. See "Interactive teleop" below to actually try it.

### Interactive teleop

`src/teleop/keyboard_teleop/keyboard_teleop.py --scene garment_fold` drives
the real arm into this scene with a live keyboard -- useful for personally
checking gripper-to-garment contact. Needs a real display (not headless):

```bash
pip install -e src/pioneer_humanoid src/simulation/garment_fold_task src/simulation/isaac_scenes --no-deps --no-build-isolation
cd src/teleop/keyboard_teleop
GARMENT_NAME=Top_Long_Seen_1 isaaclab.sh -p keyboard_teleop.py --scene garment_fold --enable_cameras
```

The garment (a particle cloth, not a declarative `AssetBaseCfg`) is built by a
`post_init(scene, sim)` hook -- a small, generic addition to
`humanoid_isaac_scenes/_register.py` and `keyboard_teleop.py`, a no-op for
every scene that doesn't need one. `tasks/teleop_scene.py` is the single
source of truth.

## To finish

1. **Reach works, grasping/pinching doesn't.** IK on the Table038 fallback: each
   arm puts its tool point on its half of the garment (< 1 mm), top-down only
   near the robot (the centre and far corners need the gripper tilted). The
   grippers still need tuning to actually pinch fabric.
2. **Full garment set**: `hf download lehome/asset_challenge` → `garment_cfg_base_path`.
3. **Data + training**: pioneer teleop → record demos → `lerobot-train` with
   LeHome's ACT / DP / SmolVLA configs.

## Notes from the port

* `Scene_00_Apartment.usd` composes only a table when opened minimally; the
  photoreal apartment (NuRec `.usdz`) renders through the full Isaac Sim
  pipeline. Both are in the LeHome `Assets/scenes/marble/`; `scene_v1.usd` (the
  room mesh) ships in neither the repo nor the HF dataset.
