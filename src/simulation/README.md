# `src/simulation/`

Simulation packages: RL tasks, teleop data-collection scenes, the training runners,
and datagen glue. The Isaac Lab ones are `pip install -e`'d into the `simulation_isaac`
image (see `docker/simulation/isaac_lab/`); `mujoco_scenes/` and `badminton/` run on MuJoCo.

```
src/simulation/
├── humanoid_rl/            # runners — train / play / distill / diagnose
│   └── humanoid_rl/scripts/
├── humanoid_rl_tasks/      # RL tasks, flat — one folder per task
│   └── humanoid_rl_tasks/
│       ├── inhand/         locomotion/
│       └── push_block/     # PPO + vision distillation; also a teleop scene
├── isaac_scenes/           # Isaac teleop data-collection scenes — @scene-discovered
│   └── humanoid_isaac_scenes/  #   bare/  vial_rack/  push_block/
├── mujoco_scenes/          # plain-MuJoCo (CPU) scenes — @scene-discovered
│   └── humanoid_mujoco_scenes/ #   bare/  peg_insert/
├── so101_vial_task/        # SO101 imitation-learning task
└── badminton/              # mjlab (MuJoCo Warp) badminton receive RL — see badminton/README.md
```

Robot URDF/USD/meshes and scene props live at the repo-root **`assets/`** (backend-neutral —
not tied to an Isaac Lab package): `assets/{pioneer_bimanual_arm, pioneer_hand,
whole_body_humanoid, props, lerobot}/`. Live RL checkpoints go to `outputs/rl/` (gitignored); frozen baselines are in the repo-root `models/`.

## RL task vs teleop scene — where does new work go?

| you're adding… | put it in | registered by |
|---|---|---|
| an RL task (has a reward, gets PPO-trained) | `humanoid_rl_tasks/<task>/` | `import_packages` in `humanoid_rl_tasks/__init__.py` — auto |
| a teleop-only scene (collect demos, no reward) | `humanoid_isaac_scenes/<name>/scene.py` with `@scene("<name>")` | `humanoid_isaac_scenes` discovery — auto |
| a scene that is **both** (like `push_block`) | geometry + env live in `humanoid_rl_tasks/<task>/scene.py`; add a 1-line `humanoid_isaac_scenes/<name>/scene.py` that re-registers it (`from humanoid_rl_tasks.<task>.scene import ...; scene("<name>")(TheSceneCfg)`) | both, via the shim |

The scene cfg declares `robot = MISSING`; the RL env cfg and the teleop registry
each plug their own arm in. Geometry constants live in exactly one `scene.py` —
never hand-copied between the RL env and teleop.

## Invoking a scene from teleop

Teleop scripts resolve `--scene <name>` through `humanoid_isaac_scenes`:

```bash
# inside the simulation_isaac container
cd src/teleop/keyboard_teleop
isaaclab.sh -p keyboard_teleop.py --scene push        # or: bare, vial_rack, …
```

`keyboard_teleop` uses this today; `quest_teleop/sim` and `task_space_controller`
will move onto the same `--scene` registry next.

## Training

```bash
cd $HUMANOID_ROOT     # checkpoints land in $HUMANOID_ROOT/outputs/rl/
rl-train --task=Isaac-Locomotion-Flat-PioneerHumanoid-v0 --headless
rl-play  --task=Isaac-Repose-Cube-PioneerHand-Play-v0 --num_envs=1
```

Per-task notes: `humanoid_rl_tasks/humanoid_rl_tasks/<task>/*.md`. Full walkthrough:
`docker/simulation/isaac_lab/README.md`.

Pioneer humanoid walking: [source-only rough-no-stairs training and playback recipe](humanoid_rl_tasks/humanoid_rl_tasks/locomotion/locomotion.md#rough-no-stairs-source-only-recipe).
Stairs research starter: [twelve steps up, a raised walkway, and twelve steps down](humanoid_rl_tasks/humanoid_rl_tasks/locomotion/STAIRS.md).
Native Windows inspection: [CPU policy export and portable GUI viewer](humanoid_rl/NATIVE_VIEWER.md).
