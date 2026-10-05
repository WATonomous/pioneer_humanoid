# Locomotion: Pioneer humanoid V1 velocity tracking

Velocity-commanded bipedal locomotion for **Pioneer humanoid Simulation Model V1** in Isaac Lab. The agent receives base velocity commands $(v_x, v_y, \omega_z)$ and is rewarded for tracking them while staying upright.

Asset: `assets/whole_body_humanoid/` (SolidWorks URDF export).
Training uses `whole_body_humanoid.urdf`, which adds a Z-up `base` link and a fixed joint so the CAD Y-long frame stands in Isaac without a spawn rotation.

**Environments**

| Task ID | Terrain | Mode |
| :--- | :--- | :--- |
| `Isaac-Locomotion-Flat-PioneerHumanoid-v0` | Plane | Train |
| `Isaac-Locomotion-Flat-PioneerHumanoid-Play-v0` | Plane | Play |
| `Isaac-Locomotion-Rough-PioneerHumanoid-v0` | Procedural rough | Train |
| `Isaac-Locomotion-Rough-PioneerHumanoid-Play-v0` | Procedural rough | Play |
| `Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-v0` | Rough without stairs, corrected knees and selective self-collision | Train |
| `Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0` | Matching rough-no-stairs recipe | Play |
| `Isaac-Locomotion-Stairs-PioneerHumanoid-v0` | Twelve steps up, raised walkway, twelve steps down | Train scaffold |
| `Isaac-Locomotion-Stairs-PioneerHumanoid-Play-v0` | Fixed 8 cm full-course inspection | Play scaffold |

Legacy aliases `Isaac-Velocity-*` register the same configs.

## Train & play

Run inside the **`simulation_isaac`** container (Isaac Lab 2.3.2 / Sim 5.1). Host setup: [`docker/simulation/isaac_lab/README.md`](../../../../../docker/simulation/isaac_lab/README.md).

```bash
# Host
cd ~/Desktop/humanoid && ./watod up -d && ./watod -t simulation_isaac

# Inside container — from $HUMANOID_ROOT
cd $HUMANOID_ROOT

# Train — flat terrain
rl-train --task=Isaac-Locomotion-Flat-PioneerHumanoid-v0 --headless

# Play — loads latest under outputs/rl/pioneer_humanoid_flat/
rl-play --task=Isaac-Locomotion-Flat-PioneerHumanoid-Play-v0 --num_envs=1

# Play — specific checkpoint
rl-play --task=Isaac-Locomotion-Flat-PioneerHumanoid-Play-v0 --num_envs=1 \
  --checkpoint outputs/rl/pioneer_humanoid_flat/<run>/model_<iter>.pt
```

Rough-terrain variants: replace `Flat` with `Rough` and use `outputs/rl/pioneer_humanoid_rough/`.

## Rough-no-stairs source-only recipe

This contribution supplies task source, not a trained policy or a claim of
hardware readiness. The selective-knee recipe has **19 active, nonzero reward
terms**, including the original squared foot/opposite-calf clearance, knee
target overshoot, and own-swing flexion costs. Weights and parameters are defined
in [rough_env_cfg.py](config/pioneer_humanoid_v1/rough_env_cfg.py) and
[knee_shaping.py](mdp/knee_shaping.py). The local swing-timing and normalized
swing-clearance experiments are not included.

Inside the team's `simulation_isaac` container, wait until the GPU is free and
start with a **fresh** 200-update wiring/learning smoke:

```bash
cd "$HUMANOID_ROOT"
"$ISAACLAB/isaaclab.sh" -p "$RL_RUNNERS/train.py" \
  --task Isaac-Velocity-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-v0 \
  --headless --num_envs 256 --max_iterations 200 --seed 43 \
  --run_name rough_no_stairs_smoke_seed43 --fail_on_nonfinite_grad
```

Inspect startup, finite learning signals, and the saved recipe before increasing
the budget. An optional longer **fresh** run is:

```bash
"$ISAACLAB/isaaclab.sh" -p "$RL_RUNNERS/train.py" \
  --task Isaac-Velocity-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-v0 \
  --headless --num_envs 6144 --max_iterations 4001 --seed 43 \
  --run_name rough_no_stairs_fresh_seed43 --fail_on_nonfinite_grad
```

Neither command resumes a policy. Tune `--num_envs` to available GPU memory;
more environments also change samples per PPO update, not just speed. Runs,
checkpoints, and saved `params/env.yaml` / `params/agent.yaml` are under ignored
`outputs/rl/pioneer_humanoid_rough_no_stairs_selective_knee_shape/`. Checkpoint
numbering starts at zero: a 200-update fresh run normally ends at `model_199.pt`.

To view your own checkpoint from this recipe, set `ROUGH_CHECKPOINT` to its exact
path and use a working GUI/display:

```bash
ROUGH_CHECKPOINT="/absolute/path/to/your/run/model_<iteration>.pt"
"$ISAACLAB/isaaclab.sh" -p "$RL_RUNNERS/play.py" \
  --task Isaac-Velocity-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0 \
  --checkpoint "$ROUGH_CHECKPOINT" --num_envs 9 --real-time
```

Use one environment for a closer inspection. `Isaac-Locomotion-*` aliases are
equivalent. Plain `Rough` is a different recipe; do not select it merely to load
an incompatible checkpoint.

Checkpoint compatibility requires the actor **and** critic observation
normalizers from the saved policy, 235 policy observations, 12 ordered joint
actions, and the matching task's scale/default-offset mapping. Both processed
knee targets are bounded to `[-0.95, -0.05]` radians after scaling and offsetting.
Negative knee position is flexion. Selective physical self-collision permits
only each foot against the opposite calf; terrain-only foot sensors prevent
those self-contacts from being mistaken for ground stance.

The supplied asset corrects both knee axes/limits, uses a 0.84 m nominal root
height, and mirrors the nominal ankle-pitch signs. These shared asset changes
also affect legacy Flat/Rough tasks. The older checked-in flat model3900 and
rough model2999 were trained with different geometry and are **not validated
against this corrected asset**; their unchanged filenames are not a
compatibility guarantee. Do not mix a policy trained with this recipe with an
older URDF/USD or remove normalizers to make another task load it.

Evaluate learned stepping symmetry, foot paths, falls, turning, and sustained
walking rather than relying on total reward or a short demonstration. Source
and wiring tests do not establish locomotion or hardware robustness.

The [stairs handoff](STAIRS.md) inherits this 19-reward recipe and provides a
complete ascent → raised walkway → descent course. It is a training scaffold;
no stairs policy is supplied. Its validation commands distinguish CPU checks,
simulator wiring, and actual locomotion performance.

**Spawn / joint smoke checks**

```bash
$ISAACLAB/isaaclab.sh -p $RL_RUNNERS/diagnose_spawn.py \
  --task=Isaac-Locomotion-Flat-PioneerHumanoid-v0 --headless --num_envs=1 --steps=10

$ISAACLAB/isaaclab.sh -p $RL_RUNNERS/diagnose_joints.py \
  --task=Isaac-Locomotion-Flat-PioneerHumanoid-v0 --headless --num_envs=1
```

## Joints & bodies

| Role | Names |
| :--- | :--- |
| Root body | `base` (merged upright root from isaac URDF) |
| Feet (contacts) | `Foot_L`, `Foot_R` |
| Hip flexion | `Hip_F_L`, `Hip_F_R` |
| Hip abduction | `Hip_A_L`, `Hip_A_R` |
| Hip rotation | `Hip_R_L`, `Hip_R_R` |
| Knee | `Knee_L`, `Knee_R` |
| Ankle pitch / roll | `Ankle_P_*`, `Ankle_R_*` |

Config: `pioneer_humanoid/whole_body.py`, tasks under `config/pioneer_humanoid_v1/`.
