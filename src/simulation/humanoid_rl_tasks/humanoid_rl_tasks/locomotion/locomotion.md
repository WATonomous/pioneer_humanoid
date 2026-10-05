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
| `Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0` | Matching frozen model4000 inspection | Play |
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

## Frozen rough-no-stairs baseline: model4000

The shared checkpoint is
[model_4000.pt](../../../../../models/pioneer_humanoid_rough_no_stairs_selective_knee_shape/model_4000.pt),
with [provenance and recipe](../../../../../models/pioneer_humanoid_rough_no_stairs_selective_knee_shape/README.md)
and a [frozen manifest](../../../../../models/pioneer_humanoid_rough_no_stairs_selective_knee_shape/manifest.json).
It comes from a fresh seed-43 rough run with 6,144 environments and 4,001 PPO
updates; the filename uses zero-based update numbering. This is a simulation
baseline, not a claim of hardware readiness or stair-climbing performance.

Run inside the team's `simulation_isaac` container, with a working GUI/display
and no other training job or viewer using the GPU:

```bash
cd "$HUMANOID_ROOT"
"$ISAACLAB/isaaclab.sh" -p "$RL_RUNNERS/play.py" \
  --task Isaac-Velocity-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0 \
  --checkpoint models/pioneer_humanoid_rough_no_stairs_selective_knee_shape/model_4000.pt \
  --num_envs 9 --real-time
```

Use `--num_envs 1` for a closer inspection. The `Isaac-Locomotion-*` alias is
equivalent. Specify this exact checkpoint and task: plain `Rough` is a different
terrain/observation-normalization recipe, and latest-run lookup can select an
unrelated local experiment.

The retained recipe has **19 active, nonzero reward terms**. It keeps the original
foot/opposite-calf squared-clearance term and the knee overshoot and own-swing
flexion terms; it does not include the later swing-timing reward or normalized
swing-clearance replacement. No reward-removal experiment is part of this
baseline. The weights and parameters are recorded in the manifest and defined
by [rough_env_cfg.py](config/pioneer_humanoid_v1/rough_env_cfg.py) and
[knee_shaping.py](mdp/knee_shaping.py).

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
compatibility guarantee. Do not mix model4000 with an older URDF/USD or remove
normalizers to make another task load it.

Observed limitations remain: unequal knee/step patterns, an occasional inward
left-foot lift or pause, and falls on some uneven terrain. Short fixed-cell
evaluation and a successful demonstration clip do not establish long-duration,
turning, staircase, or hardware robustness. Keep future gait experiments
separate from this frozen baseline.

The [stairs handoff](STAIRS.md) inherits this 19-reward baseline and provides a
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
