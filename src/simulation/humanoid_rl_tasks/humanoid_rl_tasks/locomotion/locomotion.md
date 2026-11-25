# Locomotion: Pioneer humanoid V1 velocity tracking

Velocity-commanded bipedal locomotion for **Pioneer humanoid Simulation Model V1** in Isaac Lab. The agent receives base velocity commands $(v_x, v_y, \omega_z)$ and is rewarded for tracking them while staying upright.

Asset: `assets/pioneer_whole_body_humanoid/` (whole-body URDF). The v1 flat,
rough, rough-no-stairs and Onshape tasks spawn `WHOLE_BODY_WALKER_CFG` from
`pioneer_humanoid/whole_body.py`, which walks the 12 leg joints with the arms
and head held rigid; the arm-swing variant spawns `WHOLE_BODY_ARM_SWING_CFG`
from the same module. Historical baseline notes:
[robot baseline isolation](../../../../../assets/old_leg_only_humanoid/RL_BASELINE.md).

**Environments**

| Task ID | Terrain | Mode |
| :--- | :--- | :--- |
| `Isaac-Locomotion-Flat-PioneerHumanoid-v0` | Plane | Train |
| `Isaac-Locomotion-Flat-PioneerHumanoid-Play-v0` | Plane | Play |
| `Isaac-Locomotion-Rough-PioneerHumanoid-v0` | Procedural rough | Train |
| `Isaac-Locomotion-Rough-PioneerHumanoid-Play-v0` | Procedural rough | Play |
| `Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-v0` | Rough without stairs, corrected knees and selective self-collision | Train |
| `Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0` | Matching rough-no-stairs recipe | Play |
| `Isaac-Locomotion-StairsStraight-PioneerHumanoid-v0` | Onshape straight staircase USD course | Train scaffold |
| `Isaac-Locomotion-StairsStraight-PioneerHumanoid-Play-v0` | Fixed Onshape straight course inspection | Play scaffold |
| `Isaac-Locomotion-StairsStraightArmSwing-PioneerHumanoid-v0` | Onshape straight course, whole-body walker with learned arm swing (16 actions) | Train |
| `Isaac-Locomotion-StairsStraightArmSwing-PioneerHumanoid-Play-v0` | Matching arm-swing play config | Play |
| `Isaac-Locomotion-StairsStraightFoothold-PioneerHumanoid-v0` | Onshape straight course, foothold-conditioned perception ("Watch Your Step"), arms scripted | Train |
| `Isaac-Locomotion-StairsStraightFoothold-PioneerHumanoid-Play-v0` | Matching foothold play config (loads `models/pioneer_humanoid_stairs_foothold/`) | Play |
| `Isaac-Locomotion-StairsSpiral-PioneerHumanoid-v0` | Onshape spiral staircase USD course | Train scaffold |
| `Isaac-Locomotion-StairsSpiral-PioneerHumanoid-Play-v0` | Fixed Onshape spiral course inspection | Play scaffold |
| `Isaac-Locomotion-StairsSpiralArmSwing-PioneerHumanoid-v0` | Onshape spiral course with learned arm swing (16 actions) | Train |
| `Isaac-Locomotion-StairsSpiralArmSwing-PioneerHumanoid-Play-v0` | Matching spiral arm-swing play config | Play |
| `Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-v0` | Onshape spiral course, foothold-conditioned perception | Train |
| `Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0` | Matching spiral foothold play config | Play |

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

Rough-terrain variants: replace `Flat` with `Rough` and use
`outputs/rl/pioneer_humanoid_rough/`.

Legacy Flat/Rough task IDs, log namespaces, observation normalization, and robot
configuration follow upstream. Their geometry is deliberately not changed by
the rough-no-stairs contribution. Policies still require the robot definition
used for training; a matching log directory or normalizer is not a geometry
compatibility guarantee.

## Rough-no-stairs source-only recipe

This contribution supplies task source, not a trained policy or a claim of
hardware readiness. The selective-knee recipe has **19 active, nonzero reward
terms**, including the original squared foot/opposite-calf clearance, knee
target overshoot, and own-swing flexion costs. Weights and parameters are defined
in [rough_env_cfg.py](config/pioneer_humanoid_v1/rough_env_cfg.py) and
[knee_shaping.py](mdp/knee_shaping.py). The local swing-timing and normalized
swing-clearance experiments are not included.

The final recipe is defined directly by one rewards config, one training config,
its PLAY subclass, and one normalized PPO runner, without intermediate tuning
stages. Knee terms resolve joint/action mappings at manager setup and own their
caches; rollout calls read live actions, scale/offset, joint state and contact
timers. Terrain-only sensors update filtered forces and stance timers once per
sensor update. Unused foot/opposite-calf diagnostic sensors are not instantiated.

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

If WSL/Docker headless training works but you need a Windows GUI, follow the
[CPU export and native Windows viewer guide](../../../humanoid_rl/NATIVE_VIEWER.md).
It takes an explicit trusted checkpoint, exports TorchScript plus a manifest,
and runs native Windows physics without committing models or local setup.

Checkpoint compatibility requires the actor **and** critic observation
normalizers from the saved policy, 235 policy observations, 12 ordered joint
actions, and the matching task's scale/default-offset mapping. The arm-swing
variant adds four joints and therefore needs 247 observations and 16 ordered
actions; use `tools/rl_sessions/expand_walker_checkpoint.py` to convert a
12-action checkpoint rather than loading one directly. Both processed
knee targets are bounded to `[-0.95, -0.05]` radians after scaling and offsetting.
Negative knee position is flexion. Selective physical self-collision permits
only each foot against the opposite calf; terrain-only foot sensors prevent
those self-contacts from being mistaken for ground stance.

The recipe uses the whole-body walker (`WHOLE_BODY_WALKER_CFG` in
`pioneer_humanoid/whole_body.py`) on
`assets/pioneer_whole_body_humanoid/pioneer_whole_body_humanoid.urdf`, with
every non-leg joint turned into a fixed joint at its zero pose (arms hanging at
the sides). It retains the recipe's knee axes/limits, 0.84 m nominal root height,
mirrored nominal ankle-pitch signs, and actuator settings. Isaac writes an
Isaac-specific URDF copy and converts it on first launch into ignored
`outputs/rl/assets/pioneer_whole_body_humanoid/`; generated USD is not checked
in, and later launches reuse matching conversion output. Use the pinned team
simulator stack when regenerating it, rather than substituting another robot's
USD or assuming conversion across simulator versions is identical.
See the [baseline compatibility notes](../../../../../assets/old_leg_only_humanoid/RL_BASELINE.md)
before changing its geometry. Older checked-in flat model3900 and rough
model2999 are not policies for this recipe. Do not mix a policy trained with
this recipe with another URDF/USD or remove normalizers to make another task
load it. No trained policy is included here.

Evaluate learned stepping symmetry, foot paths, falls, turning, and sustained
walking rather than relying on total reward or a short demonstration. Source
and wiring tests do not establish locomotion or hardware robustness.

The Onshape stair courses inherit this 19-reward recipe and use the committed
merged terrain USDs in `assets/staircase_onshape/`. The STEP exports and raw
conversion intermediates are intentionally not in the repo; only
`*_terrain.usd` and the profile/geometry JSONs are needed at runtime. The plain
straight and spiral tasks are training scaffolds; no trained Onshape policy is
supplied. Why each course reward/termination is set the way it is (with the
measurements behind it) is in [STAIR_REWARD_NOTES.md](STAIR_REWARD_NOTES.md).

The straight course also has a learned-arm-swing variant
(`StairsStraightArmSwing`) that frees `l_shoulder_pitch`, `r_shoulder_pitch`,
`l_elbow` and `r_elbow` (16 actions total) on the same whole-body URDF. Its
reward set adds an arm-posture anchor (`joint_deviation_arm`) and an
`arm_swing_coupling` term that tracks shoulder-pitch velocity against the
same-side hip velocity. Because the network gains four action channels, the
12-action walker checkpoint has to be expanded first; the script remaps the
observation columns by joint name (the arm joints interleave with the legs in
articulation order) and zero-initialises the new action rows:

```bash
$ISAACLAB/isaaclab.sh -p tools/rl_sessions/expand_walker_checkpoint.py \
  --src <12-action-walker>.pt --dst <walker_arm_swing>.pt

$ISAACLAB/isaaclab.sh -p $RL_RUNNERS/train.py \
  --task Isaac-Locomotion-StairsStraightArmSwing-PioneerHumanoid-v0 \
  --headless --num_envs 4096 \
  --warm_start_checkpoint <walker_arm_swing>.pt \
  --run_name straight_arm_swing
```

**Foothold-conditioned stairs ("Watch Your Step")**

The `*Foothold` tasks replace the 187-ray height scan with a learned foot-landing
predictor: for each foot the policy sees the predicted next foothold (x, y in the
base frame) and a 7 x 7 patch of terrain heights around it (the policy input is
152, or 166 with `patch_rows_behind=1`; the plain tasks are 235). A checkpoint
from one family therefore cannot be loaded into the other. The predictor trains
alongside the policy and is saved as `foot_model_*.pt` next to the checkpoints;
Play configs load it from `models/<task>/`.
The course-rebuild tooling (STEP -> terrain USD) is not part of this tree; the committed
`*_terrain.usd` and profile JSONs are all that runtime needs.
The straight foothold task keeps the arms scripted; the learned arm swing exists
in the plain `StairsStraightArmSwing` / `StairsSpiralArmSwing` tasks.
A trained straight policy and its predictor are in
`models/pioneer_humanoid_stairs_foothold/` (`model_599.pt`, `foot_model_2505.pt`).

**Gait probe**

Add `--diagnostics` to `play.py` for the per-swing, lead-foot, camera and
reward instrumentation (`[PLAYARC]`, `[PLAYLEAD]`, `[PLAYDBG]`, ...). The
`PLAY_INIT_OVERRIDE`, `PLAY_SPAWN_DY`, `PLAY_MARKER`,
`PLAY_FOLLOW_ROBOT`/`PLAY_REAR_CAM` and `PLAY_MAX_STEPS` environment variables
are read in that mode:

```bash
$ISAACLAB/isaaclab.sh -p $RL_RUNNERS/play.py \
  --task=Isaac-Locomotion-StairsStraight-PioneerHumanoid-Play-v0 \
  --checkpoint <model.pt> --diagnostics --headless
```

Pipe the output through `python3 tools/onshape_course/probe_report.py <log>`
for the symmetry and clearance summary.

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
| Feet (contacts) | `l_foot`, `r_foot` |
| Hip flexion | `Hip_F_L`, `Hip_F_R` |
| Hip abduction | `Hip_A_L`, `Hip_A_R` |
| Hip rotation | `Hip_R_L`, `Hip_R_R` |
| Knee | `Knee_L`, `Knee_R` |
| Ankle pitch / roll | `Ankle_P_*`, `Ankle_R_*` |
| Arms (arm-swing task) | `l_shoulder_pitch`, `r_shoulder_pitch`, `l_elbow`, `r_elbow` |

Robot configs: `pioneer_humanoid/whole_body.py` for all v1 tasks
(`WHOLE_BODY_WALKER_CFG` for flat/rough/Onshape, `WHOLE_BODY_ARM_SWING_CFG` for
the arm-swing variant). Task configs are under `config/pioneer_humanoid_v1/`.
