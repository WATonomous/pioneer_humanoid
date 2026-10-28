# Push-Block Task (bimanual arm)

The arm pushes a cube along a table, up a full-width ramp, and onto the flat
elevated interior floor of an open box. The gripper is **not** actuated — it is
held closed and used purely as a pushing tool, so the policy can only push, never
grasp.

- **Task package:** `humanoid_rl_tasks.push_block`
- **Gym IDs:**
  - `Isaac-Bimanual-Push-Block-v0` — training (1024 envs by default)
  - `Isaac-Bimanual-Push-Block-Play-v0` — evaluation (16 envs, no obs corruption)
  - `Isaac-Bimanual-Push-Block-Distill-v0` — vision distillation (camera + teacher/student obs)
  - `Isaac-Bimanual-Push-Block-Distill-Play-v0` — play distilled student
- **RL agent:** RSL-RL PPO (`agents/rsl_rl_ppo_cfg.py`), experiment name `push_bimanual`

---

## Scene & geometry

All positions are in the per-env frame (world minus the env origin). The arm is on
a floor stand at `ROBOT_BASE_POS = (0.15, 0, 1.1997)` and the table top is at
`TABLE_TOP_Z` (both in `workcell.py`). The box is spawned yawed −90° so its
up-the-ramp direction aligns with the env **+x** axis, so "make progress" always
means "increase the block's x". `TASK_X_OFFSET = 0.27` shifts the block, ramp, box
and target together onto the lightbox table; every x below already includes it.

| Landmark | Value | Meaning |
|---|---|---|
| Block | 50.8 mm cube (corner-origin USD), `BLOCK_HALF_SIZE = 0.0254` | the object to push |
| Block start | center ≈ `(0.48, 0.0)` on the table | in front of the ramp mouth |
| Ramp base | `RAMP_BASE_X = 0.549` | ramp meets the table |
| Ramp top | `RAMP_TOP_X = 0.578` | ramp meets the interior floor |
| Floor (visual) | `FLOOR_Z = TABLE_TOP_Z + 0.0063` | interior floor height (obs) |
| Floor (collision) | `FLOOR_Z_COLLISION = TABLE_TOP_Z + 0.0115` | effective rest height for the success check |
| Floor extent | `0.578 ≤ x ≤ FLOOR_X_MAX = 0.781`, `|y| ≤ 0.114` | interior floor footprint |
| Target | `FLOOR_TARGET = ((RAMP_TOP_X + FLOOR_X_MAX) / 2, 0) = (0.6795, 0)` | centered on the interior floor |

The block is a dynamic `RigidObject`; the box (ramp + walls) is a static
collider (`AssetBaseCfg`, no rigid-body API). All of these constants live in
`scene.py`.

---

## Getting it running

Run everything inside the `simulation_isaac` container (`./watod -t simulation_isaac`).
The launchers are `humanoid_rl/humanoid_rl/scripts/train.py` and `play.py`, exposed as
the `rl-train` and `rl-play` aliases.

### Train

```bash
rl-train --task Isaac-Bimanual-Push-Block-v0 --headless \
  --num_envs 8192 --max_iterations 500 --run_name my_run
```

- `--headless` — no GUI (recommended for training)
- `--num_envs N` — override the env count (default 1024 from `PushBlockSceneCfg`)
- `--max_iterations N` — override the 1500-iteration default
- `--run_name` — label appended to the run folder; `--seed N`, `--video` also available

Checkpoints are written to `outputs/rl/push_bimanual/<timestamp>[_<run_name>]/`
(`save_interval = 50`).

**Resume / finetune** from an existing checkpoint:

```bash
rl-train --task Isaac-Bimanual-Push-Block-v0 --headless \
  --resume True --load_run <folder_name> --checkpoint model_300.pt \
  --run_name my_finetune --max_iterations 1000
```

The step counter restarts on resume, so the DR ramp (below) restarts too.

### Evaluate (play)

```bash
rl-play --task Isaac-Bimanual-Push-Block-Play-v0 --num_envs 4                      # newest checkpoint of the newest run
rl-play --task Isaac-Bimanual-Push-Block-Play-v0 --num_envs 4 \
  --checkpoint outputs/rl/push_bimanual/<folder_name>/model_150.pt                 # a specific one
```

Play uses nominal physics (domain randomization and observation noise are off) and
exports `exported/policy.pt` (TorchScript) and `policy.onnx` before the sim loop.
**Gotcha:** `--checkpoint` is resolved literally and ignores `--load_run`; give a
real path, or omit it to load the latest from `outputs/rl/push_bimanual/`.

If the GUI aborts about 30 s after the scene renders, see the workaround in
`push_env_cfg.py` (`--kit_args=--/plugins/carb.tasking.plugin/threadCount=1`).

---

## Observations & actions

**Actions** (`ActionsCfg`): joint-position targets for the six left-arm joints,
`LEFT_ARM_JOINTS` (`joint1L` … `joint6l`), scale 0.5 rad, default offset. The
gripper has no action term and starts closed (`JOINT_POS_LIMITS` closed values).
Isaac Lab leaves un-commanded joint targets at zero (= open), so the reset event
passes `reset_joint_targets=True` to hold the closed pose.

**Observations** (`ObservationsCfg.PolicyCfg`, concatenated, corruption on in
training): joint pos/vel (relative; with noise, see Domain randomization), EE
position in the root frame, block center in the root frame, block orientation,
block linear velocity, constant ramp geometry, and last action. The EE point is
the fingertip midpoint (next section). The checkpoint's input size must match
this layout.

---

## Reward function

Weighted sum of the terms below (`RewardsCfg` in `push_env_cfg.py`, implemented in
`mdp/rewards.py`). Everything is in the env frame with **+x = up the ramp**.
`temp` is a sharpness in 1/m: the reward falls to 37% at a distance of `1 / temp`.

| Term | Weight | `temp` | What it does |
|---|---:|---:|---|
| `ee_behind_block` | **+1.0** | 15 | `exp(−temp·d)`, d = 3D distance from the fingertip midpoint to the push point (block center − 0.05 m along x). The point moves with the block |
| `block_to_target` | **+2.0** | 5 | `exp(−temp·d)`, d = xy distance from the block center to `FLOOR_TARGET` |
| `action_smoothness` | **−0.1** | 0.5 | `1 − exp(−temp·‖Δaction‖)`, bounded in [0, 1): penalizes step-to-step action changes (jitter) |

**Success metric.** The `success_rate_logger` curriculum term logs `Curriculum/success/success_rate`:
the fraction of the last 2000 episodes that ended with the block on the interior floor, at floor
height and nearly at rest (`block_on_floor` in `mdp/rewards.py`, bounds in `CurriculumCfg`). It is
a log only and has no effect on training.

**End-effector point.** The fingertip midpoint is the wrist link (`link6l`) plus the
fixed offset `LEFT_GRIPPER_TIP_OFFSET` in `pioneer_humanoid/arm_params.py`, applied
through the `ee_frame` `OffsetCfg`. `check_gripper_tip_offset` (`mdp/events.py`)
measures the fingertips from the finger meshes at startup and every 10 PPO
iterations, and prints one red line with the value to paste if it is more than 2 mm
off. The two fingers can be deflected a few mm by arm motion, which may trigger
spurious warnings.

### How the shaping fits together

1. **Get behind the block** — `ee_behind_block` pulls the fingertips to the push point.
2. **Land it on the target** — `block_to_target` pulls the block toward the floor center.
3. **Stay smooth** — `action_smoothness` penalizes jitter.
4. **Measure success** — `success_rate_logger` logs how often an episode ends on the floor.

### Terminations (`TerminationsCfg`)

- `time_out` — episode length 5.0 s (250 steps at 50 Hz).
- `object_dropping` — block falls below `BLOCK_DROP_MIN_Z = TABLE_TOP_Z − 0.10`.
- `block_off_course` — `x < 0.05`, `x > 0.8`, or `|y| > 0.20` (literals in `push_env_cfg.py`).

---

## Spawn randomization

`reset_block_center_uniform` re-places the block at every reset around the
`(0.48, 0)` anchor with a fully random yaw. Box-footprint spawns are rejected and
re-sampled.

The offset range is ramped by `spawn_ramp` (`mdp/curriculums.py`), linearly between
`SPAWN_RAMP_START_ITER = 200` and `SPAWN_RAMP_END_ITER = 300` (knobs at the top of
`push_env_cfg.py`), after the DR ramp has finished:

| | x offset (m) | y offset (m) |
|---|---|---|
| Start, `SPAWN_STAGES[0]` | (−0.06, 0.02) | (−0.06, 0.06) |
| End, `SPAWN_RAMP_TARGET = SPAWN_STAGES[1]` | (−0.10, 0.05) | (−0.12, 0.12) |

`SPAWN_STAGES[2:]` put the block beside the box, which needs a repositioning skill the
current reward has no term for, so they are not used. `spawn_offset_curriculum` is the
older performance-gated version and is not registered. Logged as `Curriculum/spawn_ramp`
(0 → 1). The play configs switch the ramp off, so play uses the narrow start range.

---

## Domain randomization

All knobs are the `DR_*` constants at the top of `push_env_cfg.py`. They are used by
the events in `EventCfg`, the observation noise in `PolicyCfg`, and the ramp in
`CurriculumCfg`.

| Knob | Default | Unit | Applies to |
|---|---|---|---|
| `DR_BLOCK_STATIC_FRICTION` / `DR_BLOCK_DYNAMIC_FRICTION` | (0.4, 1.0) / (0.3, 0.8) | coefficient | block, startup |
| `DR_FINGER_STATIC_FRICTION` / `DR_FINGER_DYNAMIC_FRICTION` | (0.5, 1.2) / (0.4, 1.0) | coefficient | fingertip links, startup |
| `DR_RESTITUTION` | (0.0, 0.05) | | block and fingertips |
| `DR_BLOCK_MASS_SCALE` | (0.8, 1.2) | × nominal | block, reset |
| `DR_ARM_LINK_MASS_SCALE` | (0.9, 1.2) | × nominal | all robot links, reset |
| `DR_ARM_STIFFNESS_SCALE` / `DR_ARM_DAMPING_SCALE` | (0.6, 1.2) / (0.8, 1.2) | × nominal | left-arm joints only (not the gripper), reset |
| `DR_JOINT_POS_NOISE` / `DR_JOINT_VEL_NOISE` | 0.01 / 0.2 | rad / rad/s | observation noise, training only |

"Arm sag" has no knob of its own: it comes from gravity load against joint
stiffness, so it is covered by the stiffness scale (below 1 = softer = more sag) and
the link mass scale.

**Ramp.** `DR_RAMP_START_ITER = 100` to `DR_RAMP_END_ITER = 200`: the `dr_ramp`
curriculum term (`mdp/curriculums.py`) linearly moves mass, actuator gains and
observation noise from nominal to the full ranges above, and logs `Curriculum/dr_ramp`
(0 → 1). `PPO_STEPS_PER_ITER` must equal `num_steps_per_env`. **Friction is not ramped:**
its materials are sampled once at startup, so it is at full range from iteration 0.
The ramp edits the observation manager's private term configs (Isaac Lab 2.3.2).

The nominal friction of the block, fingertips and table was not checked, so the
friction ranges are a first guess. There is no action delay or latency randomization.
The play configs switch these events and the ramp off.

---

## Minimal vision distillation

Distills a privileged PPO teacher into a camera student (RGB 64×64 + proprio, no
block state). Uses a small custom DAgger-style loop (`rsl_rl_scripts/distill_push.py`),
not RSL-RL's MLP-only DistillationRunner.

**Train** (from `$HUMANOID_ROOT` with Isaac Lab env):

```bash
PYTHONPATH=$(pwd) $ISAACLAB/isaaclab.sh -p $RL_RUNNERS/distill_push.py \
  --task Isaac-Bimanual-Push-Block-Distill-v0 --headless --enable_cameras \
  --teacher outputs/rl/push_bimanual/<run>/model_XXXX.pt \
  --num_envs 64 --max_iterations 2000
```

**Play:**

```bash
PYTHONPATH=$(pwd) $ISAACLAB/isaaclab.sh -p $RL_RUNNERS/play_distill_push.py \
  --task Isaac-Bimanual-Push-Block-Distill-Play-v0 --enable_cameras \
  --checkpoint outputs/rl/push_distill/<run>/nn/student_XXXX.pt --num_envs 16
```

Checkpoints land in `outputs/rl/push_distill/<timestamp>/nn/`.

### Vision domain randomization (sim2real template)

Train distill env (`*-Distill-v0`) runs reset-time vision DR via `mdp/vision_dr.py`:

| Term | What |
|------|------|
| Camera | ±3 cm position, ±3° look-at jitter on the external tiled cam |
| Light | Dome intensity 1500–4500, mild RGB tint |
| Materials | Diffuse tint / roughness / metallic on table, block, box |
| Textures | Random pick from `tasks/push/assets/textures/` (3 small PNGs) |

Play distill (`*-Distill-Play-v0`) keeps vision DR **off** for clean eval. To scale
sim2real later: widen the ranges in `DistillEventCfg` and/or drop more PNGs into
`assets/textures/` (no Git LFS required for a small pack).

### Scope record (what we have vs a fuller stack)

This distill path is intentionally a **mono-RGB + light sim2real template**:

| Area | This push distill setup | Fuller stacks (e.g. DextrAH-style) |
|------|-------------------------|-------------------------------------|
| Sensors | Single external RGB (64×64) | Mono and/or stereo RGB, optional depth |
| Vision DR | Camera jitter, dome tint/intensity, material tint/roughness, small texture pack | Large texture/HDRI packs, arm appearance DR, richer lighting |
| Image augs | None (env appearance only) | Post-render RGB/depth augs (color jitter, blur, background paste, depth noise) |
| Physics DR | Friction, block/link mass, arm gains, joint obs noise (mass/gains/noise ramped); no latency | Full ADR (friction, mass, joint gains, obs noise, etc.), often maxed during distill |
| Student train loop | Custom DAgger/BC over RSL-RL teacher | Custom distill loop (often rl-games), aux heads, multi-GPU |

**Main differences:** we stay on pure RGB with a lighter DR template; fuller systems add depth/stereo, heavier appearance DR, post-render augs, and stronger physics ADR. Same idea (privileged teacher → vision student); they scale sensors and randomization further.

**Next levers if real transfer is weak:** more textures, verify/rematch camera to the real mount, widen DR ranges, then optionally depth/stereo or image augs.

---

## PPO config (`agents/rsl_rl_ppo_cfg.py`)

`num_steps_per_env = 24`, `max_iterations = 1500`, MLP actor/critic
`[256, 128, 64]` (ELU), `learning_rate = 3.333e-5` (adaptive schedule),
`gamma = 0.98`, `lam = 0.95`, `entropy_coef = 0.002`, `clip = 0.2`,
`desired_kl = 0.01`.
