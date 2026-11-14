# Push-Block Task (bimanual arm)

The left arm of `pioneer_bimanual_arm_8dof` (`BIMANUAL_ARM_8DOF_CFG`) pushes a cube along a table, up a full-width ramp, and onto the flat
elevated interior floor of an open box. The gripper is **not** actuated — it is
held closed and used purely as a pushing tool, so the policy can only push, never
grasp.

- **Task package:** `humanoid_rl_tasks.push_block`
- **Gym IDs:**
  - `Isaac-Bimanual-Push-Block-v0` — training (512 envs by default)
  - `Isaac-Bimanual-Push-Block-Play-v0` — evaluation (16 envs, no obs corruption)
  - `Isaac-Bimanual-Push-Block-Distill-v0` — vision distillation (camera + teacher/student obs)
  - `Isaac-Bimanual-Push-Block-Distill-Play-v0` — play distilled student
- **RL agent:** RSL-RL PPO (`agents/rsl_rl_ppo_cfg.py`), experiment name `push_bimanual`

---

## Scene & geometry

All positions are in the per-env frame with the robot base at the origin. The box
is spawned yawed −90° so its up-the-ramp direction aligns with the env **+x** axis
— i.e. "make progress" always means "increase the block's x".
The robot stands 0.279 m to the right (`scene.ROBOT_BASE_POS`), so the task lies in front of
its left shoulder: the left arm reaches only ~0.05 m past the robot's midline.

| Landmark | Value | Meaning |
|---|---|---|
| Block | 50.8 mm cube (corner-origin USD), `BLOCK_HALF_SIZE = 0.0254` | the object to push |
| Block start | center ≈ `(0.21, 0.0)` on the table | in front of the ramp mouth |
| Ramp base | `RAMP_BASE_X = 0.279` | ramp meets the table (z = 0) |
| Ramp top | `RAMP_TOP_X = 0.308` | ramp meets the interior floor |
| Floor (visual) | `FLOOR_Z = 0.0063` | interior floor height (obs / scoop) |
| Floor (collision) | `FLOOR_Z_COLLISION = 0.0115` | effective rest height for success check |
| Floor extent | `x_max = 0.511`, `|y| ≤ 0.114` | interior floor footprint |
| Target | `FLOOR_TARGET = (0.37, 0.0)` | goal point on the interior floor |

The block is a dynamic `RigidObject`; the box (ramp + walls) is a static
collider (`AssetBaseCfg`, no rigid-body API).

---

## Getting it running

Prerequisites: an Isaac Lab 2.3 / Isaac Sim 5.1 environment with this task
package importable, plus the host project's RSL-RL `train` / `play` /
`list_envs` entry points (the standard Isaac Lab external-project workflow). The
commands below assume those entry points are available and are run from the host
project root.

This folder provides only the task itself — the environment configs, MDP terms
(rewards, observations, events, terminations, curriculum), and the PPO agent
config. It does not ship the training/eval launcher scripts or the robot/box/block
assets; those come from the host project.

Confirm the task is registered:

```bash
uv run list_envs | grep Push
```

Sanity-check the environment with dummy agents (opens the GUI):

```bash
uv run zero_agent   --task Isaac-Bimanual-Push-Block-Play-v0   # zero actions
uv run random_agent --task Isaac-Bimanual-Push-Block-Play-v0   # random actions
```

### Train

```bash
uv run train --task Isaac-Bimanual-Push-Block-v0 --headless
```

Useful flags (provided by the host project's RSL-RL training entry point):

- `--headless` — no GUI (recommended for training)
- `--num_envs N` — override env count (default 512; the aggressive curriculum runs used 4096)
- `--max_iterations N` — override the 1500-iteration default
- `--seed N`, `--video` — reproducibility / record rollout videos

Checkpoints are written to `outputs/rl/push_block/<timestamp>[_<run_name>]/`
(`save_interval = 50`).

**Resume / finetune** from an existing checkpoint:

```bash
uv run train --task Isaac-Bimanual-Push-Block-v0 --headless \
  --resume --load_run <folder_name> --checkpoint model_1998.pt \
  --run_name my_finetune --max_iterations 3000
```

### Evaluate (play)

```bash
uv run play --task Isaac-Bimanual-Push-Block-Play-v0 --load_run <folder_name>
```

Play exports `exported/policy.pt` (TorchScript) and `policy.onnx` before the
sim loop. **Gotcha:** `--checkpoint <name>` is resolved literally against the CWD
and ignores `--load_run` — pass a full absolute path to `--checkpoint`, or omit it
and rely on `--load_run` alone.

---

## Observations & actions

**Actions** (`ActionsCfg`): joint-position targets for the seven left-arm joints
(`arm_8dof_params.LEFT_ARM_JOINTS`, scale 0.5, default offset). `left_gripper` has no
action term: it starts closed and its PD actuator holds it there. 7 action dims.

**Observations** (`ObservationsCfg.PolicyCfg`, concatenated, corruption on in
training): joint pos/vel (relative, all 18 joints), the left TCP position in the root
frame (`TCP_LOCAL` on `left_link7`), block center in root frame, block orientation,
block linear velocity, constant ramp geometry (base x, top x, floor z, target xy,
push-dir xy), and last action. 63 dims total. Checkpoints trained on the old 6-DoF
arm (58 obs, 6 actions, e.g. `models/push/model_1499.pt`) don't load.

---

## Reward function

Weighted sum of the terms below (`RewardsCfg` in `push_env_cfg.py`, implemented in
`mdp/rewards.py`). Everything is expressed in the env frame with **+x = up the
ramp**.

| Term | Weight | What it does |
|---|---:|---|
| `ee_behind_block` | **+2.0** | tanh reach reward for keeping the EE just *behind* the block (`center − 0.05·x̂`), the side you push from |
| `push_progress` | **+5.0** | dense signed block x-velocity, clamped to ±0.5 — rewards advancing up the ramp, penalizes sliding back |
| `backslide` | **−8.0** | extra asymmetric penalty on the *backward* (−x) block speed only |
| `block_to_target` | **+8.0** | tanh distance from block center to the floor target `(0.37, 0)`, wide kernel `std = 0.15` |
| `block_to_target_fine` | **+5.0** | same, sharp kernel `std = 0.05` — precise final placement |
| `block_on_floor` | **+30.0** | **success bonus**: block center inside the floor footprint, at floor rest height, and nearly at rest (`speed < 0.05`) |
| `scoop_penalty` | **−40.0** | location-aware anti-cheat: penalizes block height above the *local* support surface, but only when the block is *outside* the ramp+box footprint (≈0 on the ramp/floor where height gain is required) — stops the arm from lifting/scooping the block in open table space |
| `lateral_deviation` | **−10.0** | penalizes sideways drift `|y|` beyond a 0.06 m corridor |
| `ee_reposition` | **0.0 → 1.5** | shaping to route the EE behind an *off-corridor* block toward the ramp mouth; off by default, switched on by the curriculum at stage ≥ 2 |
| `action_rate` | −1e-4 → −1e-2 | action-smoothness penalty (raised by curriculum at 10 k steps) |
| `joint_vel` | −1e-4 → −1e-2 | joint-velocity penalty (raised by curriculum at 10 k steps) |

### How the shaping fits together

1. **Get behind the block** — `ee_behind_block` pulls the EE to the push point.
2. **Drive it up the ramp** — `push_progress` rewards +x motion; `backslide`
   punishes losing ground; `lateral_deviation` keeps it in the corridor.
3. **Land it on the target** — `block_to_target` (coarse) + `_fine` (precise)
   guide the block to `(0.37, 0)`.
4. **Settle it** — `block_on_floor` pays a large one-shot bonus once the block is
   inside the footprint, at floor height, and at rest.
5. **Don't cheat** — `scoop_penalty` makes lifting the block in open space
   costly, while staying ≈0 on the ramp/floor where climbing is legitimate.
6. **Reposition when needed** — for hard spawns beside/behind the box,
   `ee_reposition` (enabled by the curriculum) routes the EE to push the block
   back toward the ramp mouth rather than straight +x.

### Terminations (`TerminationsCfg`)

- `time_out` — episode length 5.0 s.
- `object_dropping` — block falls below −0.05 m (off the table).
- `block_off_course` — block leaves the corridor: `x < 0.05`, `x > 0.55`, or
  `|y| > 0.20`.

---

## Spawn curriculum

`CurriculumCfg.spawn` (`mdp/curriculums.py`, `spawn_offset_curriculum`)
performance-gates the block's spawn randomization. It widens the block's spawn
*offset* range one stage at a time around the fixed `(0.21, 0)` anchor:

| Stage | x offset | y offset | notes |
|---|---|---|---|
| 0 | (−0.06, 0.02) | (−0.06, 0.06) | moderate, ramp-approach side |
| 1 | (−0.10, 0.05) | (−0.12, 0.12) | wider, out to the ramp mouth edges |
| 2 | (−0.12, 0.20) | (−0.20, 0.20) | **reaches beside the box** → `ee_reposition` turns on |
| 3 | (−0.14, 0.30) | (−0.26, 0.26) | full reachable table incl. beside/behind box |

Yaw is always fully randomized. A stage advances only once **both** hold: the
policy has trained at that stage for ≥ `min_stage_steps` (1200 env steps), **and**
the rolling `block_on_floor` success rate over the last 100 episodes clears
`threshold = 0.70`. The dwell gate prevents a competent (resumed) policy from
cascading through all stages within one PPO iteration. Box-footprint spawns are
rejected and re-sampled so a widened range never drops the block inside the walls.

Logged under `Curriculum/spawn/{stage, success_rate, x_off_hi, reposition_active}`.

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
| Physics DR | Task spawn curriculum + existing push MDP | Full ADR (friction, mass, joint gains, obs noise, etc.), often maxed during distill |
| Student train loop | Custom DAgger/BC over RSL-RL teacher | Custom distill loop (often rl-games), aux heads, multi-GPU |

**Main differences:** we stay on pure RGB with a lighter DR template; fuller systems add depth/stereo, heavier appearance DR, post-render augs, and stronger physics ADR. Same idea (privileged teacher → vision student); they scale sensors and randomization further.

**Next levers if real transfer is weak:** more textures, verify/rematch camera to the real mount, widen DR ranges, then optionally depth/stereo or image augs.

---

## PPO config (`agents/rsl_rl_ppo_cfg.py`)

`num_steps_per_env = 24`, `max_iterations = 1500`, MLP actor/critic
`[256, 128, 64]` (ELU), `learning_rate = 3.333e-5` (adaptive schedule),
`gamma = 0.98`, `lam = 0.95`, `entropy_coef = 0.006`, `clip = 0.2`,
`desired_kl = 0.01`.
