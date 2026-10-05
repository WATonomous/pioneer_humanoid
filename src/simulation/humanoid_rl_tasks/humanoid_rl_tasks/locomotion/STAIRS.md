# Pioneer humanoid stairs locomotion

This is a team-facing training scaffold, not a trained stairs policy or evidence
of stair-climbing performance. It uses the repository's `simulation_isaac` Docker
stack (Isaac Lab 2.3.2 / Isaac Sim 5.1). Configuration contract tests can run while
another job trains; simulator checks, training, and playback must wait until the
GPU is free.

## Tasks and implementation

| Task | Purpose |
| --- | --- |
| `Isaac-Velocity-Stairs-PioneerHumanoid-v0` | Fresh PPO training with a stairs curriculum |
| `Isaac-Velocity-Stairs-PioneerHumanoid-Play-v0` | Fixed-height, forward-walking inspection |

The equivalent `Isaac-Locomotion-Stairs-PioneerHumanoid-v0` and
`Isaac-Locomotion-Stairs-PioneerHumanoid-Play-v0` aliases are also registered.

- Environment: [config/pioneer_humanoid_v1/stairs_env_cfg.py](config/pioneer_humanoid_v1/stairs_env_cfg.py).
- Course mesh: [terrains/stair_course.py](terrains/stair_course.py).
- Completion, corridor, and curriculum terms: [mdp/stair_course.py](mdp/stair_course.py).
- PPO runner: [config/pioneer_humanoid_v1/agents/stairs_ppo_cfg.py](config/pioneer_humanoid_v1/agents/stairs_ppo_cfg.py).
- Baseline: `PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg`. The robot,
  corrected knee geometry, selective self-collision, ground-contact sensors,
  knee-shaping rewards, observation layout, action ordering, actuators, and physics
  come from this repository baseline. No new stairs-specific reward is added.
- The stairs task does **not** import the local swing-timing experiment or require
  any helper under `outputs/`. Existing rough-terrain task defaults are unchanged.

When preparing a PR, include the baseline's robot/asset, selective-collision,
contact-sensor, and knee-shaping dependencies as well as these stairs files. A
copy of `stairs_env_cfg.py` alone is not a standalone environment. This
contribution is source-only: keep trained checkpoints, policy exports, training
outputs, local Docker configuration, native-viewer helpers, and private
experiment results out of the PR.

## Starting terrain and commands

The training generator has 10 difficulty rows and 10 terrain-family columns, with
16 m long by 6 m wide cells and a 20 m outer border. The ordered families are 80%
complete stair courses and 20% flat calibration. A stair cell contains the whole
route, not separate ascending and descending environments:

```text
                raised walkway: 2 m
                 ______________
              __|              |__
           __|                    |__
spawn ____|                          |____ finish
      12 steps up                 12 steps down
                        travel +X →
```

The flights span the full cell width, preventing a shortcut around their sides.
There is a 2 m flat approach, twelve 35 cm stair treads up (a 4.2 m flight), an
exactly 2 m raised walkway, twelve matching treads down to the original ground
height, and a flat exit. The walkway occupies local x = 6.2–8.2 m. The stair
footprint ends at local x = 12.4 m; the completion line is at x = 13.4 m, one
metre beyond the stair footprint. The final descending drop reaches ground at
x = 12.05 m. The spawn is on the approach at local x = 0.9 m.
Local coordinates are measured from the cell's lower-X edge, not the world origin.
There are no holes, pyramid rings, or separate ascent/descent spawn platforms.

Risers increase from 3 cm to 15 cm, so the raised walkway is 36–180 cm above the
ground. Training starts at difficulty row zero. The custom curriculum promotes an
environment only after the robot reaches the flat exit upright and within the
course corridor; unsuccessful episodes can demote it. Merely covering radial
distance, walking partway up, or reaching the raised walkway is not a completed
course. Moving more than 2 m laterally from the course center terminates the
episode. Successful completion is recorded as a time-limit-style reset so PPO
does not treat reaching the exit as a physical failure.

The generator seed is fixed at 42 for a repeatable map; the training `--seed`
controls the learning/reset random streams.

Training commands request 0.25–0.6 m/s forward, zero lateral velocity, and heading
zero, with up to 0.5 rad/s of corrective yaw. Reset position varies by ±0.2 m in X
and Y and reset yaw by ±0.15 rad, all on the approach. Episodes last 75 seconds;
the nominal 12.5 m spawn-to-finish distance takes 50 seconds at the lowest
commanded speed, leaving time for stair-climbing and recovery. Playback requests 0.4 m/s
forward, uses zero reset yaw, disables observation corruption, and fixes risers
at 8 cm: the walkway is 96 cm high. Its 5-row, 10-column map retains stair courses
and flat calibration but does not advance difficulty. With one environment, the
first column is a complete stair course; use 10 or 50 environments to inspect
multiple instances and the flat calibration cells together.

The 15 cm ceiling is a starting research setting, not an established capability.
Reduce the height range if the robot cannot reliably clear the easier rows. Change
one terrain or reward setting at a time, and record the recipe and seed.

## Validation and first run

Run the CPU-only tests from the repository root. They do not
launch Isaac Sim or use the GPU. The standard-library tests exercise configuration statements
and the course-generating function with lightweight test doubles, including
recorded box dimensions and surface profiles. They are not real Isaac API, mesh
integration, or physics tests:

```bash
python3 -m unittest discover \
  -s src/simulation/humanoid_rl_tasks/tests \
  -p 'test_stair*.py' -v
```

Hosts without Torch run 29 source/geometry tests and skip five episode-logic
tests. To run all 34 on CPU, use the provided image's Python inside the container
(this does not start Isaac, so it is safe while a training job runs):

```bash
cd "$HUMANOID_ROOT"
CUDA_VISIBLE_DEVICES="" "$PYTHON" -m unittest discover \
  -s src/simulation/humanoid_rl_tasks/tests -p 'test_stair*.py' -v
```

The Isaac validator starts `AppLauncher` even without `--runtime`, so
both modes below must wait until the GPU is free. Its default mode checks the real
loaded configuration, registration, generated geometry, and Hydra serialization;
`--runtime` also creates two environments and runs 20 zero-action steps, checking
finite observations, 235 policy observations, 12 leg actions, and contact sensors.
Zero-action smoke steps check wiring, not locomotion success.

Enter the existing `simulation_isaac` development container from the host:

```bash
./watod -t simulation_isaac_dev
```

The image sets `HUMANOID_ROOT`, `ISAACLAB`, and `RL_RUNNERS`; the commands below run
inside that container. If another job uses the same GPU, leave it running and do
not execute simulator commands yet. Do not recreate or stop its container.

Run the config/geometry check, then the runtime smoke:

```bash
cd "$HUMANOID_ROOT"
"$ISAACLAB/isaaclab.sh" -p \
  src/simulation/humanoid_rl_tasks/scripts/check_stairs.py --headless
"$ISAACLAB/isaaclab.sh" -p \
  src/simulation/humanoid_rl_tasks/scripts/check_stairs.py --headless --runtime
```

Once the GPU is free, start with a small **fresh** 200-update wiring/learning smoke:

```bash
cd "$HUMANOID_ROOT"
"$ISAACLAB/isaaclab.sh" -p "$RL_RUNNERS/train.py" \
  --task Isaac-Velocity-Stairs-PioneerHumanoid-v0 \
  --headless --num_envs 256 --max_iterations 200 --seed 43 \
  --run_name stairs_smoke_seed43 --fail_on_nonfinite_grad
```

Runs, saved `params/env.yaml` / `params/agent.yaml`, checkpoints, and TensorBoard
events are under `outputs/rl/pioneer_humanoid_stairs/<timestamp>_<run_name>/`.
The runner saves every 25 updates. Use the checkpoint path actually saved by the
run; a 200-update fresh run normally ends at `model_199.pt` because indexing starts
at zero.

Inspect the smoke checkpoint before increasing the budget. Set `STAIRS_CHECKPOINT`
to that run's exact checkpoint path, then use a working GUI/display:

```bash
"$ISAACLAB/isaaclab.sh" -p "$RL_RUNNERS/play.py" \
  --task Isaac-Velocity-Stairs-PioneerHumanoid-Play-v0 \
  --checkpoint "$STAIRS_CHECKPOINT" --num_envs 10 --real-time
```

Alternatively, record a finite headless clip (rendering still uses the GPU):

```bash
"$ISAACLAB/isaaclab.sh" -p "$RL_RUNNERS/play.py" \
  --task Isaac-Velocity-Stairs-PioneerHumanoid-Play-v0 \
  --checkpoint "$STAIRS_CHECKPOINT" --num_envs 10 \
  --headless --video --video_length 3000 --enable_cameras
```

The 3,000-step clip covers 60 simulated seconds. The nominal route takes 31.25
seconds at the fixed playback command; the longer clip allows time to inspect
both twelve-step flights and the raised walkway, including pauses and recovery.
Clips are written beside the checkpoint under `videos/play/`; playback
also exports the selected policy under that run's `exported/` directory. The
playback task uses fixed 8 cm steps, so a very early policy trained on the shallow
initial row may fail this harder inspection. Distinguish that failure from a
wiring error.

If startup, learning signals, and visual inspection look promising, run a fresh
1,000-update pilot before planning a long job:

```bash
"$ISAACLAB/isaaclab.sh" -p "$RL_RUNNERS/train.py" \
  --task Isaac-Velocity-Stairs-PioneerHumanoid-v0 \
  --headless --num_envs 1024 --max_iterations 1000 --seed 43 \
  --run_name stairs_pilot_seed43 --fail_on_nonfinite_grad
```

Tune `--num_envs` to measured GPU memory and updates/second. More parallel
environments change the samples per PPO update; they are not a pure speed knob.
The pilot is fresh by default, not a continuation of the smoke policy. Do not use
`--resume` to silently switch a rough checkpoint to a different terrain recipe;
any later warm-start experiment should be named separately and document restored
policy/normalizer state versus optimizer and iteration state.

## Teammate handoff checklist

- Check the startup log and saved recipe: correct task, seed, course dimensions,
  two terrain families, initial row zero, normalized policy,
  and separate stairs output folder.
- Review reward, episode length, terrain level, forward tracking, target knee
  rewards, value/policy losses, exploration noise, and numerical-failure diagnostics.
  Total reward alone does not establish stair success.
- Inspect the entire approach → ascent → raised walkway → descent → flat exit
  sequence, left/right stepping, foot clearance, self-contact, and recovery after
  a bad foothold. A policy walking on the approach or raised walkway has not yet
  demonstrated a completed course.
- Record falls, completed courses, command tracking, and test seeds on fixed
  stair heights; compare easier and harder stairs rather than only random
  curriculum playback. Repeat promising results on another seed.
- Treat the inherited rewards and terrain range as the starting hypothesis.
  Keep tuning isolated to the stairs task until a change is supported by evidence.

No stairs policy has been trained or evaluated for climbing success as part of
this scaffold preparation. Wiring smoke tests are not locomotion validation.
GUI inspection and the initial fresh pilot remain the next acceptance gates.

Preparation checks (2026-10-04): all 33 CPU tests passed in the provided container,
including upright full-course completion, lane bounds, initial-reset neutrality,
and rejection of simultaneous falls or an inverted robot at the finish line.
These tests cover the updated twelve-step course, superseding the earlier
six-step scaffold.

A separate CPU-only check executed the course generator with real NumPy/trimesh,
without importing Isaac or initializing CUDA. It verified twelve rises and twelve
descents, the 2 m walkway, 36/108/180 cm peak heights at curriculum difficulties
0/0.5/1, the 96 cm playback peak, and the finish line at local x = 13.4 m. Flat and
3/8/15 cm-riser profiles have continuous ground across the 16 m by 6 m cell; the
stair course contains 25 watertight positive-volume meshes. Source-contract and
mesh checks do not establish locomotion performance.

PR preparation checks (2026-10-05): all 63 tests passed on CPU in the provided
image (33 stairs contracts plus 26 knee-shaping and four numerical-guard tests).
The real Isaac validator also passed registration, baseline/config preservation,
Hydra round-trip, actual mesh profiles, actor/critic normalization, and exactly
19 nonzero walking rewards. The runtime smoke created two environments and ran
20 zero-action steps with finite 235-value observations, 12 actions, rewards, and
terrain-contact sensor data, then exited successfully. On this local WSL setup,
graphics initialization and CUDA shutdown warnings were logged despite the
completed headless physics assertions. This does not validate Docker GUI
rendering, stair traversal, or long-running training stability.

The source-only scope cleanup on the same date passed all 64 CPU tests (34 stairs,
26 knee-shaping, and four numerical-guard tests), including the additional
contract limiting public tasks to Flat, Rough, the final selective-knee recipe,
and Stairs. Reward weights, robot physics, and the final PPO recipe were unchanged.
