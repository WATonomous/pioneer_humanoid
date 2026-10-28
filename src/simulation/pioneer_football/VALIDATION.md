# Astronaut locomotion validation

Validated in the cloud on 2026-10-28, on branch `pioneer-football`, based on
repository commit `d104f22937dd2db38f6e5e64671f7f00b6d34598` plus this workspace.
This establishes a runnable training baseline and basic simulated support. It
does **not** establish a learned walking policy, hardware fidelity, or laptop
GPU performance.

## Reproducible environment

- Python 3.12; `uv.lock` installed with `uv sync --locked --extra cpu`.
- mjlab 1.6.0; MuJoCo/MuJoCo-Warp 3.11.0; Warp 1.18.0.
- PyTorch 2.10.0+cpu; rsl-rl-lib 5.4.2.
- No NVIDIA GPU available in this cloud environment.
- Source URDF SHA-256:
  `6091fa44e768abbcc17fc48fd3bbaa86f18f9392e3be193868f55758d929a25c`.

Run commands from `src/simulation/pioneer_football`. The README gives the CUDA
equivalents for Linux/WSL2 and the reported 16 GB laptop.
The cloud shell used the following writable cache/venv locations; these exports
are specific to this cloud checkout and are unnecessary on a normal laptop:

```bash
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PROJECT_ENVIRONMENT=/workspace/.venvs/pioneer-football
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cache/matplotlib
```

## Model and physics

```bash
uv run --locked --extra cpu python -m unittest discover -s tests -v
uv run --locked --extra cpu pioneer-validate
```

All **nine tests passed**: free root, mass preservation, actuator ordering,
canonical axes and joint directions, initial contacts, collision filtering,
static support, actual contact friction, and the ported reward formulas.

The source model contains 44.6931 kg. The training adapter has a six-degree free
root and 12 actuated leg joints, with the upper body fixed at the documented
pose. It has no root constraints or gravity compensation. The original URDF is
unchanged. All source inertia tensors pass positivity and triangle-inequality
checks with automatic inertia repair disabled.

All **nine 10-second physics scenarios** passed numerical checks and retained
support:

- 1 ms and 2 ms physics steps, each with nominal and two perturbed starts.
- One 30 N forward push lasting 100 ms, at the training timestep.
- Two additional perturbed starts at sole friction 0.6 and 1.0.

Across these cases, minimum pelvis height was 0.8394–0.8425 m, final horizontal
drift 2.24–4.98 mm, maximum effort fraction at most 0.147, and maximum speed
fraction at most 0.271. There were no MuJoCo numerical warnings. Independent
100 ms free-fall checks dropped the center of mass 0.04953 m and 0.05000 m at
the two timesteps, close to the analytical 0.04905 m.

The contact regression checks the solver's contact friction, not just geometry
attributes: sole values 0.6 and 1.0 remain effective against a floor with friction
1.0. Higher sole contact priority prevents the floor from overriding this
randomization. The nominal support cases use friction 0.8.

`outputs/validation.json` contains each case's measurements. A CAD-mesh stance
render was also inspected. To reproduce a headless render with EGL available:

```bash
MUJOCO_GL=egl uv run --locked --extra cpu pioneer-validate \
  --render outputs/astronaut-stance.png
```

These are support and numerical sanity checks. The gains were selected using
support diagnostics; these results are not independent evidence of locomotion
or robust push recovery.

## Environment and learning pipeline

```bash
uv run --locked --extra cpu python -m pioneer_football.preflight \
  --device cpu --num-envs 2 --steps 12

uv run --locked --extra cpu pioneer-train Pioneer-Football-Locomotion-v0 \
  --gpu-ids None --env.scene.num-envs 4 --agent.max-iterations 2 \
  --agent.num-steps-per-env 8 --agent.save-interval 1 \
  --agent.run-name cpu_smoke_final --log-root outputs/smoke
```

The final preflight passed 12 vector steps across two environments (24 environment
transitions), including four episode endings. It checked 48 actor observations, 12 actions, joint and
left/right foot sensor order, finite rewards/observations, and automatic resets.

The final CPU PPO smoke run completed two updates / 64 environment transitions
with finite losses, saved both checkpoints, TensorBoard events, and environment
and agent configurations. This is a software integration test, far too short to
learn walking. CPU throughput at this small batch is not a GPU benchmark.
Loading `model_1.pt` through the training CLI's resume path and completing one
additional PPO update also passed. Repository pre-commit checks and Ruff passed.

Checkpoint evaluation commands and interpretation are in the README. The
evaluator applies seven fixed command scenarios, preserves observation noise,
and reports physical tracking errors as well as survival. Its provisional
thresholds are mean planar velocity error below 0.10 m/s and yaw error below
0.2 rad/s, over a 10-second episode without falling. These thresholds are chosen
engineering criteria, not research-established guarantees.

The final smoke checkpoint
`outputs/smoke/pioneer_football_locomotion/2026-10-28_04-59-11_cpu_smoke_final/model_1.pt`
was evaluated on CPU with two environments, seven episodes, and seed 2026. All
seven episodes survived, but **only standing passed tracking (1/7)**. Under the
0.3 m/s forward command, mean velocity error was 0.3029 m/s and displacement was
only 0.0276 m. This confirms that survival does not establish learned walking;
the checkpoint is not a usable locomotion policy. Results are in
`outputs/smoke/evaluation-final.json`.

## Remaining training validation

1. Run the CUDA preflight and short GPU smoke job on the actual laptop. Measure
   memory usage and throughput before choosing the number of parallel robots.
2. Train the baseline; inspect videos and command-conditioned tracking metrics,
   including turning, sliding and foot crossing. A high reward or standing
   survival alone is insufficient.
3. Evaluate with held-out seeds and repeat training with at least three seeds.
   Report per-scenario results, not only pooled averages. A useful initial
   engineering target is at least 95% tracking passes in each scenario; also
   inspect behavior near the edges of the training command range and in longer
   rollouts before calling locomotion reliable.
4. If the baseline fails, first inspect failure modes and reward components;
   change one factor at a time and compare against the same evaluation suite.
   Add harder disturbances only after useful walking emerges.

Upper-body compliance/control, missing payload mass, identified actuators,
torque-speed curves, sensor estimation, delay, and physical robot transfer remain
outside this simulation baseline. There is no soccer, vision, or PAM task yet.
Generated outputs and checkpoints are ignored by Git.
