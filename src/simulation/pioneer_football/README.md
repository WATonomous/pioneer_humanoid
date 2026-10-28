# Pioneer football — astronaut locomotion

Project workspace for the staged soccer plan. This first task trains **12 leg
joints on the complete 44.6931 kg astronaut**, with the upper body rigid at a
specified pose. It includes model validation, the repository's flat-locomotion
reward definitions ported to mjlab, PPO, checkpoint playback, and evaluation.
There is no soccer task or trained walking checkpoint yet.

## Laptop setup (NVIDIA, Linux or WSL2)

Use a recent NVIDIA driver, Python 3.12 and [uv](https://docs.astral.sh/uv/).
On Windows, use Ubuntu in WSL2 with NVIDIA GPU access (`nvidia-smi` must work
inside WSL). Native Windows is not a validated target. Keep the checkout in the
Linux filesystem, not `/mnt/c`, for file performance. No Docker or Isaac Sim is
required. Keep the **whole repository**: the package loads assets outside this
folder. The authoritative URDF is never regenerated or edited by setup.

From the repository root:

```bash
cd src/simulation/pioneer_football
uv sync --locked --extra cu128
uv run --locked --extra cu128 python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
uv run --locked --extra cu128 pioneer-validate
uv run --locked --extra cu128 python -m unittest discover -s tests -v
uv run --locked --extra cu128 python -m pioneer_football.preflight --device cuda:0 --num-envs 4 --steps 100
```

The lock includes PyTorch 2.10 CUDA 12.8 wheels for RTX 50-series support; do not
replace them with older CUDA wheels. Warp also has its own runtime requirements:
update the NVIDIA driver if its device compilation reports an unsupported CUDA
version. The preflight executes real Torch operations, MuJoCo-Warp stepping,
contact sensors, rewards, timeouts, and automatic resets on the selected device.
A successful `nvidia-smi` or CUDA import alone is insufficient.

**CPU/cloud checks:** substitute `--extra cpu` and `--device cpu` above. The CPU
extra is for debugging and short smoke training, not practical long training.
Do not install both extras together. Always include the selected extra in `uv run`
to preserve the intended Torch build. No W&B account or credentials are needed.

## Train

First test a short job on the laptop:

```bash
uv run --locked --extra cu128 pioneer-train Pioneer-Football-Locomotion-v0 \
  --gpu-ids 0 --env.scene.num-envs 64 --agent.max-iterations 10 \
  --agent.save-interval 5 --agent.run-name gpu_smoke
```

Then start the baseline:

```bash
uv run --locked --extra cu128 pioneer-train Pioneer-Football-Locomotion-v0 \
  --gpu-ids 0 --env.scene.num-envs 512 --agent.max-iterations 10000 \
  --agent.seed 42 --agent.run-name astronaut_baseline
```

512 environments is a **starting choice**, not a measured VRAM guarantee. For the
reported 16 GB laptop, start here; reduce to 256/128 on allocation failure, or
increase after measuring memory and steps/s. Initial Warp compilation takes time.
Train headless; avoid simultaneous training and rendering on the laptop.

Run directories are `logs/rsl_rl/pioneer_football_locomotion/<timestamp>_<name>/`.
They contain TensorBoard events, `params/env.yaml`, `params/agent.yaml`, and
`model_<iteration>.pt`. Parameters and checkpoints are ignored by Git.

```bash
uv run --locked --extra cu128 tensorboard --logdir logs/rsl_rl
```

Resume an explicitly chosen run/checkpoint (replace these example names):

```bash
uv run --locked --extra cu128 pioneer-train Pioneer-Football-Locomotion-v0 \
  --gpu-ids 0 --agent.resume True \
  --agent.load-run '2026-10-27_12-00-00_astronaut_baseline' \
  --agent.load-checkpoint 'model_999.pt' --agent.run-name astronaut_resume
```

Do not load the old legs-only checkpoint: morphology, joint mapping and
observations differ. Resume only a compatible checkpoint of this task.

## Watch and evaluate

```bash
uv run --locked --extra cu128 pioneer-play Pioneer-Football-Locomotion-v0 \
  --checkpoint-file logs/rsl_rl/pioneer_football_locomotion/RUN/model_999.pt \
  --num-envs 1 --viewer viser

uv run --locked --extra cu128 pioneer-evaluate \
  --checkpoint logs/rsl_rl/pioneer_football_locomotion/RUN/model_999.pt \
  --device cuda:0 --num-envs 16 --episodes 112 --seed 2026 \
  --output outputs/evaluation-seed2026.json
```

The evaluation uses stand, forward, left/right, turning in each direction, and a
walk-turn-stop sequence. Episodes are allocated in advance so early failures do
not cause over-sampling of easy-to-fail scenarios. It records falls/timeouts,
velocity error, yaw error, displacement, and actuator effort/speed fractions.
A provisional tracking pass requires surviving 10 seconds, mean planar velocity
error <0.10 m/s, and mean yaw-rate error <0.2 rad/s. These are diagnostics, not a
hardware certification. Noisy observations remain enabled in evaluation; reset
and foot-friction randomization remain enabled. Use several held-out evaluation
seeds and independently trained policies before claiming reliable locomotion.

A high reward or a surviving stationary robot is not sufficient: inspect speed
error and actual displacement under nonzero commands, and watch foot sliding and
crossing. Training noise itself can make a stationary policy look active.

## Model adaptation and physics

- Source: `assets/pioneer_whole_body_humanoid/pioneer_whole_body_humanoid.urdf`.
- The CAD root was near floor level and its +X axis pointed left. The adapter
  creates a pelvis-height free root with +X forward, +Y left, +Z up. No root welds
  or gravity compensation are used.
- Upper-body joints are locked, with 0.15 rad shoulder abduction to clear the
  hands from the thighs. Their mass/inertia is preserved. This approximates a
  rigid upper body; it does not validate arm motor holding capacity.
- Robot self-collision remains enabled. Only the two same-side shank-to-foot
  pairs are excluded: the shank collision box encloses the intervening two-axis
  ankle mechanism. Adjacent parent/child bodies are also excluded by MuJoCo's
  normal collision filtering. Ground and opposite-leg contacts remain active.
- Foot collision geoms have higher contact priority than the floor, so the
  randomized sole friction determines the actual foot-ground contact friction.
- All 12 leg actuators use position PD with URDF range clamps and inherited
  effort caps (222/120/60 Nm by motor group). Actual speed fractions are logged
  and excessive speed is penalized. **There is no hard joint-speed constraint,
  identified torque-speed curve, latency, or thermal model in this baseline.**
- Initial hip pitch is 0.08 rad, knee -0.22, ankle pitch -0.14; hip/ankle roll
  0.04. The imported leg-only gains did not hold the heavier astronaut. Stiffness
  is 4x and damping 2x that baseline, with effort caps unchanged. Effective
  (kp, kd): hip pitch/knee (400,16), hip roll (320,10), hip yaw (240,8), ankles
  (320,12). These are simulation assumptions, not measured hardware settings.
- Physics runs at 500 Hz and policy control at 50 Hz. Joint armature is 0.01;
  inertias are audited and compiler inertia repair is disabled. Initial support
  is checked at both 1 ms and 2 ms; training uses 2 ms.
- CAD mass/inertia are estimates; battery, electronics and wiring are omitted.
  Payload/actuator calibration remains future hardware-alignment work.

## Rewards and initial learning distribution

`mdp.py` ports the formulas from
`humanoid_rl_tasks/humanoid_rl_tasks/locomotion/mdp/rewards.py`; `task.py` starts
from the old flat task's weights. It does not import Isaac Lab.

| Term | Weight | Notes |
|---|---:|---|
| Linear / yaw velocity tracking | 1 / 1 | Exponential kernels, std 0.5; yaw-aligned linear/world yaw |
| Biped single-support time | 2 | Capped at 0.2 s; gated off for zero translation |
| Foot sliding | -0.1 | Foot velocity when ground force exceeds 1 N |
| Foot crossing | -40 | Squared deficit below 0.08 m lateral separation, left/right ordered |
| Upright / vertical velocity | -1 / -0.2 | Gravity projection and vertical speed |
| Action rate / joint acceleration | -0.005 / -1e-7 | Smoothness |
| Torque / joint-limit / hip deviation | -2e-6 / -1 / -0.1 | Joint-limit term now covers all legs |
| Overspeed | -1 | New diagnostic regularizer above source velocity limits |
| Fall | -100 | Timeout is not counted as a fall |

Initial commands: forward 0–0.6 m/s, lateral ±0.2 m/s, yaw ±0.5 rad/s, 10%
standing. Heading is not separately commanded. Resets randomize yaw, small joint
and velocity offsets and spawn clearance; foot friction is 0.6–1.0. No push or
mass randomization during this initial training. Once useful walking emerges,
expand one difficulty at a time and rerun evaluation. Fall conditions are >0.8 rad
root tilt, pelvis height <0.55 m, or non-foot ground contact; the old 0.30 m
threshold is inappropriate for this root frame.

Actor/critic each receive 48 values: base linear/angular velocity, gravity,
command, 12 joint offsets, 12 joint velocities, and 12 previous actions. Base
velocity is simulator state: this is **simulation locomotion**, not a deployable
state estimator. There are no cameras/PAM/soccer skills in this task yet.

## Cloud verification

See [VALIDATION.md](VALIDATION.md) for commands, results and remaining limits.
A short PPO smoke run only establishes that simulation, learning, logging and
checkpoint saving execute. Full convergence and RTX GPU performance must be
checked on the intended training machine.
