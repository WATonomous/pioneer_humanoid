# Pioneer soccer demo and experiment baseline

Approach → dribble → shoot on a modeled pitch. Five frozen RL experts control
the joints; a scripted coordinator selects skills. The optional second robot
uses possession/patrol logic, not active defending or learned match tactics.

## Setup and playback

Use Linux or WSL2, Python 3.12, a recent NVIDIA driver and
[uv](https://docs.astral.sh/uv/). Keep the full repository: the robot loads the
URDF and meshes from `assets/pioneer_whole_body_humanoid/`.

```bash
cd src/simulation/pioneer_football
uv sync --locked --extra cu128
```

Obtain `pioneer-soccer-demo-weights.tar.gz` from the demo maintainer and extract
it here. It installs `demo/{gait,dribble,strike,patrol,turn}.pt`. These binaries
are separate from the PR; the archive is available locally under `outputs/`.

```bash
tar -xzf /path/to/pioneer-soccer-demo-weights.tar.gz
.venv/bin/python -m pioneer_football.demo \
  --mode play --autoplay --bundle demo/manifest.json
```

Open the printed URL, normally `http://localhost:8087`. Add `--port 8088` if
needed, or `--opponent` for the two-robot prototype. The manifest verifies the
exact weights, controller and runtime source before playback. Existing local
checkpoint files do not need to be extracted again.

For a headless run:

```bash
.venv/bin/python -m pioneer_football.demo \
  --bundle demo/manifest.json --num-envs 6 --seconds 60 --seed 27001 \
  --output outputs/demo/sequence.json
```

For a full match, use `--opponent --seconds 180` instead of
`--num-envs 6 --seconds 60`. Add `--video outputs/demo/replay.mp4` with
`MUJOCO_GL=egl` to record a headless run. Playback does not train a policy.

## Continuing experiments

- `demo.py`: skill coordination, shared policy history, viewer and headless runner.
- `runtime.py`: direct environment configuration, motor/action model,
  observations, task rewards and policy loading.
- `gait.py`: the accepted walking reward recipe, exposed by `make_rewards()`.
- `model.py`, `ball.py`, `pitch_scene.py`: robot assets and physical scene.
- Navigation/history helpers: routes, turning commands and coordinate transforms.
- `match_rules.py`, `match_audit.py`: scoring, restarts, falls and field boundaries.

These are the current baseline definitions, with the historical experiment
inheritance removed. A training runner is not included in this trimmed package.
Use separate checkpoints and configurations for future training. Source changes
invalidate the frozen bundle's hashes; validate an experiment before creating a
new bundle rather than silently updating the baseline manifest.

## Validation

The current regression suite is consolidated into one file:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

All **80 tests pass**. The runtime refactor matched the previous configuration,
gait rewards and all 482 compiled robot arrays. Two CPU worlds × thirty action
steps had identical observations, rewards, positions, velocities and torques;
all five policies had identical outputs on 64 probe states each. The refactored
GPU demo passed **6/6 solo sequences and 2/2 mirrored goal/restart checks**, with
zero falls or hidden resets. Full 180-second matches were not rerun after the
refactor. A compact summary is retained in `demo/manifest.json`.

Earlier GPU replays varied from 3/6 to 6/6 goals even with the same seed. These
checks do not establish perfect repeatability, natural-looking gait or exhaustive
robustness. Source motor limits and ball physics are preserved; gains/contact
settings remain simulation assumptions, not hardware calibration.

Raw reports, original diagnostics and experiment sources are archived locally
under the ignored `outputs/` directory; they are not required for playback.
The dependency lock pins mjlab 1.6.0 and PyTorch 2.10/CUDA 12.8. Use `--extra cpu`
for CPU checks instead of `--extra cu128`; do not install both extras together.

The simulation design adapts [Joga](https://joga.csail.mit.edu/index.html).
The walking reward recipe originates from Wilson's repository implementation
(`f68a5904`). Robot geometry, limits and trained policies are Pioneer-specific.
