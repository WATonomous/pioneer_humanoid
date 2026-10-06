# Whole-body motion tracking on mjlab

mjlab (MuJoCo Warp) port of the Wato motion tracking pipeline in
[WATonomous/Humanoid_motion_tracking](https://github.com/WATonomous/Humanoid_motion_tracking)
(GMR retargeting + BeyondMimic training in Isaac Lab). The training side runs on
mjlab's own BeyondMimic port (`mjlab.tasks.tracking`) with the Wato robot and
task settings copied from the Isaac config. Retargeting (BVH → CSV) still
happens in that repo with GMR.

```
human BVH ──GMR (other repo)──▶ CSV ──csv_to_npz.py──▶ NPZ ──train.py──▶ policy ──play.py──▶ video / viewer
```

## Setup

```bash
uv sync --extra train            # Linux + NVIDIA GPU (CPU: --extra train-cpu, smoke tests only)
./scripts/fetch_assets.sh        # Wato MJCF + meshes and the retargeted CSVs → data/
```

`fetch_assets.sh` pulls a pinned commit of the motion tracking repo: the CSVs
are joint angles for *its* robot model (GMR's `watonomous.xml`, 28 joints),
which is not the same model as `assets/whole_body_humanoid/` here.

## Use

```bash
# 1. CSV → NPZ (50 fps), plus an MP4 of the reference motion
uv run scripts/csv_to_npz.py --input-file data/motions/boxing.csv --input-fps 120 \
    --output-file data/motions/boxing.npz --video True

# 2. train (GPU)
uv run scripts/train.py Mjlab-Tracking-Flat-Wato \
    --env.commands.motion.motion-file data/motions/boxing.npz \
    --env.scene.num-envs 4096 --agent.run-name boxing

# 3. watch it (viser web viewer on a headless box, or record an MP4)
uv run scripts/play.py Mjlab-Tracking-Flat-Wato --motion-file data/motions/boxing.npz \
    --checkpoint-file logs/rsl_rl/wato_tracking/<run>/model_<n>.pt --viewer viser
uv run scripts/play.py Mjlab-Tracking-Flat-Wato --motion-file data/motions/boxing.npz \
    --checkpoint-file <model.pt> --video True --video-length 1770
```

Runs log to Weights & Biases by default (`--agent.logger tensorboard` to stay
local); checkpoints land in `logs/rsl_rl/wato_tracking/<timestamp>/`. Training
also accepts `--registry-name <wandb motion registry entry>` instead of a local
NPZ, like the Isaac version.

Motions in `data/motions/`: `boxing.csv` (Xsens, 120 fps, 35 s), `curling.csv`,
`squatting.csv`. To play a motion faster, give `csv_to_npz.py` a larger
`--input-fps` than it was recorded at (240 for a 120 fps take = 2× speed).

## Generated footwork

`scripts/generate_footwork.py` builds boxing step-drag footwork (forward, back,
left, right) without mocap: stance and guard copied from the boxing take,
feet and centre of mass placed by the step-drag rules (foot nearest the
direction moves first, the other pushes and follows the same distance, weight
onto the support foot before each lift, reset to stance after every step),
legs solved by IK. It prints a check against the motor speed caps.

```bash
uv run scripts/generate_footwork.py --sequence F F F B B B L L L R R R \
    --output-file data/motions/footwork.csv
uv run scripts/csv_to_npz.py --input-file data/motions/footwork.csv --input-fps 50 \
    --output-file data/motions/footwork.npz --video True
```

Orthodox only. Longer steps, a longer stance or a bigger weight shift push the
knees past their 3.67 rad/s cap; the script's report shows when.

Headless rendering without a GPU: `apt install libosmesa6` and set
`MUJOCO_GL=osmesa`.

## Differences from the Isaac Lab version

| | Isaac Lab | here |
|---|---|---|
| robot | `whole_body_humanoid_raw_export.urdf` | GMR's `watonomous.xml` (same robot, converted from that URDF) |
| collisions | convex hulls of every mesh, self-collision off | box under each foot only; meshes are visual |
| motor speed caps | `velocity_limit_sim` | DC motor model with full torque up to the cap, braking above it |
| `undesired_contacts` reward | yes | no (mjlab's base tracking task has none) |
| PPO, rewards, observations, terminations, gains, armature, action scale | — | same values |

## Layout

```
wato_tracking/   robot.py (entity + motors), env_cfg.py (task), rl_cfg.py (PPO); __init__ registers the tasks
scripts/         fetch_assets.sh, csv_to_npz.py, generate_footwork.py, train.py, play.py
data/            fetched model + motions, generated NPZs (gitignored)
```
