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

## Generated boxing movement

`scripts/generate_moves.py` builds boxing movement without mocap, chained
freely from a move list:

| move | what |
|---|---|
| `F` `B` `L` `R` | step-drag: the foot nearest the direction moves first, the other pushes and follows the same distance |
| `D` / `DD` | duck, shallow and fast (10 cm, 10° lean) / deep (20 cm, 25° lean): knees bend, back straight, feet planted |
| `SL` `SR` | slip: weight onto that side's foot, dip and tilt the head off line |
| `WL` `WR` | bob and weave, head ending left / right: U-shaped path under a hook |
| `PL` `PR` | pivot on the lead foot: rear foot swings round in short steps, body turns 45°, lead foot turns to match |
| `H` | hold the stance |

The stance and guard come from the boxing take; foot paths and the
centre-of-mass / hip height / heading / lean curves come from the rules
above, and the legs are solved by IK. Moves blend into each other without
stopping. Any move that drives a leg motor past 80% of its speed cap
(`--max-cap-use`) is slowed down automatically, so what comes out is within
what the motors can do with headroom for balance.

```bash
uv run scripts/generate_moves.py --sequence F F D PL SL SR WL WR DD PR B B \
    --output-file data/motions/moves.csv
uv run scripts/csv_to_npz.py --input-file data/motions/moves.csv --input-fps 50 \
    --output-file data/motions/moves.npz --video True
```

Each move has its own settings (`--duck.depth`, `--pivot.angle`,
`--step.forward`, …; see `--help`). Orthodox only.

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
scripts/         fetch_assets.sh, csv_to_npz.py, generate_moves.py, train.py, play.py
data/            fetched model + motions, generated NPZs (gitignored)
```
