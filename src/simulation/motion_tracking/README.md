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

## Boxing arena (two fighters)

`wato_boxing/` sets up red vs blue in a ring for fighting RL: scene, rewards
and round endings. Skill actions and opponent observations are not in yet.

- **Fighters**: the tracking robot recoloured, in the orthodox stance from
  `generate_moves.py`, 1.3 m apart facing each other (`distance`): lead
  gloves ~0.2 m apart, a jab lands only after ~0.3 m of stepping in.
- **Gloves and hitboxes**: glove spheres on the wrists; head, torso, pelvis
  and arm/leg capsules fitted to the CAD meshes (hidden, group 3). They hit
  the opponent, the ropes and the floor, never their own body. The CAD
  meshes stay visual only.
- **Ring**: 4 m between the ropes (scaled to Wato), 3 ropes, 4 posts.
- **Hit sensors**: `<attacker>_hits_<defender>_head` / `_body`, glove vs
  opponent head / torso, with contact count and net force per glove.

```bash
MUJOCO_GL=osmesa uv run scripts/view_arena.py --out-dir videos/arena --seconds 3   # checks + stills + video
uv run scripts/play.py Mjlab-Boxing-Arena-Wato --agent zero --viewer viser         # interactive
```

**Rewards and round endings** (`wato_boxing/mdp.py`, weights in
`REWARD_WEIGHTS` in `arena_env_cfg.py`, points not scaled by dt, scored from
the `learner`'s side). The robot cannot get up, so the first fighter down
(non-foot hitbox on the canvas, pelvis < 0.4 m or torso tilt > 60°) ends the
round:

| term | points |
|---|---|
| knockdown: opponent down within 1 s of a glove hit | +100 |
| opponent down without a hit (slip) | +10 |
| knocked down / fell | −100 |
| both down | −50 each |
| self-collision (legs through each other, arm through torso), round ends | −100 |
| bell (20 s), nobody down | ±10 × tanh(landed-hit difference / 3) |
| clean hits, per step of glove contact × force/100 N (head ×1, body ×0.5) | +2 |
| hits taken | −1 |
| stability: no extra lean beyond the stance, pelvis at stance height | +0.02/step |
| shoving: head/torso/pelvis/legs pressing on the opponent's torso or pelvis | −0.5 × force/100 N |
| overspeed (joint speed past 80% of the motor cap), torque at 95% of the limit, joint limits, joint acceleration | −1 / −0.5 / −1 / −1e-6 |
| planted foot sliding, both feet off the floor, touching the ropes | −0.5/(m/s), −2, −2 |

`scripts/test_fight_rewards.py` forces each case in the sim (falls with and
without a hit, both from red's and blue's side, double down, dragging the
feet, a drop, thrashing, legs crossed, ropes, the bell) and checks what
fires; it also checks the self-collision test against MuJoCo's own geom
distances on 150 random poses.

With no policy the fighters stand on their own: the arena holds the stance
with 4x the tracking stiffness (`gain_scale`, same torque limits, peak use
32% at the rear ankle). The tracking gains (1x) are soft by design and need
a trained policy to stay up.

Headless rendering without a GPU: `apt install libosmesa6` and set
`MUJOCO_GL=osmesa`.

## Differences from the Isaac Lab version

| | Isaac Lab | here |
|---|---|---|
| robot | `whole_body_humanoid_raw_export.urdf` | GMR's `watonomous.xml` (same robot, converted from that URDF), with its ±10 Nm per-joint force cap removed |
| collisions | convex hulls of every mesh, self-collision off | box under each foot only; meshes are visual |
| motor speed caps | `velocity_limit_sim` | DC motor model with full torque up to the cap, braking above it; forearm roll, wrist and claws use implicit position actuators instead (the explicit DC-motor PD shakes on joints that light) and have no cap |
| `undesired_contacts` reward | yes | no (mjlab's base tracking task has none) |
| PPO, rewards, observations, terminations, gains, armature, action scale | — | same values |

## Layout

```
wato_tracking/   robot.py (entity + motors), env_cfg.py (task), rl_cfg.py (PPO); __init__ registers the tasks
wato_boxing/     fighters.py (gloves, hitboxes, stance), ring.py, mdp.py (fight state, rewards, round endings), arena_env_cfg.py
scripts/         fetch_assets.sh, csv_to_npz.py, generate_moves.py, view_arena.py, test_fight_rewards.py, train.py, play.py
data/            fetched model + motions, generated NPZs (gitignored)
```
