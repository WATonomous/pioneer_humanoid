# Whole-body motion tracking on mjlab

Motion tracking (BeyondMimic) and a two-fighter boxing arena for the **Pioneer humanoid**
(`assets/pioneer_whole_body_humanoid`), on mjlab (MuJoCo Warp, one GPU, no Isaac Sim). The training
side is mjlab's own BeyondMimic port (`mjlab.tasks.tracking`) with the task settings of the Isaac Lab
Wato tracking config in [WATonomous/Humanoid_motion_tracking](https://github.com/WATonomous/Humanoid_motion_tracking).

```
generate_moves.py ──▶ CSV ──csv_to_npz.py──▶ NPZ ──train.py──▶ policy ──play.py──▶ video / viewer
```

## The robot (`humanoid_tracking/robot.py`)

The whole-body URDF with a free-floating base, set up for this package when it loads:
- **Re-rooted:** the base frame sits at the hip joints and faces +y, with +x the robot's right (the URDF's
  root is on the floor and faces −y). Only frames change; the robot's shape and joints don't.
- **Fists:** the hands are closed into fists and their 40 finger joints removed; gloves go over them in
  the arena. That leaves **29 joints**: waist yaw, neck (2), arms (7 each, with the wrist), legs (6 each).
- **Motors:** the URDF's torque and speed per joint (real for the legs and arms, estimates for the
  waist, neck and wrists); 44.7 kg.
- **Collisions:** only a box under each sole touches the floor; the meshes are visual.

## Setup

```bash
uv sync --extra train            # Linux + NVIDIA GPU (CPU: --extra train-cpu, smoke tests only)
```

The robot is in the repo (`assets/pioneer_whole_body_humanoid`); motions are generated (below).

## Use

```bash
# 1. motions: boxing footwork for this robot, then CSV -> NPZ (50 fps) plus an MP4 of the reference
uv run scripts/generate_moves.py --output-file data/motions/moves.csv
uv run scripts/csv_to_npz.py --input-file data/motions/moves.csv --input-fps 50 \
    --output-file data/motions/moves.npz --video True

# 2. train (GPU)
uv run scripts/train.py Mjlab-Tracking-Flat-Pioneer \
    --env.commands.motion.motion-file data/motions/moves.npz \
    --env.scene.num-envs 4096 --agent.run-name moves

# 3. watch it (viser web viewer on a headless box, or record an MP4)
uv run scripts/play.py Mjlab-Tracking-Flat-Pioneer --motion-file data/motions/moves.npz \
    --checkpoint-file logs/rsl_rl/pioneer_tracking/<run>/model_<n>.pt --viewer viser
uv run scripts/play.py Mjlab-Tracking-Flat-Pioneer --motion-file data/motions/moves.npz --agent zero --viewer viser
```

Runs log to Weights & Biases by default (`--agent.logger tensorboard` to stay
local); checkpoints land in `logs/rsl_rl/pioneer_tracking/<timestamp>/`. `--agent zero` replays the
reference with no policy.

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

The stance copies the previous robot's (Wato's) orthodox stance: the same foot placement and
pelvis blade, made by IK on this robot, with the hips 7 cm below standing (deeper puts the rear
ankle at its limit) and the gloves and elbows placed from the shoulders, scaled to these longer arms.
The elbows bend only 90°, so the guard is a little more open than Wato's. Foot paths and the
centre-of-mass / hip height / heading / lean curves come from the rules above, and the legs are
solved by IK. Moves blend into each other without
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

`humanoid_boxing/` sets up red vs blue in a ring for fighting RL: scene, rewards,
round endings and opponent perception. Skill actions are not in yet.

- **Fighters**: the tracking robot recoloured, in the orthodox stance from
  `generate_moves.py`, 1.45 m apart facing each other (`distance`): lead
  gloves 23 cm apart, the lead glove 65 cm from the opponent's head.
- **Gloves and hitboxes**: a glove sphere (7 cm) round each fist; head, torso, pelvis
  and arm/leg capsules fitted to the CAD meshes (hidden, group 3). They hit
  the opponent, the ropes and the floor, never their own body. The CAD
  meshes stay visual only.
- **Ring**: 4 m between the ropes, 3 ropes, 4 posts.
- **Hit sensors**: `<attacker>_hits_<defender>_head` / `_body`, glove vs
  opponent head / torso, with contact count and net force per glove.

```bash
uv run scripts/play.py Mjlab-Boxing-Arena-Pioneer --agent zero --viewer viser
```

**Rewards and round endings** (`humanoid_boxing/mdp.py`, weights in
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

**Perception** (`humanoid_boxing/perception.py`). The policy sees the opponent
only through simulated sensors, fused by an EKF per target (opponent head,
left glove, right glove, torso):

```
D455 RGB camera (25 Hz, 40 ms late) ─┐  pixel (u, v) + apparent size → bearing + rough range
chest ultrasonic (25 Hz, 20 ms late) ─┼─▶ EKF (constant velocity) ─▶ actor: per target pos, vel, std, time unseen
(optional) D455 depth                 ─┘                              critic: the same + exact opponent state
```

- Sensors are simulated from geometry, not rendered (fast enough for
  thousands of envs): pinhole projection with the D455's 86°×60° field of
  view, occlusion by either fighter's gloves and arms (own guard included),
  pixel noise, random dropouts, latency and sensor rate.
- The ultrasonic returns the nearest thing in its 15° cone, which may be its
  own guard; a reading is used only when it fits a tracked target (3σ gate).
- Hidden targets coast on their last velocity for 0.2 s, then freeze and
  their uncertainty grows; after 1 s unseen the track is dropped.
- Observation: 32 numbers (4 targets × [position 3, velocity 3, std, time
  unseen]) in the fighter's heading frame. The critic also gets the exact
  positions and velocities (24), so a student policy can learn from it.
- Settings in `PerceptionCfg` (`CameraCfg`, `UltrasonicCfg`, `EkfCfg`),
  passed as `boxing_arena_env_cfg(perception=...)`.

Checked against MuJoCo (on the previous robot, Wato; the code is unchanged and robot-independent):

| check | result |
|---|---|
| camera occlusion vs MuJoCo ray casting (200 random arm poses) | 658/658 agree, 71 hidden |
| ultrasonic cone vs the axis ray | never longer; its own guard is the nearest thing in ~19% of readings |
| mean error, opponent standing | RGB 3.6 cm, + ultrasonic 3.6 cm, + depth 1.1 cm |
| RMS error, opponent moving 0.4 m/s | RGB 9.3 cm, + ultrasonic 6.3 cm, + depth 7.6 cm |

The camera sits on the front of the head's visor at eye height, looking forward; in the bladed stance
it points to the right of the opponent (the pelvis turns), so the opponent's left side leaves the
image first.

With no policy the fighters stand on their own: the arena holds the stance
with 4x the tracking stiffness (`gain_scale`, same torque limits); over 5 s
the pelvis sinks 4 mm and nobody falls. The tracking gains (1x) are soft by
design and need a trained policy to stay up.

Headless rendering without a GPU: `apt install libosmesa6` and set
`MUJOCO_GL=osmesa`.

## Differences from the Isaac Lab version

| | Isaac Lab | here |
|---|---|---|
| robot | the Wato humanoid | the Pioneer humanoid (`assets/pioneer_whole_body_humanoid`), hands as fists |
| collisions | convex hulls of every mesh, self-collision off | box under each foot only; meshes are visual |
| motor speed caps | `velocity_limit_sim` | DC motor model with full torque up to the cap, braking above it; forearm roll, wrists and neck use implicit position actuators instead (the explicit DC-motor PD shakes on joints that light) and have no cap |
| `undesired_contacts` reward | yes | no (mjlab's base tracking task has none) |
| PPO, rewards, observations, terminations, gains, action scale | — | same values |

## Layout

```
humanoid_tracking/   robot.py (entity + motors), env_cfg.py (task), rl_cfg.py (PPO); __init__ registers the tasks
humanoid_boxing/     fighters.py (gloves, hitboxes, stance, sensor sites), ring.py, mdp.py (fight state, rewards, round endings),
                 perception.py (camera + ultrasonic + EKF), arena_env_cfg.py
scripts/         generate_moves.py, csv_to_npz.py, train.py, play.py
data/            generated motions and NPZs (gitignored)
```
