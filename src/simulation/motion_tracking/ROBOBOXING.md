# Roboboxing: Wato learns to box

Two Wato humanoids box in a simulated ring with full physics (mjlab / MuJoCo Warp on one GPU). A **fight
policy** decides what to throw and when. **Tracking skills** carry each move out while keeping the robot
balanced. The opponent is first a still target, then a scripted fighter, then copies of the policy itself.

```
                ┌──────────────── fight policy (decides) ────────────────┐
 camera + ──▶  │ sees: itself + the opponent (camera / ultrasonic / EKF)  │ ──▶ "jab, a bit faster, aimed lower"
 ultrasonic     └─────────────────────────────────────────────────────────┘
                                         │ move + technique
                                         ▼
                ┌──────────────── move player ───────────────────────────┐
                │ streams that move's clip (from guard), placed where the │
                │ robot stands                                            │
                └─────────────────────────────────────────────────────────┘
                                         │ reference pose, 50 Hz
                                         ▼
                ┌──────────── tracking skills (carry it out) ────────────┐
                │ punch skill (guard + 5 punches) / footwork skill        │ ──▶ 28 motor targets
                │ follow the reference and keep the robot balanced        │
                └─────────────────────────────────────────────────────────┘
```

All paths below are relative to `src/simulation/motion_tracking/`.

---

## Quick start: watch it or box against it

You need the trained checkpoints. They're not in git (`logs/` is gitignored), so ask for them and put them
in the same paths:

| What | Path |
|---|---|
| punch skill | `logs/rsl_rl/wato_tracking/2026-10-17_00-01-11_power_v8/model_22000.pt` |
| footwork skill | `logs/rsl_rl/wato_tracking/2026-10-17_00-34-56_footwork_v1/model_23500.pt` |
| fight policy | `logs/rsl_rl/wato_fight/2026-10-17_14-52-15_fight_v7g/model_8000.pt` |
| move library | `data/moves/` (or build it: see phase 1) |

```bash
uv sync --extra train
./scripts/fetch_assets.sh          # robot model + base motions -> data/

FIGHT=logs/rsl_rl/wato_fight/2026-10-17_14-52-15_fight_v7g/model_8000.pt

# box against it: you are red, it is blue
uv run scripts/fight_teleop.py --opponent ai --checkpoint-file $FIGHT --viewer viser

# watch it fight itself
uv run scripts/play.py Mjlab-Boxing-Fight-Wato --num-envs 1 --checkpoint-file $FIGHT --viewer viser

# one robot, no opponent: drive the punch and footwork skills yourself and hit a target ball
uv run scripts/teleop.py --mode policy --viewer viser \
    --checkpoint-file logs/rsl_rl/wato_tracking/2026-10-17_00-01-11_power_v8/model_22000.pt \
    --footwork-checkpoint-file logs/rsl_rl/wato_tracking/2026-10-17_00-34-56_footwork_v1/model_23500.pt
```

Open **http://localhost:8080**. On a remote machine, forward the port first:
`ssh -N -L 8080:localhost:8080 <you>@<host>`. In VS Code Remote-SSH, the Ports tab does it for you.

**`fight_teleop.py` keys** (type them in the terminal, or click the "Your moves" buttons in the browser).
Moves queue up to 4 and start when you're back in guard:

| Keys | Move |
|---|---|
| `i` `k` `j` `l` | step forward / back / left / right |
| `u` `o` | pivot left / right |
| `f` `g` | slip left / right |
| `r` `t` | bob-and-weave left / right |
| `space` | duck |
| `c` | deep duck |
| `1` `2` `3` `4` `5` | jab, cross, lead hook, rear hook, lead uppercut |
| `x` | clear the queue |
| `?` | help |

The opponent can also be `--opponent hold` (a still target) or `--opponent scripted` (random moves).
`teleop.py` uses the same keys; each punch puts a target ball where it lands and prints the momentum it got.
`--mode reference` replays the clips with no physics, to check the moves themselves.

---

## How it was built: the 5 phases

| Phase | What it adds | Main files |
|---|---|---|
| 1. Move library | 18 boxing moves as clips that all start and end in the same guard, chained into training combos | `scripts/build_move_library.py`, `scripts/retarget_mixamo_punches.py` |
| 2. Tracking skills | train the robot to perform the moves under physics: hard punches, bigger steps, always balanced | `humanoid_tracking/power.py`, `humanoid_tracking/env_cfg.py` |
| 3. Teleop | press a key, that move comes out (one robot, punching a target ball) | `scripts/teleop.py`, `humanoid_tracking/teleop.py` |
| 4. Fight policy | two robots box: the policy picks moves, learns against a curriculum and self-play | `humanoid_boxing/skills.py`, `humanoid_boxing/impact.py`, `humanoid_boxing/fight_env_cfg.py` |
| 5. Box against the bot | you vs the trained bot (or a still / scripted opponent) | `scripts/fight_teleop.py` |

The arena itself (ring, gloves and hitboxes, fight rewards, camera + ultrasonic perception) is described in
`README.md`, under "Boxing arena".

### Phase 1: move library

Every move is its own clip, and **every clip starts and ends in the same guard pose**. That's what lets any
move follow any other.

- **Punches** (jab, cross, lead hook, rear hook, lead uppercut) are retargeted from motion capture:
  Mixamo boxing takes and ACCAD martial-arts clips (AMASS). The whole body moves (hip turn, weight shift),
  slowed only where a motor would pass its speed limit.
- **Footwork and defence** (steps, pivots, slips, weaves, ducks) are generated by `scripts/generate_moves.py`.

**Source data (not in git, licensed):**
- The Mixamo take `Boxing_mixamo.fbx` in `../../../../motion_data/Combat/` (read through `assimp`, which
  must be on your PATH).
- ACCAD `Male2MartialArtsPunches_c3d` (AMASS SMPL-X) in `data/amass/ACCAD/`.
- The SMPL-X body models in `../../../../body_models/smplx/`.

One command builds everything:
- all 18 moves into `data/moves/` (`library.yaml`, `csv/`, `npz/`), with each punch's impact labelled;
- the punch and footwork combos into `data/motions/` (`punch_training.*`, `footwork_training.*`).

```bash
uv run scripts/build_move_library.py
uv run scripts/build_move_library.py --combos-only   # rebuild only the combos, from the existing moves
```

### Phase 2: tracking skills

BeyondMimic-style tracking trains one policy per motion. To learn many moves, each policy is trained on a
**combo**: the moves chained end to end. Training starts at random points in the combo, thousands of
robots at once, so it practises every move (and the joins through guard) in short pieces. The policy only
sees "where to be right now", so a move played on its own looks exactly like the same move inside the
combo.

**Stages:**
- **Stage 1 (tracking):** copy the punch combo.
- **Stage 2 (power):** keep tracking, but put a real 2–40 kg target ball where each punch lands. Rewards
  cover ball momentum, hips leading, weight transfer and staying steady. Tracking is loosened around each
  punch, so the robot finds its own mechanics.
- **Footwork:** the power task without punch rewards, rewarding bigger steps with balance. It starts from
  the punch skill.

```bash
# stage 1
uv run scripts/train.py Mjlab-Tracking-Flat-Wato --env.commands.motion.motion-file data/motions/punch_training.npz \
    --env.scene.num-envs 4096 --agent.logger tensorboard --agent.run-name punches
# each punch's momentum baseline (the power reward compares against it)
uv run scripts/measure_power_baseline.py --checkpoint-file logs/rsl_rl/wato_tracking/<punches run>/model_<n>.pt
# stage 2: power, continuing stage 1
uv run scripts/train.py Mjlab-Tracking-Flat-Wato-Power --env.commands.motion.motion-file data/motions/punch_training.npz \
    --env.scene.num-envs 4096 --agent.logger tensorboard --agent.run-name power \
    --agent.resume True --agent.load-run <punches run> --agent.load-checkpoint model_<n>.pt
# footwork, continuing the power skill
uv run scripts/train.py Mjlab-Tracking-Flat-Wato-Footwork --env.commands.motion.motion-file data/motions/footwork_training.npz \
    --env.scene.num-envs 4096 --agent.logger tensorboard --agent.run-name footwork \
    --agent.resume True --agent.load-run <power run> --agent.load-checkpoint model_<n>.pt
```

The shipped skills came from a longer chain of resumed runs:
generated-moves tracker → `punches_v1` → `power_v1`…`v8` → `footwork_v1`.

### Phase 3: teleop (one robot)

```bash
uv run scripts/teleop.py --mode reference --viewer viser    # replay the clips, no physics policy
uv run scripts/teleop.py --mode policy --viewer viser \
    --checkpoint-file <punch skill> --footwork-checkpoint-file <footwork skill>
```

These use the same keys as `fight_teleop.py`. Each punch places a 2 kg target ball where it lands, and
prints the momentum it got.

### Phase 4: fight policy

Task `Mjlab-Boxing-Fight-Wato`:
- **Red is the learner.** Its policy picks 1 of 18 moves (a new move starts only in guard). With each
  punch it also picks a **technique**: small arm-joint offsets (up to 0.1 rad) and a speed of 0.75–1.25×.
  Both fighters' moves are carried out by the frozen skills from phase 2. Set their paths in
  `PUNCH_CHECKPOINT` / `FOOTWORK_CHECKPOINT` in `humanoid_boxing/fight_env_cfg.py`.
- **What it sees:** its own state, plus the opponent through the simulated camera + ultrasonic + EKF only.
  The critic also gets the true opponent state.
- **Opponent curriculum** (`OPPONENT_STAGES`, fractions of the run):
  - first 20%: still target;
  - 20–35%: scripted (random footwork, defence and punches);
  - 35–100%: 80% self-play against snapshots of red (a pool of 8), plus 15% scripted and 5% still.

**Rewards** (`humanoid_boxing/impact.py`):

| Reward | Points |
|---|---|
| Clean punch: a punch move landing on head or torso at ≥ 0.4 m/s (leaning and bumps don't count), one per punch | 15 × zone × speed × counter bonus × force (below) |
| Balance damage per clean punch | 1 per degree of tilt past stance + 5°; 10 per metre pushed back; +20 stagger (tilt past 35°); +100 knockdown |
| Clean punch taken | −5 × its value |
| Punching range (0.8–1.05 m; guard gloves already touch at ~0.7 m) | small reward for holding it and for punching from it; small penalty for crowding inside 0.75 m |
| Realism penalties (overspeed, torque, joint limits, jerkiness, foot slip) | the arena's, at 0.25× |
| Stability, falls and ropes | the arena's |

Clean punch factors:
- **zone:** head 1.0, liver/ribs 0.8, solar plexus 0.7, chest 0.4;
- **speed:** glove speed ÷ 1 m/s, capped at 1.5;
- **counter bonus:** 1.5 if the opponent was mid-punch;
- **force:** peak contact force ÷ 50 N, capped at 2.

```bash
uv run scripts/train.py Mjlab-Boxing-Fight-Wato --env.scene.num-envs 4096 --agent.logger tensorboard --agent.run-name fight
```

This runs 6000 iterations, about 2–2.7 s each on one GPU (3.5–4.5 h). The curriculum restarts on every
resume; to resume straight into self-play, see the comment under `OPPONENT_STAGES`.

**Watching training:** `tensorboard --logdir logs/rsl_rl/wato_fight`. The punch numbers are logged per
round under `Curriculum/punches/` (`clean_per_round`, `head`, `liver_ribs`, `solar_plexus`, `chest`,
`force_per_punch_N`, `taken_per_round`), and the opponent mix under `Curriculum/opponent/`.

**What healthy training looks like:**
- clean punches above 0.5 per round by iteration ~300;
- clean punches above 6 by the end of the still stage;
- round length (`Train/mean_episode_length`) ~1000 steps (no falls).

**Results** (`fight_v7g/model_8000`):

| Against | Clean punches per round | Punches taken | Force per punch | Falls |
|---|---|---|---|---|
| Still target | 10.3 | 0 | 88 N | 0 |
| Itself | 9.0 | 8.9 | 89 N | 1 in 128 rounds |

It's mostly a jabber: jabs are 57–76% of its time, with step-forward, crosses and some step-left. Hooks,
uppercuts and defence are barely used yet.

### Phase 5: box against the bot

```bash
uv run scripts/fight_teleop.py --opponent hold|scripted|ai [--checkpoint-file <fight policy>] --viewer viser
```

You box as red; the terminal prints every punch landed, knockdowns and the round score. Keys are in the
quick start.

---

## Every file

What each file does and what needs it. Paths are relative to `src/simulation/motion_tracking/`.

**Foundation** (tracking port, arena, perception; everything below builds on it)

| File | What it does | Needed for |
|---|---|---|
| `pyproject.toml`, `uv.lock` | Python dependencies (mjlab, MuJoCo Warp), pinned | everything (`uv sync`) |
| `.gitignore` | keeps `data/`, `logs/`, `videos/` and `wandb/` out of git | repo hygiene |
| `README.md` | the tracking port and arena docs (this file covers roboboxing) | docs |
| `scripts/fetch_assets.sh` | downloads the Wato model and base motions into `data/` | everything (run once) |
| `scripts/csv_to_npz.py` | converts a motion CSV into the NPZ format training reads | building the move library |
| `scripts/generate_moves.py` | generates the footwork and defence moves (steps, pivots, slips, weaves, ducks) by IK | building the move library |
| `scripts/train.py`, `scripts/play.py` | entry points: register the tasks, then run mjlab's train / play | training, watching |
| `humanoid_tracking/robot.py` | the Wato robot: model, motors, speed caps, action scale | every task |
| `humanoid_tracking/env_cfg.py` | the tracking task (BeyondMimic) with Wato's settings, plus the power and footwork variants | training the skills |
| `humanoid_tracking/rl_cfg.py` | PPO settings for tracking (and the power stage) | training the skills |
| `humanoid_tracking/__init__.py` | registers the tracking tasks: `Mjlab-Tracking-Flat-Wato`, `-Power`, `-Footwork`, and `-No-State-Estimation` (a variant without base velocity in the policy's input, for a real robot without a velocity estimate) | training the skills |
| `humanoid_boxing/fighters.py` | turns the robot into a fighter: team colours, gloves, hitboxes, the boxing stance, sensor mount points | the arena |
| `humanoid_boxing/ring.py` | the ring: floor, ropes, posts | the arena |
| `humanoid_boxing/mdp.py` | fight state each step (hits, falls, contacts) and the arena's rewards and round endings | the arena, fight training |
| `humanoid_boxing/perception.py` | simulated D455 camera + chest ultrasonic, fused by a Kalman filter into what each fighter sees of the other | fight training |
| `humanoid_boxing/arena_env_cfg.py` | the two-fighter scene: fighters, ring, sensors, rewards, observations | fight training builds on it |
| `humanoid_boxing/__init__.py` | registers `Mjlab-Boxing-Arena-Wato` (and, from phase 4, the fight task) | viewing the arena, fight training |

**Phase 1: move library**

| File | What it does | Needed for |
|---|---|---|
| `scripts/build_move_library.py` | one command builds all motion data: the 18 move clips, each punch's impact label, and the punch and footwork training combos | training the skills; the fight policy reads `data/moves/` |
| `scripts/retarget_mixamo_punches.py` | converts recorded punches (Mixamo boxing take, ACCAD clips) onto Wato's body, within its motor speed limits | building the move library |
| `humanoid_tracking/target_ball.py` | the punchable target ball (touches only the gloves) | power training, teleop |

**Phase 2: tracking skills**

| File | What it does | Needed for |
|---|---|---|
| `humanoid_tracking/power.py` | the power stage: a real 2–40 kg target at each punch's impact point; rewards for momentum, weight transfer and balance | training the punch skill |
| `scripts/measure_power_baseline.py` | measures each punch's momentum before power training; the power reward scores against it | training the punch skill |
| `humanoid_tracking/env_cfg.py`, `rl_cfg.py`, `__init__.py` (changes) | add and register the power and footwork tasks | training the skills |

**Phase 3: teleop**

| File | What it does | Needed for |
|---|---|---|
| `humanoid_tracking/teleop.py` | move scheduler: queues moves and places each one where the robot stands | `scripts/teleop.py` |
| `scripts/teleop.py` | drive one robot by keyboard with a trained skill, punching a target ball; also replays clips with no policy | checking skills and clips by eye (optional: `fight_teleop.py --opponent hold` covers most of it) |

**Phase 4: fight policy**

| File | What it does | Needed for |
|---|---|---|
| `humanoid_boxing/skills.py` | runs both fighters: plays each chosen move from guard, runs the frozen punch / footwork skills, applies punch technique; opponent modes and curriculum | fight training, `fight_teleop.py` |
| `humanoid_boxing/impact.py` | scores punches (clean punches by zone, speed, force, counters), balance damage and punching range; logs punch stats | fight training |
| `humanoid_boxing/fight_env_cfg.py` | the fight task: reward weights, curriculum, what the policy sees, skill checkpoint paths | fight training |
| `humanoid_boxing/distribution.py` | the policy's output: a move plus technique, learning only from steps where a move can start | fight training, loading the fight policy |
| `humanoid_boxing/rl_cfg.py` | PPO settings, and the runner that hands the policy to the opponent for self-play | fight training, loading the fight policy |
| `humanoid_boxing/__init__.py`, `scripts/train.py` (changes) | register the fight task; make `train.py` see it | fight training |

**Phase 5: box against the bot**

| File | What it does | Needed for |
|---|---|---|
| `scripts/fight_teleop.py` | you vs a still, scripted or trained opponent, by keyboard or browser buttons | playing against the bot |
| `ROBOBOXING.md` | this guide | docs |
| `README.md` (change) | links here | docs |

## Record a video

`play.py` can record with `--video True --video-length <frames>` (50 frames per second). Run headless with
`MUJOCO_GL=egl`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No space left on device` while compiling GPU kernels on a cluster node | `/tmp` is tiny there: `export TMPDIR=/var/tmp/$USER` |
| Browser shows nothing | forward the viser port (8080, or the one printed in the viser box) |
| Clean punches stay 0 and it steps forward constantly | the old walk-into-guard exploit: start fresh. Don't resume from a policy that does it |
| Red falls after its own punches | lower `MAX_OFFSET` in `humanoid_boxing/skills.py` |
| `OSError` deep-copying the policy in a custom script | run the actor under `torch.no_grad()` |

## Limits and next steps

- **One move at a time, always through guard.** Punching while stepping, or a hook straight out of a pivot,
  isn't in the training data. Adding blended clips to the library, or a general motion tracker, would
  allow it.
- **The camera is simulated from geometry.** The real robot needs a detector for the opponent's head,
  gloves and torso in the D455 image.
- **Checkpoints and data aren't versioned.** Hosting them (e.g. on Hugging Face) would make the quick
  start one command.
