# hand_synergies

Grasp synergies (PCA / eigengrasps), in-hand manipulation with sampling MPC, and a thumb-mount design study
for the 20-DOF pioneer hand -- plain MuJoCo on CPU. No Isaac, no GPU, no training.

Synergies: run PCA over many stable grasp postures; the first few principal components then drive all 20
joints from a handful of numbers, `q = mean + sum_k a_k PC_k`. The grasps come from the hand itself in
simulation (not human data), so they use every joint within the real limits and geometry.

## Run

```bash
pip install mujoco numpy scipy matplotlib pillow
pip install -e src/pioneer_humanoid -e src/simulation/hand_synergies
cd src/simulation/hand_synergies            # outputs go to out/ (git-ignored)
export MUJOCO_GL=osmesa                     # headless rendering without a GPU (egl with one)

python -m hand_synergies.grasp_gen --trials 20000 --out out/grasps.npz   # ~9 min on 4 cores
python -m hand_synergies.synergies fit      # PCA, variance.png, loadings.png, pc1-4.gif
python -m hand_synergies.synergies eval     # grasps rebuilt from k PCs
python -m hand_synergies.search             # pre-shape search on new objects
python -m hand_synergies.lift --stored 600  # can the stable grasps be carried?
python -m hand_synergies.lift --trials 1000 --samplers prior,synergy-3   # pre-shape -> grasp -> lift
python -m hand_synergies.mpc --task spin|yaw|roll --seconds 10 --gif out/x.gif
python -m hand_synergies.opposition         # thumb-mount search (kinematics only, ~30 s)
```

## Files

| File | |
|---|---|
| `pioneer_humanoid/mujoco_hand.py` | the hand for MuJoCo: URDF + position actuators (Isaac gains), fingertip sites; options: thumb mount, finger-finger collisions, torque limit |
| `scene.py` | hand fixed palm-down at the origin + one resizable free object |
| `grasp_gen.py` | grasp generator: pre-shape, GraspIt-style autograsp, squeeze, 6-direction shake test |
| `synergies.py` | PCA, plots, per-PC GIFs, reconstruction test |
| `search.py` | random pre-shape search on new objects: generator prior vs synergy samplers |
| `lift.py` | grasp an object off a fixture and lift it 10 cm; or carry stored grasps |
| `mpc.py` | predictive-sampling MPC on a palm-up hand: cube spin, cube goal yaw (the Isaac task), ball rolling |
| `manip.py` | PCA of MPC's joint commands ("manipulation synergies") vs grasp synergies |
| `thumb.py`, `opposition.py` | hypothetical thumb mounts; fingertip-opposition score and mount search |

## How grasps are made (`grasp_gen.py`)

1. **Pre-shape**: random finger fan, thumb circumduction and opposition, slight curl. **Object**: sphere,
   cylinder or box of random size and orientation under the palm (palm faces -Z in the hand frame).
2. **Autograsp** (GraspIt-style): the participating digits' closing joints ramp toward the palm; when a link
   touches the object, the closing joints between it and the palm stop and the distal ones keep wrapping.
   Grasp types: power (all digits, 50%), tripod (thumb, index, middle), pinch (thumb, index). The object is
   pinned while the hand closes.
3. **Squeeze** the joints that stopped on contact by 0.08 rad, release, settle (gravity off).
4. **Shake test**: gravity along +-X, +-Y, +-Z for 0.5 s each; kept if the object moves < 15 mm and turns
   < 0.3 rad in all six, and no step went unstable.

The synergies inherit this sampling prior; every range is in `SAMPLING` in `grasp_gen.py`.

## Grasp synergies

1006 -> **2543 stable grasps from 20k trials** (2199 power, 294 tripod, 50 pinch) after the scene fixes
below. **3 PCs explain 79% of the posture variance, 5 PCs 89%, 6 PCs 91%.**

| PC | var | what it does |
|---|---|---|
| 1 | 37% | ring + pinky curl together (they either wrap the object or close into a fist) |
| 2 | 24% | thumb opposition (MCP_A_thumb) against thumb curl, with the middle finger |
| 3 | 17% | index curl with thumb opposition (the pinch side) |
| 4 | 7% | small mixed index/thumb/middle curl |

Spread (MCP_A_1-4) and circumduction barely load.

**Variance explained is not grasp quality** (`synergies eval`, 300 grasps, shake test):

| k PCs | mean | 1 | 2 | 3 | 5 | 8 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|
| k-PC posture held as the grasp | | 49% | 49% | 51% | 61% | 70% | 73% | 93% |
| k-PC pre-shape, then close to contact | 47% | 48% | 54% | 57% | 55% | 62% | 63% | 69% |

A grasp needs mm-accurate contacts that 3 PCs don't carry. Used as a pre-shape plus close-to-contact (how
Ciocarlie's eigengrasp planner uses them), 3 PCs reach ~80% of the 20-PC pre-shape ceiling.

**Pre-shape search on new objects** (`search.py`, 150 objects, 16 tries each, power grasp):

| sampler | success per try | objects grasped within 16 tries |
|---|---|---|
| generator prior (20-D) | 21.4% | 57% |
| Gaussian over all 20 PCs | 24.1% | 55% |
| synergy-1 | 25.5% | 39% |
| synergy-3 | 26.7% | 38% |
| synergy-5 | 26.9% | 40% |

Synergy pre-shapes succeed *more* often per try (27% vs 21%, ~2400 tries each) but grasp a third fewer
distinct objects: they concentrate on the typical grasp, and the unusual ones that solve awkward
placements live in the dropped dimensions.

**Grasp and lift** (`lift.py`): the hand (palm down, on a wrist that slides vertically) closes on an object
held by a fixture (a weld), the fixture lets go, the object settles into the grasp, and the wrist lifts 10 cm
in 1 s and holds 1 s. Success: it rose >= 8 cm, stayed within 3 cm of where it sat in the hand, and no step
went unstable. Same objects and placement as `grasp_gen`. `out/lift.gif`.

Carrying the shake-test-stable grasps (`--stored 600`, postures rebuilt from k PCs plus the same squeeze):

| posture | as stored | 10 PCs | 5 PCs | 3 PCs | 1 PC |
|---|---|---|---|---|---|
| carried 10 cm | 78% | 67% | 53% | 43% | 40% |

Pre-shape -> autograsp -> lift, 1000 trials each (finger collisions on, +-1.1% s.e.):

| pre-shape sampler | stock thumb | near36 thumb |
|---|---|---|
| generator prior | 14.1% | 13.3% |
| Gaussian, all 20 PCs | 13.7% | 15.0% |
| synergy-1 | 16.5% | 18.3% |
| synergy-3 | 14.9% | 19.0% |
| synergy-5 | 15.1% | 17.2% |

Synergy pre-shapes lift a few points more than random ones -- clearly on near36 (+6, ~3.5 s.e.),
marginally on stock -- even though the synergies were fitted to stock grasps. A first version lifted tall
objects off a table and lifted none: the stock thumb hangs ~10.6 cm below the palm, so it hits the table
beside anything shorter.

## In-hand manipulation with sampling MPC (`mpc.py`)

Palm-up hand, predictive sampling: every 20 ms, 32 perturbations of a 3-knot plan over 0.3 s are rolled out
with `mujoco.rollout` and the cheapest is kept. No training. ~0.05-0.1x real time on 4 cores.

**spin** (cube, 4 cm): spin about the palm normal at 1 rad/s, stay in the palm, don't tumble. Degrees turned
in 6 s, 3 seeds; the action space differs only in where the noise lives:

| action space | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| joint (20-D) | 116 | 107 | 73 | 99 |
| manipulation synergy-3 | 58 | 67 | 99 | 75 |
| manipulation synergy-5 | 83 | 142 | 105 | 110 |
| grasp synergy-3 | 26 | 40 | 56 | 41 |
| grasp synergy-5 | 25 | 12 | 61 | 33 |

No run drops the cube. Joint space and the manipulation synergies (PCA of joint-space MPC's own commands)
are about equal; **the grasp synergies spin the cube 2-3x less**: confined to the grasp subspace, the
planner can't make the finger-against-finger motions spinning needs (see the subspace comparison below). Every space spins
far below the target (~0.2 rad/s), matching the Isaac in-hand log: palm-normal spin is hard for this hand.
A sweep over noise (0.25-1.0), horizon (0.15-0.5 s) and samples (32/64) didn't do better without drops.
The motion is jerky because the planner keeps the single best sample; MPPI averaging, a command-change
penalty or a low-pass (`--smooth`, `--w-ctrl`, `--filter`) halve the jerk but cut the spin to 13-54 deg /
6 s -- the planner spins the cube *with* the flicks.

**yaw** -- the Isaac in-hand task: turn the cube to a random goal yaw; success when the orientation error is
< 0.4 rad, then a new goal (at least 0.8 rad away). The cost tracks a spin toward the goal (2 x the signed
yaw error, capped at 1 rad/s); with the orientation error alone the planner just held still. Stock hand,
8 seeds x 20 s: **12 goals (one per ~11 s), 2 drops**, mean orientation error 0.95-2.44 rad per seed. The
Isaac PPO policy's best mean error was ~0.94 rad (TRAINING_LOG.md); the metrics aren't defined identically,
so this is a rough comparison, not a win. `out/mpc_yaw.gif` (green ghost cube = goal).

**roll** -- a 1.5 cm ball rolled to random targets within +-2 cm on the palm (each at least 1.5 cm from the
ball), success within 1 cm: **40-58 targets in 8-10 s, never dropped** (3 seeds). `out/mpc_roll.gif`.

**Grasp vs manipulation synergies** (`manip.py`: PCA of the joint targets from two 15 s joint-space spin runs):

| k | grasp PCs 1-k capture of the spinning motion | random k-D subspace |
|---|---|---|
| 1 | 5% | 5% |
| 2 | 25% | 10% |
| 3 | 27% | 15% |
| 5 | 38% | 25% |
| 10 | 72% | 50% |

The spinning motion is low-dimensional itself (its own 3 PCs: 70%) and overlaps the grasp synergies more
than chance, but most of it lies elsewhere (principal angles to the grasp top-3: 40, 65, 81 deg): grasp
synergies move the fingers together, spinning moves them against each other. Two runs only; the logged
targets include the planner's isotropic sampling noise.

## Thumb mount study (`thumb.py`, `opposition.py`)

The Isaac log concluded palm-normal spin needs a thumb that opposes the fingers the way human, Shadow and
Allegro thumbs do. As built, the thumb base sits mid-palm between index and middle, the thumb hangs ~10.6 cm
below the palm, and its MCP_A swing carries it toward the fingertips (head-on, claw-like opposition).
`hand_spec(thumb_pos=..., thumb_yaw_deg=...)` moves the base and turns the chain about the palm normal
(`--thumb` on `mpc`, `grasp_gen`, `lift`).

**Opposition score** (kinematics only): the fraction of each fingertip's workspace the thumb tip gets within
1 cm of. "Buildable" = the thumb's base links don't collide with any finger (straight / half / fully curled).

| mount | index | middle | ring | pinky | mean | buildable |
|---|---|---|---|---|---|---|
| stock | .25 | .39 | .48 | .50 | .41 | yes |
| yaw90 (swing across the palm) | .07 | .02 | .27 | .54 | .22 | no |
| radial45 (index side near the wrist) | .19 | .14 | .02 | .00 | .09 | yes |
| radial90 (index side, swing across) | .03 | .06 | .13 | .00 | .05 | yes |
| best of 3000 random mounts | .68 | .65 | .63 | .40 | .59 | no |
| **near36** (best buildable: base 12 mm toward the knuckles, turned 36 deg) | .42 | .50 | .57 | .52 | .50 | yes |

The stock thumb already reaches all four fingertips; "human-like" radial mounts barely do (their arc ends
over the palm near the wrist, they touched the object in only ~15% of grasps). The best unconstrained mounts
put the base inside the finger roots; with finger collisions on, such a thumb gets thrown out of its range.

**near36 vs stock, finger collisions on:**

| | stock | near36 |
|---|---|---|
| cube spin, 20 runs (10 seeds x 2 directions), 8 s | 90 +- 11 deg, 1 drop | 112 +- 10 deg, 0 drops (Welch p = 0.12) |
| cube goal-yaw task, 6 seeds x 20 s | 10 goals, 0 drops | 13 goals, 2 drops |
| stable grasps, objects placed over a wide area, 20k trials | 821 | 853 |
| ... of which tripod / pinch | 89 / 19 | 114 / 28 |
| thumb touches the object | 63% | 75% |
| pre-shape -> grasp -> lift, best sampler | 16.5% (synergy-1) | 19.0% (synergy-3) |

Verdict: near36 uses the thumb more and makes ~30% more precision grasps, and spins ~20% more, but the
spin and total-grasp differences aren't statistically solid and the goal task is a wash. It's the
direction the kinematics point (more fingertip opposition), not a proven upgrade; worth a CAD check of
whether the base fits there. Larger redesigns don't fit without moving the fingers too. Turning finger
collisions on cut the stock spin from 119 to 87 deg / 8 s: part of the planner's spin came from fingers
passing through each other.

## Bugs found on the way (and what they affected)

- **Soft contacts**: MuJoCo's default contact let fingers sink 3-12 mm into objects (visible clipping).
  Fixed (`scene.CONTACT_SOLREF/SOLIMP`); all numbers above use the stiff contact.
- **Object centre of mass ~11 cm off**: setting the object's inertia explicitly left its inertial frame at
  the body's spawn position. Every grasp result before the fix held a lopsided object. Fixed (`ipos`).
- **Missed contacts on resized objects**: MuJoCo's per-body bounding volume is built at compile time, so an
  object resized larger afterwards missed contacts until fingers were 1-4 mm inside. Objects are now
  compiled at the largest size used (`scene.MAX_SIZE`) and only shrunk.
- **Silent resets**: MuJoCo resets an unstable simulation and carries on; a squeezed cylinder can spin up and
  blow up (~1% of grasp trials). Those now count as failures (`GraspGen.unstable`).
- **Teleported wrist**: moving the hand by setting a mocap pose each step gives contact friction no hand
  velocity, so a held object slides out. The lift moves the wrist on a slide joint instead.
- **Pinned objects**: resetting a pinned object's pose each step lets fingers sink in; the stored overlap
  can fire a small object off at tens of m/s on release. `lift` uses a weld fixture; `grasp_gen` still pins
  (it can only cost grasps, not fake them).
- **Grasp-synergy MPC rows and the first thumb grasp counts** came from the buggy grasp scene; they were
  rerun or dropped.

## Caveats

- Every joint is an independent position servo with the Isaac gains and no torque limit (the real motors'
  torque isn't in the repo; URDF effort=0). A 0.08 rad squeeze at kp 50 is ~4 Nm per joint -- 100+ N on a
  small object -- far more than a hand this size produces. `hand_spec(torque_limit=...)` /
  `grasp_gen --torque-limit` clamp it. With an assumed 0.5 Nm per joint: 2699 stable grasps (vs 2543), the
  same synergies (top-3 subspaces within 1.6-7.6 deg, 3 PCs = 77%), and 90% of them carried 10 cm (vs 78%;
  gentler squeezes release more smoothly). The synergy findings don't hinge on the motor strength.
- No tendon coupling: if the real hand couples PIP/DIP, those joints will correlate more on hardware.
- MuJoCo collides meshes as convex hulls, so the palm is solid where the real one is a frame.
- Finger-finger collisions are off by default (as in the Isaac in-hand task) except where noted.
