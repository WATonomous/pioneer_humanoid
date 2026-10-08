# hand_synergies

Grasp synergies (PCA / eigengrasps) for the 20-DOF pioneer hand, in plain MuJoCo on CPU. No Isaac, no GPU.

The idea: run PCA over many stable grasp postures; the first few principal components ("synergies") then
drive all 20 joints from a handful of numbers, `q = mean + Σ a_k · PC_k`. A 3-5-D hand is far cheaper to
search, plan or learn in than a 20-D one (eigengrasp planning, sampling MPC, RL action spaces, teleop).

The grasps come from the hand itself in simulation, not from human data, so they use every joint
(including finger spread and thumb circumduction) within the real limits and geometry.

## Run

```bash
pip install mujoco numpy matplotlib pillow
pip install -e src/pioneer_humanoid -e src/simulation/hand_synergies
cd src/simulation/hand_synergies

# 1. stable grasps (~80 ms per trial per core, ~8% pass the shake test)
python -m hand_synergies.grasp_gen --trials 20000 --out out/grasps.npz

# 2. PCA, plots, per-PC GIFs (headless: MUJOCO_GL=egl, or osmesa without a GPU)
MUJOCO_GL=osmesa python -m hand_synergies.synergies fit --grasps out/grasps.npz --out out

# 3. how many synergies a grasp needs: rebuild each from k PCs and re-run the shake test
python -m hand_synergies.synergies eval --grasps out/grasps.npz --out out
```

`out/` is git-ignored.

## How grasps are made (`grasp_gen.py`)

1. **Pre-shape**: random finger fan (index and pinky spread opposite ways), thumb circumduction and initial
   opposition, slight curl. **Object**: sphere, cylinder or box of random size and orientation, under the
   palm (the palm faces -Z in the hand frame).
2. **Autograsp** (GraspIt-style): the participating digits' closing joints ramp toward the palm; when a link
   touches the object, the closing joints between it and the palm stop and the distal ones keep wrapping.
   Grasp types: power (all digits, 50%), tripod (thumb + index + middle), pinch (thumb + index). The object is
   pinned while the hand closes.
3. **Squeeze** the joints that stopped on contact by 0.08 rad, release the object, settle.
4. **Shake test**: gravity along ±X, ±Y, ±Z, 0.5 s each. Kept only if the object moves < 15 mm and turns
   < 0.3 rad in all six.

The synergies inherit the sampling prior: every range is in `SAMPLING` in `grasp_gen.py`.

## Files

| File | |
|---|---|
| `pioneer_humanoid/mujoco_hand.py` | the hand for MuJoCo: URDF + position actuators (gains from `hand_cfg.py`), fingertip sites, closing directions |
| `hand_synergies/scene.py` | hand fixed at the origin + one resizable free object (one compiled model per run) |
| `hand_synergies/grasp_gen.py` | grasp generator (multiprocess) |
| `hand_synergies/synergies.py` | PCA, plots, GIFs, reconstruction test |

Outputs in `out/`: `grasps.npz` (posture `q`, commanded `ctrl`, object, contacts per digit),
`synergies.npz` (`mean`, `components`, `stds`, `variance_ratio`), `variance.png`, `loadings.png`,
`synergies.png`, `pc1.gif`...`pc4.gif`, `reconstruction.png/.csv`.

## Results (stiff contacts; default `SAMPLING`)

All numbers below are from runs with the contact settings in `scene.CONTACT_SOLREF/SOLIMP`. An earlier
round used MuJoCo's default soft contacts, which let fingers sink 3-12 mm into objects (visible clipping);
those numbers are superseded.

**Grasps**: 1006 stable from 20k trials (886 power, 104 tripod, 16 pinch -- pinches rarely pass the shake
test). **Variance**: 3 PCs explain 78%, 5 PCs 88%, 6 PCs 91%.

| PC | var | what it does |
|---|---|---|
| 1 | 38% | ring + pinky curl together (they either wrap the object or close into a fist) |
| 2 | 25% | thumb opposition (MCP_A_thumb) against thumb curl, with the middle finger |
| 3 | 15% | index curl with thumb opposition (the pinch side) |
| 4 | 7% | small mixed thumb/index/middle curl |

Spread (MCP_A_1-4) and circumduction barely load: they vary in the pre-shape prior but the grasps don't
depend on them much.

**Variance explained is not grasp quality** (`synergies eval`, 300 grasps):

| k PCs | mean | 1 | 2 | 3 | 5 | 8 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|
| k-PC posture held as the grasp | | 30% | 29% | 30% | 37% | 52% | 63% | 95% |
| k-PC pre-shape, then close to contact | 42% | 46% | 47% | 51% | 50% | 59% | 57% | 58% |

A grasp needs mm-accurate contacts that 3 PCs don't carry; used as a pre-shape plus close-to-contact
(how Ciocarlie's eigengrasp planner uses them), 3 PCs reach ~90% of the 20-PC pre-shape ceiling.

**Pre-shape search on new objects** (`search.py`, 150 objects, 16 tries each, power grasp):

| sampler | success per try | objects grasped within 16 |
|---|---|---|
| generator prior (20-D) | 7.8% | 31% |
| Gaussian, all 20 PCs | 7.8% | 24% |
| synergy-1 | 7.6% | 17% |
| synergy-3 | 7.4% | 17% |
| synergy-5 | 7.8% | 21% |

Synergy pre-shapes succeed as often per try but cover about half as many objects: they reproduce the
typical grasp, and the unusual ones that solve awkward placements live in the dropped dimensions.

## Cube spin with sampling MPC (`mpc.py`)

Palm-up hand, 4 cm cube, predictive sampling (32 rollouts x 0.3 s every 20 ms via `mujoco.rollout`), cost
= spin about +Z at 1 rad/s + stay in the palm + don't tumble. No training. The action space differs only in
where the noise lives: all 20 joints, or the first k PCs of a basis (grasp or manipulation synergies).
An episode ends when the cube falls off the hand.

```bash
MUJOCO_GL=osmesa python -m hand_synergies.mpc --space joint --seconds 10 --gif out/mpc_joint.gif
python -m hand_synergies.mpc --space synergy-3 --seconds 10
python -m hand_synergies.mpc --space synergy-3 --synergies out/manip_synergies.npz
```

~0.05-0.1x real time. Degrees turned in 6 s, 3 seeds each (noise 0.5 rad, horizon 0.3 s, 32 samples):

| action space | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| joint (20-D) | 116 | 107 | 73 | 99 |
| grasp synergy-3 | 88 | 72 | 81 | 80 |
| grasp synergy-5 | 103 | 71 | 73 | 82 |
| manipulation synergy-3 | 58 | 67 | 99 | 75 |
| manipulation synergy-5 | 83 | 142 | 105 | 110 |

No run drops the cube in 6 s and no space clearly wins: seed spread is about as big as any gap. Every space
spins far below the 1 rad/s target (~0.2 rad/s), which matches the Isaac in-hand log: palm-normal spin is
hard for this hand whatever the controller (see the thumb study below). Before the contact fix, a sweep over
noise (0.25-1.0), horizon (0.15-0.5 s) and samples (32/64) didn't beat ~120 deg / 6 s without dropping.

**Grasp synergies vs manipulation synergies** (`manip.py`, PCA of the joint targets from two 15 s
joint-space MPC runs):

| k | grasp PCs 1-k capture of the spinning motion | random k-D subspace |
|---|---|---|
| 1 | 6% | 5% |
| 2 | 24% | 10% |
| 3 | 27% | 15% |
| 5 | 44% | 25% |
| 10 | 70% | 50% |

The spinning motion is low-dimensional itself (its own 3 PCs: 70%). Grasp synergies overlap it more than
chance (27% vs 15% at k=3) but miss most of it (principal angles to the grasp top-3: 39, 65, 82 deg). With
soft contacts the overlap was at chance, so this is sensitive to the setup; two runs only. The logged
targets also include the planner's isotropic sampling noise.

## Caveats

- The hand model has no tendon coupling or torque limits: every joint is an independent position servo with
  the Isaac gains. If the real hand couples PIP/DIP, those joints will correlate more on hardware.
- MuJoCo collides meshes as convex hulls, so the palm is solid where the real one is a frame.
- Hand self-collision is off by default (as in the Isaac in-hand task); `hand_spec(finger_collisions=True)`
  / `mpc --finger-collisions` makes the digits collide with each other.
