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

## Results so far (1891 grasps from 20k trials, default `SAMPLING`)

**Variance**: 3 PCs explain 78%, 5 PCs 89%, 6 PCs 91%.

| PC | var | what it does |
|---|---|---|
| 1 | 41% | ring + pinky curl together (they either wrap the object or close into a fist) |
| 2 | 21% | thumb opposition (MCP_A_thumb) against thumb curl (PIP/DIP) |
| 3 | 16% | index + middle curl (the precision side) |
| 4 | 7% | mixed finger curl |

Spread (MCP_A_1-4) and circumduction barely load: they vary in the pre-shape prior but the grasps don't
depend on them much.

**Variance explained is not grasp quality** (`synergies eval`, 300 grasps):

| k PCs | mean | 1 | 2 | 3 | 5 | 8 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|
| k-PC posture held as the grasp | | 17% | 18% | 21% | 27% | 41% | 46% | 85% |
| k-PC pre-shape, then close to contact | 49% | 48% | 52% | 59% | 60% | 68% | 73% | 73% |

A grasp needs mm-accurate contacts that 3 PCs don't carry; used as a pre-shape plus close-to-contact
(how Ciocarlie's eigengrasp planner uses them), 3 PCs get ~80% of the full-posture ceiling.

**Pre-shape search on new objects** (`search.py`, 150 objects, 16 tries each, power grasp):

| sampler | success per try | objects grasped within 16 |
|---|---|---|
| generator prior (20-D) | 16.5% | 49% |
| Gaussian, all 20 PCs | 17.1% | 43% |
| synergy-3 | 15.2% | 24% |
| synergy-5 | 14.9% | 25% |

Synergy pre-shapes succeed as often per try but cover half as many objects: they reproduce the typical
grasp, and the unusual ones that solve awkward placements live in the dropped dimensions.

## Cube spin with sampling MPC (`mpc.py`)

Palm-up hand, 4 cm cube, predictive sampling (32 rollouts x 0.3 s every 20 ms via `mujoco.rollout`), cost
= spin about +Z at 1 rad/s + stay in the palm + don't tumble. No training. The action space differs only in
where the noise lives: all 20 joints, or the first k grasp PCs.

```bash
MUJOCO_GL=osmesa python -m hand_synergies.mpc --space joint --seconds 10 --gif out/mpc_joint.gif
python -m hand_synergies.mpc --space synergy-3 --seconds 10
```

First run (untuned): joint 140 deg in 10 s, synergy-3 102 deg, neither dropped the cube. ~0.1x real time on
4 cores.

## Caveats

- The hand model has no tendon coupling or torque limits: every joint is an independent position servo with
  the Isaac gains. If the real hand couples PIP/DIP, those joints will correlate more on hardware.
- MuJoCo collides meshes as convex hulls, so the palm is solid where the real one is a frame.
- Hand self-collision is off (as in the Isaac in-hand task).
