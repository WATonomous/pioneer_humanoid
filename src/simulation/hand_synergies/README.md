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

## Caveats

- The hand model has no tendon coupling or torque limits: every joint is an independent position servo with
  the Isaac gains. If the real hand couples PIP/DIP, those joints will correlate more on hardware.
- MuJoCo collides meshes as convex hulls, so the palm is solid where the real one is a frame.
- Hand self-collision is off (as in the Isaac in-hand task).
