# Rubik’s cube pick and place with ACT

The WATcloud job trains LeRobot ACT on all 31 episodes of the
[Rubik’s cube demonstration dataset](https://huggingface.co/datasets/learning-law/pioneer_v1_left_arm_rubiks_cube).
The dataset has 3,143 frames at 25 Hz from the recording-era 6-DoF left arm
and gripper, with ego and left wrist RGB images.

Camera-guided retargeting estimates the cube’s starting position and yaw
from the ego image. It shifts the wrist trajectory from ACT’s reference
start to the estimated cube pose, solves for joint targets, and reuses ACT’s
gripper commands. Exact-pose retargeting uses the simulator’s cube pose for
comparison.

## Held-out simulator results

Each row uses 100 matched starting poses for all three controllers, with
starting arm joint offsets of up to ±3°. A successful rollout lifts the cube
at least 5 cm, then places it within 5 mm of its original height without
robot contact for about 0.4 seconds. The rollout lasts up to eight seconds
at 25 Hz and allows any placement location.

| Cube variation | ACT | Camera-guided retargeting | Exact-pose retargeting |
| --- | ---: | ---: | ---: |
| ±2 cm per axis, ±10° yaw | 40/100 | 98/100 | 100/100 |
| ±6 cm per axis, ±30° yaw | 5/100 | 67/100 | 64/100 |

The camera estimator was fit on 800 synthetic images (seeds 51000–51799)
and selected on 100 validation images (53000–53099). Both splits mix the
original and wider pose ranges. The two test suites use separate seeds
(55000–55099 and 56000–56099). The test keeps the original camera mount,
lighting, sticker colors, and resting height. Median XY error is 0.75 mm
in the original range and 1.16 mm in the wider range. These results are
from simulation.

[Fit report](../../docs/evaluations/rubiks_camera_pose/fit.json) ·
[Per-trial test report](../../docs/evaluations/rubiks_camera_pose/test.json)

## Train on WATcloud

From the repository root, submit a 10-step smoke run to set up the
standalone environment, then run the baseline:

```bash
SETUP_ENV=1 STEPS=10 SAVE_FREQ=10 sbatch scripts/training/train_pioneer_rubiks_cube_act.slurm
SAVE_FREQ=2000 sbatch scripts/training/train_pioneer_rubiks_cube_act.slurm
```

The smoke run was verified on an RTX 3090 in job `760320`. The baseline
uses one RTX 3090, eight CPUs, 32 GB RAM, 10,000 steps, batch size eight,
and seed 1000. The setup pins LeRobot 0.4.3, PyTorch 2.7.1 with CUDA 12.6,
torchvision 0.22.1, and TorchCodec 0.5. Video uses PyAV. Training disables
Hub and W&B uploads.

`TRAIN_STORAGE` defaults to `/mnt/wato-drive2/$USER/pioneer-training`. Set
it consistently for setup, training, and evaluation. The setup records
resolved versions in `requirements-resolved.txt`. Training writes the
run to `<TRAIN_STORAGE>/runs/act-<job ID>`.

## Checkpoint and simulator

The results use the 10,000-step checkpoint from job `760323`, originally
saved at `<TRAIN_STORAGE>/runs/act-760323/checkpoints/010000/pretrained_model`.
Use the whole LeRobot `pretrained_model` directory, including the config,
weights, and pre/postprocessor files. Keep the checkpoint under ignored
`outputs/` when working locally. The test report records the SHA-256 of
`model.safetensors`. The frozen camera estimator is checked in at
[models/rubiks_camera_pose/ego-pose.npz](../../models/rubiks_camera_pose/ego-pose.npz).

From the repository root, fetch the pinned recording-era MuJoCo revision
from the upstream GitHub remote. It provides the scene, arm, and camera
assets used to record the demonstrations.

```bash
rubiks_sim_revision=66b5961d07b0c0706441f3f6ce5952d3fc1862f5
git fetch --no-tags origin "$rubiks_sim_revision"
```

On WATcloud, evaluate the fixed reference pose or a seeded set of varied
starts. Set `rubiks_policy` to the full path to the checkpoint directory.

```bash
sbatch scripts/training/eval_watcloud_rubiks.slurm "$rubiks_policy"
POSE_EPISODES=40 POSE_SEED=2026 \
  sbatch scripts/training/eval_watcloud_rubiks.slurm "$rubiks_policy"
```

The fixed-pose job writes a rollout video. The varied-pose job writes
per-trial results under `<TRAIN_STORAGE>/eval/rubiks-<job ID>`. The fixed
reference is excluded from the varied-pose rate. Job `760337` scored
15/40 with the 10,000-step ACT checkpoint (95% Wilson interval 24.2% to
53.0%). On the same starts, replanning after 25 or 50 actions scored
14/40 and 15/40, respectively. Temporal ensembling scored 4/40
(coefficient 0.01). The evaluation job accepts `POSE_ACTION_STEPS`,
`POSE_ENSEMBLE_COEFF`, `POSE_PERTURB`, and `POSE_INCLUDE_BASELINE` for
these comparisons.

## Reproduce the camera pose benchmark

Activate a Python environment with LeRobot 0.4.3, MuJoCo 3.17.0, and
Transformers 4.57.1. For headless Linux with EGL, set `MUJOCO_GL=egl`.
From the repository root, use the pinned revision fetched above:

```bash
rubiks_source="$(mktemp -d)"
git archive "$rubiks_sim_revision" src/robot_learning src/pioneer_humanoid \
  src/simulation/mujoco_scenes assets/pioneer_bimanual_arm/meshes \
  assets/pioneer_bimanual_arm/urdf assets/props/lightbox_mujoco | tar -x -C "$rubiks_source"
export PYTHONPATH="$rubiks_source/src/robot_learning:$rubiks_source/src/pioneer_humanoid:$rubiks_source/src/simulation/mujoco_scenes"

rubiks_camera_run=outputs/eval/rubiks-camera-pose
mkdir -p "$rubiks_camera_run"
cp models/rubiks_camera_pose/ego-pose.npz "$rubiks_camera_run/"
python scripts/training/train_rubiks_camera_pose.py evaluate \
  --output "$rubiks_camera_run" --policy "$rubiks_policy"
```

To refit the estimator, run the same script with
`fit --output "$rubiks_camera_run"` in place of the copy step. Then run
`evaluate` to write `test.json` in that directory.

## Earlier experiments

Exact-pose retargeting was developed against the simulator before adding the
camera estimator. On a separate 100-pose holdout (29000–29099), it scored
100/100 versus 39/100 for ACT. To reproduce it in the same environment:

```bash
python scripts/training/eval_rubiks_retarget.py --policy "$rubiks_policy" \
  --seed 29000 --episodes 100 --compare-act \
  --output outputs/eval/rubiks-retarget-act-760323/holdout.json
```

A PPO residual pilot froze ACT and learned corrections using simulator pose
and velocity. Job `761116` selected its 24,576-step checkpoint on 12
development poses. It tied ACT at 17/40 on the original benchmark and
scored 20/40 versus 19/40 for ACT on fresh starts. Submit
`scripts/training/train_watcloud_rubiks_residual.slurm "$rubiks_policy"` with
`sbatch` to repeat the pilot. Its checkpoint and report are generated under
`<TRAIN_STORAGE>/runs/rubiks-residual-<job ID>`.
