# Adapters for running the papers' own code on our MuJoCo scenes

What is here is only our side: dataset / environment / config files added to the authors' repositories, and the
scripts that connect them to our scenes. The authors' code, their environments and all data stay out of git.
Results and conclusions are in `docs/research/synthetic_teleop_policy_learning.md`.

## Layout and setup

The scripts were run from `outputs/third_party/pioneer_adapter/` (git-ignored), next to clones of the two repositories:

```
outputs/third_party/
  patch_policy/            git clone https://github.com/gaoyuezhou/patch_policy        (at 55ccb4f)
  gam/                     git clone https://github.com/GehaoZhang6/Grounded-Action-Model   (at 46549ce)
  pioneer_adapter/         <- scripts/ from here
  venv-patchpolicy/        Python 3.9: pip install -r patch_policy/requirements_from_conda_env.txt
                           (their conda_env.yml's pip list) + setuptools==70.0.0 (listed under conda there)
  venv-gam/                Python 3.11: their scripts/install.sh, plus mujoco==3.15.0 for our scenes
```

Copy `patch_policy/{datasets,envs,configs}/*` and `gam/configs/*` from here into the clones. The yaml files contain
absolute paths to `/home/wilson/pioneer_humanoid_synth`; change them for another checkout.

## Patch Policy (arXiv 2607.18236)

- `scripts/convert_lerobot.py`: a LeRobot demo dataset -> per-episode memory-mapped arrays (2 views, 224 x 224).
- `patch_policy/datasets/pioneer.py`: their `datasets/your_dataset.py` hook for those arrays (frames read lazily).
- `patch_policy/envs/pioneer_dummy.py`: a placeholder env for their trainer (their `envs/dummy_env.py` imports a helper
  the release does not include). The policy is rolled out in MuJoCo separately.
- `patch_policy/configs/train_pioneer_peg.yaml`: their LIBERO-Goal two-camera 1-GPU recipe, no goal;
  `train_pioneer_peg_chunk10.yaml`: the same with 10-step action chunks (their real-robot setting).
- `scripts/rollout.py`: loads a snapshot of their model and drives our scene with the control loop of their
  `eval_on_env` (including their chunk averaging); scored like `sim_policy_eval.py`.

Outcome on peg_insert (180 demos, layouts 1000+): 0/50 at every evaluated epoch (5, 11 with one-step actions, which
freeze at the start pose; 6, 10, 20 and 0/46 at 30 with 10-step chunks). Their Push-T config on their data reached
final coverage 0.06 at epoch 10 of 400 (the paper reports 0.69 at 400; 400 epochs are ~100 h on a 16 GB card).

## Grounded Action Model (arXiv 2609.23863)

- `gam/configs/pioneer_peg.yaml`: their `lerobot_single_arm.yaml` with the exact sim calibration, the arm URDF,
  `pcd_ref: none` (their `ee` option runs Franka kinematics) and `gripper_dims: [6]`.
- `gam/configs/pioneer_peg_rgbd.yaml`: the same with the simulator's depth as a depth sensor, the objects named
  (`red peg`, `blue block`) and `geometry_source: box2d_depth`. `pioneer_duplo.yaml`: written, not run (their LeRobot
  route takes one instruction per episode; duplo demos have three).
- `scripts/gam_env.py`: our scenes as a `GAM.eval.run` environment (RGB, and depth when the config asks for it).
- `scripts/gam_track_lowmem.py`: their `GAM.tools.track` with CoTracker run on 120-frame windows (their one-shot call
  does not fit in 16 GB).
- `scripts/gam_precompute_rgbd.py`: their `GAM.precompute` with each take replayed from its seed to render depth.
- `scripts/gam_prompt_test.py`, `scripts/gam_geometry_test.py`: grounding accuracy against the true sim poses.

Status: STOPPED BEFORE TRAINING (2026-10-14, by decision, not by a failure). The RGB-D feature cache reached 14 of 50
demos. Measured on 15 frames of 3 peg demos: the span head's "hole" lands on the other arm's gripper (2D error 31 px;
named slots 6 px); monocular 3D (their default) is 24-32 cm off; with sim depth and `box2d_depth` the peg is 3.2 cm and
the block 6.6 cm off (median). CoTracker point tracks drift onto the arm when it covers the hole. Their detector
fine-tuning, which they used for their own simulated benchmark, is not released. With grounding errors of several cm
against a 2 mm clearance, and true object poses already giving no gain over images on this task, the run was not
expected to beat image-only ACT.
