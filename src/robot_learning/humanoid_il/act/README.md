# ACT for `tidy_table` (MuJoCo sim)

Record demonstrations, train [ACT](https://huggingface.co/docs/lerobot/act) with LeRobot, and score the policy
closed-loop in the same scene. Sim only; nothing here talks to the real arm.

| file | what it does |
|------|--------------|
| `tidy_sim.py` | The scene with the leader teleop's observation / action contract (below). Used by the demo generator and the evaluator, so training data and evaluation can't drift apart. |
| `scripted_demos.py` | Scripted demonstrations in the teleop's LeRobot format; keeps only episodes the scene scores a success. |
| `train_act.sh` | `lerobot-train` with ACT defaults for this task. |
| `train_diffusion.sh` | `lerobot-train` with Diffusion Policy for this task (needs `lerobot[diffusion]`). |
| `eval_sim.py` | Runs a checkpoint (ACT, Diffusion Policy, any LeRobot type) in `tidy_table` on unseen layouts and prints success / tidiness per episode. |
| `live_view.py` | Runs a checkpoint like `eval_sim.py`, streamed live to a browser page (MJPEG on `--port`, default 8080). |

## 1. Record

Leader arm (each take: `S` start, `N` save, `D` discard; end every take back at home with the gripper open):

```bash
python src/teleop/pioneer_leader_arm_teleop/pioneer_leader_arm_teleop.py \
    --target mujoco --scene tidy_table --record --cameras top,wrist_left
```

or the keyboard (no leader arm needed; keys go to the scene window: a take starts with the first move key and, once
the last object is binned, goes home and saves by itself; `B` discards):

```bash
python src/teleop/keyboard_teleop/mujoco_keyboard_teleop.py --scene tidy_table --record --cameras top,wrist_left
```

and/or scripted demos (headless, `MUJOCO_GL=egl` without a display):

```bash
cd src/robot_learning/humanoid_il/act
MUJOCO_GL=egl python scripted_demos.py --episodes 200 --dataset_root ../../../../datasets/tidy_table_scripted
```

The scripted controller succeeds on ~88% of layouts (failures are discarded). Rendering two 640x480 cameras
needs a GPU to be quick; on a CPU-only machine each episode takes minutes.

Every frame holds:

| feature | content | policy input? |
|---------|---------|---------------|
| `observation.state` | 6 left-arm joints (rad) + finger closure 0..1 | yes |
| `observation.images.top`, `.wrist_left` | 640x480 RGB video | yes |
| `observation.environment_state` | `done`: 1 once every object is binned (go home) | yes |
| `action` | 6 joint targets (rad) + gripper command 0..1 | target |
| `leader_angles`, `leader_counts` | leader encoders, servos A..G: rad (calibrated) and raw counts; NaN in scripted demos | no |
| `subtask_index`, `task` | objects binned so far and the instruction ("put everything in the trash bin") | no |

## 2. Train

```bash
src/robot_learning/humanoid_il/act/train_act.sh <dataset root> outputs/act_tidy
```

Chunks of 50 actions (2 s at 25 fps), batch 8, 100k steps, checkpoints every 10k. Pass any other
`lerobot-train` flag after the two paths (e.g. `--steps=20000`, `--policy.device=cpu`). With no access to
download.pytorch.org (the ImageNet ResNet18), add `--policy.pretrained_backbone_weights=null` -- it learns
slower from scratch. The dataset folder is read in place; nothing is pushed to the Hub.

Diffusion Policy takes the same arguments (needs `lerobot[diffusion]`); it sees 2 frames, predicts 32 actions and
re-plans every 12:

```bash
src/robot_learning/humanoid_il/act/train_diffusion.sh <dataset root> outputs/diffusion_tidy
```

## 3. Evaluate

```bash
cd src/robot_learning/humanoid_il/act
MUJOCO_GL=egl python eval_sim.py <output dir>/checkpoints/last/pretrained_model --episodes 20 --video eval.mp4
```

Any LeRobot checkpoint works; the policy type is read from its `config.json`. Seeds start at 100000, layouts no
demo was recorded on. An episode ends when every object is binned and the arm is home, or after 90 s. Success is
everything in the bin with the arm home; `dropped` / `toppled` are printed as notes only.

To watch a checkpoint act live instead, in a browser at `http://localhost:8080` (forward the port from the node):

```bash
MUJOCO_GL=egl python live_view.py <output dir>/checkpoints/last/pretrained_model --device cuda
```

## Running from a checkout

The scripts put the repo's `pioneer_humanoid`, `humanoid_mujoco_scenes` and `humanoid_robot_learning` on
`sys.path` themselves. They need `mujoco`, `lerobot>=0.4` and `torch` (`pip install -e src/robot_learning[sim]`).
