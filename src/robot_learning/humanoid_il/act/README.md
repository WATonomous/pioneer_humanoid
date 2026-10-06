# ACT for `tidy_table` (MuJoCo sim)

Record demonstrations, train [ACT](https://huggingface.co/docs/lerobot/act) with LeRobot, and score the policy
closed-loop in the same scene. Sim only; nothing here talks to the real arm.

| file | what it does |
|------|--------------|
| `tidy_sim.py` | The scene with the leader teleop's observation / action contract (below). Used by the demo generator and the evaluator, so training data and evaluation can't drift apart. |
| `scripted_demos.py` | Scripted demonstrations in the teleop's LeRobot format; keeps only episodes the scene scores a success. |
| `train_act.sh` | `lerobot-train` with ACT defaults for this task. |
| `eval_sim.py` | Runs a checkpoint in `tidy_table` on unseen layouts and prints success / tidiness per episode. |

## 1. Record

Leader arm (each take: `S` start, `N` save, `D` discard; end every take back at home with the gripper open):

```bash
python src/teleop/pioneer_leader_arm_teleop/pioneer_leader_arm_teleop.py \
    --target mujoco --scene tidy_table --record --cameras top,wrist_left
```

or the keyboard (no leader arm needed; keep the terminal focused: `P` start, `N` save, `B` discard, `H` home):

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
| `observation.environment_state` | which object is next: one-hot colour (8) + shape (3), `done` | yes |
| `action` | 6 joint targets (rad) + gripper command 0..1 | target |
| `leader_angles`, `leader_counts` | leader encoders, servos A..G: rad (calibrated) and raw counts; NaN in scripted demos | no |
| `subtask_index`, `task` | the step index and its instruction ("put the red ball in the ball bin") | no |

Look inside a dataset, or export every value to CSV (one file per episode):

```bash
python -m humanoid_robot_learning.export_dataset <dataset root>          # from src/robot_learning
```

## 2. Train

```bash
src/robot_learning/humanoid_il/act/train_act.sh <dataset root> outputs/act_tidy
```

Chunks of 50 actions (2 s at 25 fps), batch 8, 100k steps, checkpoints every 10k. Pass any other
`lerobot-train` flag after the two paths (e.g. `--steps=20000`, `--policy.device=cpu`). With no access to
download.pytorch.org (the ImageNet ResNet18), add `--policy.pretrained_backbone_weights=null` -- it learns
slower from scratch. The dataset folder is read in place; nothing is pushed to the Hub.

## 3. Evaluate

```bash
cd src/robot_learning/humanoid_il/act
MUJOCO_GL=egl python eval_sim.py outputs/act_tidy/checkpoints/last/pretrained_model --episodes 20 --video eval.mp4
```

Seeds start at 100000, layouts no demo was recorded on. An episode ends when every object is binned and the
arm is home, or after 90 s; it is scored with `tidy_table`'s checks (in order, nothing dropped or toppled).

## Running from a checkout

The scripts put the repo's `pioneer_humanoid`, `humanoid_mujoco_scenes` and `humanoid_robot_learning` on
`sys.path` themselves. They need `mujoco`, `lerobot>=0.4` and `torch` (`pip install -e src/robot_learning[sim]`).
