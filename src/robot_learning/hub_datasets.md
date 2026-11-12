# Datasets on the Hugging Face Hub

Datasets are not committed to git (`datasets/` is gitignored). They live on the Hub; this file says where and how
each one was made.

## `zabsy/pioneer_8dof_peg_insert_synthetic` (private)

Synthetic peg-insertion demos, 8dof left arm, MuJoCo `peg_insert` scene (2 mm clearance a side), made by the
simulated operator in `src/teleop/pioneer_leader_arm_teleop/synthetic/`. Base demos for the RL-token (RLT) experiments.

- Format: LeRobot v3, 25 fps, `pioneer_8dof_dataset_schema.yaml` (8-value state and action, `ego` and `wrist_left`
  cameras at 640x480). `meta/synthetic_takes.jsonl` logs each take's seed and operator style.
- Seeds: part `a` from 0, part `b` from 100000. Evaluate on seeds neither uses (e.g. `evaluate.py --seed 1000`).
- Only successful takes are kept (about 45% of seeds succeed with the scripted operator).

Generated with (one per part, in parallel; a rerun with the same `--dataset_root` resumes):

```bash
MUJOCO_GL=egl python synthetic/synthetic_teleop.py --scene peg_insert --record --num_episodes 100 \
  --seed 0 --dataset_root <out>/peg_ds          # part b: --seed 100000 --dataset_root <out>/peg_ds_b
humanoid-merge-takes <out>/peg_ds <out>/peg_ds_b --output <out>/merged --push_to_hub zabsy/pioneer_8dof_peg_insert_synthetic
```

Rendering is the slow part: on a CPU-only box (software OpenGL) a take costs about 10 minutes, mostly the shadow
passes of the workcell's 7 lights; on a GPU it is seconds.

Load it:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("zabsy/pioneer_8dof_peg_insert_synthetic")
```
