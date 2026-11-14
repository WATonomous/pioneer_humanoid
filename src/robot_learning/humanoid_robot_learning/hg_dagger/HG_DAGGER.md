# HG-DAgger

Let a policy run, take over when it makes a mistake, and save useful corrections.
Combine accepted corrections with the original demonstrations, refine the policy,
and repeat. This follows the [HG-DAgger](https://arxiv.org/abs/1810.02890)
aggregation loop using ACT/LeRobot 0.4, without the paper's ensemble doubt estimator.

## Workflow

1. Start with a policy trained for your scene and robot.
2. Intervene when needed; explicitly accept or discard your corrections.
3. Validate and combine **all original demos + all accepted corrections**, counting each source once.
4. Refine the existing checkpoint into a new output directory, preserving normalization.
5. Compare unassisted success before/after on the same held-out seeds and timeout.

## Collect with Quest

From the repo root, using Python 3.10–3.12:

```bash
pip install -e 'src/robot_learning[hg-dagger,hg-dagger-mujoco]'

src/teleop/quest_controller_teleop/start_quest.sh \
  --scene YOUR_SCENE --hg-dagger \
  --policy outputs/bootstrap/checkpoints/last/pretrained_model \
  --corrections datasets/round_001/corrections \
  --task_description 'your task instruction' --device cpu
```

The checkpoint must match the selected scene and current arm. Use `--device cuda`
for GPU inference and `PY=/path/to/python` to select the launcher's environment.

| Control | Result |
|---|---|
| Hold right grip | Human takeover; motion/triggers control the selected arms/grippers |
| Release right grip | Return to policy; corrections remain unsaved |
| Right B / desktop **8** | Save pending corrections and reset |
| Left Y / desktop **6** or **9** | Discard pending corrections and reset |
| Close window / Ctrl+C | Quit without saving pending corrections |

Physics waits for VR readiness and grip release, and holds on tracking loss or a
full buffer. Only human actions become labels; each uninterrupted takeover is a
separate segment. `--record` and `--hg-dagger` are mutually exclusive.

If existing data/checkpoints lack `contract.json`, bind them once before collecting:

```bash
python src/teleop/quest_controller_teleop/mujoco/mujoco_sim.py \
  --scene YOUR_SCENE --hg-dagger \
  --policy outputs/bootstrap/checkpoints/last/pretrained_model \
  --bind-dataset datasets/bootstrap
```

Binding checks names, shapes, cameras and fps; you assert the data's scene and units.
See the [Quest README](../../../teleop/quest_controller_teleop/README.md) for headset setup.

## Prepare the next training round

```bash
python -m humanoid_robot_learning.hg_dagger.storage datasets/round_001/corrections
python -m humanoid_robot_learning.hg_dagger.learning --base datasets/bootstrap \
  --corrections datasets/round_001/corrections --output datasets/round_001/aggregate
```

Validation checks data integrity, not task success. Review corrections before
training. Refine through [train.py](train.py) using the standard
[LeRobot training flags](../../README.md#3-train). Use fresh output directories;
keep weights and datasets under ignored `outputs/` and `datasets/`.

## Use another scene or controller

Implement the `Scene`, `Teleop` and `Policy` interfaces in [core.py](core.py), then
connect them through `Session` and `CorrectionStore`. Applications with an existing
loop use `Session.decide()` and `Session.record()`; `Session.tick()` runs a complete
tick otherwise. The contract keeps joint names, units, cameras and fps compatible.
The core does not depend on a particular robot, simulator or controller.
`core.evaluate()` measures unassisted success using your scene's success predicate.
