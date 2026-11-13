# HG-DAgger

Run a policy, take over when a correction is needed, accept useful human
segments, aggregate them with the original demonstrations, and refine the
policy with supervised learning. Measure unassisted success before and after
using the same held-out seeds, timeout and success definition.

The pipeline does not select a scene, robot, task or input device. Your scene
adapter supplies observations, action execution, reset and success; your teleop
adapter supplies human actions and authority signals. The core has no simulator,
VR, robot asset or fixed joint-count dependency. No task YAML is required.

The original [HG-DAgger paper](https://arxiv.org/abs/1810.02890) uses a driving
ensemble and five collection rounds. This implements the human-gated aggregation
loop with ACT/LeRobot support, without its ensemble doubt estimator.

## Install

Run the commands below from the repository root. For LeRobot 0.4 learning/export
support, use Python 3.10–3.12:

```bash
pip install -e 'src/robot_learning[hg-dagger]'
```

Core collection and correction validation only need this package's base
installation. Simulator/controller dependencies belong to the adapter you select.

## Connect your application

Implement the `Scene` and `Teleop` protocols in [core.py](core.py), exposing:

```python
# In your application's importable package, for example my_project.hg_adapters:
def make_scene(options):
    return YourSceneAdapter(**options)

def make_teleop(scene, options):
    return YourTeleopAdapter(scene, **options)
```

The scene provides `contract`, `reset(seed)`, `observe()`, `execute(action)`,
`success()` and `close()`. The contract records names, units, image shapes, robot,
scene identity and fps. State and action dimensions may differ; neither is fixed.
Images are HWC RGB uint8; state/actions are finite float32 vectors. `execute`
advances exactly one contract tick and returns the executed target after limits.
When the collector holds, physics must not advance in the background.

The teleop provides `poll() -> Signals`, `seed()`, `action()` and `close()`.
`seed()` reanchors human control to the current measured state. Signals report
readiness, takeover state, and single-shot accept/discard/quit events. Map those
to your controller's buttons, keyboard, or other UI in your adapter.

Optional `feedback(status)` and `preview(status)` methods display progress;
preview returns true to quit. Factories own cleanup if initialization fails.
The core resets policy chunks whenever authority changes, input readiness is
lost, or the scene resets. Robot limits and hardware safety belong to adapters.

Choose your factories and task at runtime:

```bash
humanoid-hg-dagger collect \
  --scene-adapter my_project.hg_adapters:make_scene \
  --teleop-adapter my_project.hg_adapters:make_teleop \
  --task 'your task instruction' \
  --policy outputs/bootstrap/checkpoints/last/pretrained_model \
  --corrections datasets/round_001/corrections --seed 1000
```

Optional `--scene-options` and `--teleop-options` accept JSON objects passed
unchanged to your factories. Their contents belong to your application, not the
HG-DAgger schema. The policy must match your data contract, including action
meaning and joint order. `Session` also accepts any custom policy implementing
`reset()` and `action(observation, task)` for use outside the LeRobot CLI.

## Collection decisions

| Signal | Result |
|---|---|
| takeover true | Human controls execution; record human labels |
| takeover false | Policy resumes; human segments stay unsaved |
| accept | Save all pending human segments atomically, then reset |
| discard | Clear pending segments, then reset; overrides simultaneous accept |
| quit / Ctrl+C | Exit without saving pending segments |

Startup and resets wait for readiness and takeover release. Loss of readiness
holds physics, ends the segment and requires release before resuming. At the
1,500 pending-frame limit, physics holds until accept or discard. Nothing
implicitly saves at hand-back, input loss, buffer limit or exit.

Each uninterrupted takeover is a separate episode; policy frames are excluded.
Frames pair pre-action observations with actual limited human targets. One
accept can save multiple segments. The status shows unsaved frames and accepted
segments in the current process. To validate all stored corrections and count
segments/frames across sessions:

```bash
humanoid-hg-dagger validate datasets/round_001/corrections
```

Accepted transactions are atomic `.npz` files with boundaries, task, reset seed
and contract. Resuming appends without overwriting; contract mismatches fail.
Validation checks integrity and finite values, not whether the human correction
achieves task success. Inspect corrections before training. Use one collector
per correction directory.

## Existing data and checkpoints

Bootstrap data and weights must come from the same robot, scene and data
contract. To attach a contract to an existing local dataset/checkpoint:

```bash
humanoid-hg-dagger bind \
  --scene-adapter my_project.hg_adapters:make_scene \
  --dataset datasets/bootstrap \
  --policy outputs/bootstrap/checkpoints/last/pretrained_model
```

`bind` checks saved episodes, fps, joint order, camera shapes and checkpoint
feature shapes. You assert the scene and units; they cannot be inferred from
weights. Different existing contracts are rejected. Include the complete saved
normalization processors with a checkpoint. To bootstrap a new policy, use your
application's normal demonstration recording and training, then bind the result.

## Aggregate, refine, evaluate

Include all original demonstrations and all accepted corrections so far, with
each source directory listed once. Choose a fresh output outside source folders:

```bash
humanoid-hg-dagger aggregate \
  --base datasets/bootstrap \
  --corrections datasets/round_001/corrections \
  --output datasets/round_001/aggregate

python -m humanoid_robot_learning.hg_dagger.train \
  --dataset.repo_id=local/hg_dagger \
  --dataset.root=datasets/round_001/aggregate --dataset.video_backend=pyav \
  --policy.path=outputs/bootstrap/checkpoints/last/pretrained_model \
  --policy.device=cuda --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null \
  --policy.optimizer_lr=0.000005 --policy.optimizer_lr_backbone=0.000005 \
  --steps=5000 --batch_size=4 --num_workers=2 --save_freq=1000 \
  --eval_freq=0 --wandb.enable=false --output_dir=outputs/round_001
```

These training flags are ACT examples, not paper hyperparameters or a quality
guarantee. Training warm-starts the checkpoint with a new optimizer, preserves
saved normalization and annotates saved checkpoints with the contract. It does
not overwrite the parent or upload artifacts. Aggregation re-encodes each source
frame once to avoid append/merge video-index problems. Never include a previous
aggregate alongside the original inputs it already contains.

Evaluate before and after each round, using the same arguments except policy
and report paths:

```bash
humanoid-hg-dagger evaluate \
  --scene-adapter my_project.hg_adapters:make_scene \
  --task 'your task instruction' \
  --policy outputs/round_001/checkpoints/last/pretrained_model \
  --seed 81000 --episodes 50 --seconds 30 --output outputs/after.json
```

Evaluation runs without teleoperation and requires a real `success()` predicate.
Each rollout ends at success or timeout. Freeze the configuration, seeds and
predicate for comparison, and keep test seeds separate from collection seeds.
Use the resulting checkpoint for the next collection round. Keep weights and
data under ignored `outputs/` and `datasets/`, or your external artifact store.

## Optional MuJoCo/Quest integration

This adapter reuses the Pioneer model, scene registry and Quest driver already
on `main`. It adds no robot model, VR transport/UI, or task assets. Install its
dependencies only if you use it:

```bash
pip install -e 'src/robot_learning[hg-dagger,hg-dagger-mujoco]'
```

Use the factory paths
`humanoid_robot_learning.hg_dagger.adapters.mujoco_quest:make_scene` and
`humanoid_robot_learning.hg_dagger.adapters.mujoco_quest:make_teleop` in place of
your own factories. You must supply `--scene-options '{"scene":"YOUR_REGISTERED_SCENE"}'`;
there is no default scene. Options also accept `arms`, `fps`, `cameras` and
`headless`, matched to your bootstrap data. It uses the upstream seven-joint arms
plus normalized gripper closure (8 dimensions per arm). The integration requires
an editable source checkout; the core pipeline does not require this model.

Forward the port with `adb reverse tcp:8444 tcp:8444`, then open
`http://localhost:8444/` in Meta Browser and enter VR. Right grip is takeover,
release returns to policy, right B accepts/resets, and left Y discards/resets.
Triggers control the selected grippers. Camera views appear on the desktop;
the upstream Quest page supplies tracking. Set `headless: true` and configure
MuJoCo's rendering backend (for example `MUJOCO_GL=egl`) for no desktop window.
No Isaac container is required. Its evaluator uses the selected registry scene's
success/progress hook and refuses to invent success metrics if none exists.
