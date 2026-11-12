# Quest controller HG-DAgger for the trained NASA cargo policy

For the current fixed-cargo, moving-tray policy, use the commands in
[the moving-tray HG-DAgger guide](MOVING_TRAY.md). The USB/browser setup and
controls below apply; its collection command selects the current checkpoint.

For more visible position variation within checked arm reach, use
[`--randomization wide`](WIDE_SPAWNS.md).

For varied cargo colours, spawn positions and arm starting poses, use
`--randomization varied` and the updated checkpoint in
[the quick randomized-update guide](RANDOMIZED_QUICK_UPDATE.md).

The `vr-teleop-v2` controller stack is imported under
`src/teleop/quest_controller_teleop` (source commit `4d0fce67`). The cargo runner
uses its WebXR transport, relative controller mapping, wrist calibration and
URDF IK through `cargo_vr.py`. It keeps the trained left-arm policy, receiving
tray, 100 g cargo bag, two 320×240 RGB observations, 25 Hz and seven-action
contract. The right arm stays parked.

Use the cargo entry point below for HG-DAgger. The standalone VR launcher in
the imported directory has its own scenes and recording configuration.

## 1. Host: container and USB connection

Connect the Quest over USB, enable developer mode/USB debugging and authorize
the computer in the headset. From a host terminal:

```bash
cd /home/rwahib/wato/humanoid
ACTIVE_MODULES="simulation_isaac" ./watod up -d
adb devices
adb reverse tcp:8444 tcp:8444
./watod -t simulation_isaac
```

`adb devices` should list the headset as `device`, rather than `unauthorized`.
The simulation compose service uses host networking, so USB forwarding reaches
its local server. Reapply `adb reverse` after reconnecting the USB cable.

## 2. Container: dependencies and collection

Inside that container, install the small VR dependency set once if needed:

```bash
cd /workspace/humanoid
/workspace/isaaclab/_isaac_sim/python.sh -m pip install \
  -r src/teleop/quest_controller_teleop/requirements-isaac.txt
```

Start the trained policy with VR human interventions:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode collect --teleop vr --device cuda --single-task-thread \
  --policy outputs/train/nasa_cargo_act_v1/checkpoints/100000/pretrained_model \
  --dataset-root datasets/nasa_cargo/round_001/vr_corrections \
  --episodes 50 --seed 2000 --round-id round_001
```

This starts the cargo scene and its embedded HTTP/WebSocket server on port
8444. The trained policy runs autonomously until you intervene. Close a previous
Isaac viewer before starting this command. The task-worker flag carries over
the VR branch's startup workaround; its native-runtime behavior has not been
retested for this integration.

## 3. Headset and controls

Open **`http://localhost:8444/` in Meta Browser**, refresh, then select **Enter
VR**. Start with both grip buttons released, so button state can synchronize.
The page/HUD identifies `POLICY`, `HUMAN` and pause/tracking-wait states.

| Input | Cargo HG-DAgger behavior |
|---|---|
| Hold **right grip button** (middle finger) | Take over from the policy |
| Release right grip | Save this human segment, clear stale policy actions and return to the policy |
| Move/rotate **left controller** | Move the left arm while you have human authority |
| **Left index trigger** | Proportional gripper closure after matching the current held grip |
| Hold **X** | Fine translation/rotation at 25% increments |
| Hold left thumbstick click, rotate controller, release | Pause arm input and fit wrist-pivot calibration |
| **Left Y** | Reset arm/bag for the next seed; discard the current unsaved segment |
| GUI keyboard **H** | Alternative takeover/return toggle |
| GUI keyboard **P** | Freeze/resume physics |
| GUI keyboard **N** / **Delete** | Save / discard the current human segment |
| GUI keyboard **B** | Reset/discard for the next attempt |
| GUI **Esc**, or terminal **Ctrl+C** | Exit; discard unsaved frames |

Taking over reanchors controller movement at the measured arm pose. If the
policy is carrying a bag, its gripper command is retained: squeeze the left
trigger to match the held closure before releasing it to open. A relaxed trigger
at takeover therefore does not immediately drop the bag.

Human control requires both controllers and head tracking. During human
control, tracking loss, stale input or left pivot calibration holds
physics and recording. Tracking return reanchors movement. A reconnect does
not silently hand control back to the policy; if needed, press/release the right
grip again or use keyboard H. Policy mode can keep running while the headset is
disconnected.

The headset shows the **fixed policy overview as a monoscopic view**, plus the
left wrist view (both wrist panels show that same left camera). This preserves
the recorded policy camera contract; head movement does not move those cameras.
Default translation gain is 1.0; use `--vr-gain 0.5` for smaller movements.

Each intervention is a separate episode, split at 750 frames if necessary.
Only human-controlled executed actions become correction labels. Policy-only
execution is excluded. `--episodes 50` counts saved segments, not complete
pick-and-place attempts. For a completed policy attempt, use Left Y/B to reset.
Existing output directories require `--resume`; choose a new round directory
for a new collection round.

For VR-only demonstrations using this same data contract, use `--mode teleop
--teleop vr`, omit `--policy`, and choose a new dataset directory. That mode
starts in HUMAN authority; pressing/releasing the right grip saves a segment
and keeps human control. Append `--headless` for headset-only operation; GUI
keyboard shortcuts then require switching back to a windowed run.

## 4. Aggregate and train v2

Stop the VR collector first, then inside the container:

```bash
export PYTHONPATH="/workspace/humanoid/src/robot_learning:${PYTHONPATH:-}"
/workspace/isaaclab/isaaclab.sh -p -m humanoid_robot_learning.sim.cargo_data \
  validate datasets/nasa_cargo/round_001/vr_corrections

/workspace/isaaclab/isaaclab.sh -p -m humanoid_robot_learning.sim.cargo_data merge \
  datasets/nasa_cargo/synthetic_50 \
  datasets/nasa_cargo/round_001/vr_corrections \
  --output datasets/nasa_cargo/round_001/aggregate

CARGO_DATASET_ROOT=/workspace/humanoid/datasets/nasa_cargo/round_001/aggregate \
CARGO_OUTPUT_DIR=/workspace/humanoid/outputs/train/nasa_cargo_act_v2 \
bash src/robot_learning/scripts/cargo_train_overnight.sh
```

Then evaluate v2 autonomously as described in [the pipeline guide](HG_DAGGER.md),
and use its checkpoint for the next correction round. The existing v1 checkpoint
and synthetic dataset remain available.

## Integration validation

Unit checks run without launching a simulator, headset session or server:
12 cargo/gate checks passed (one optional PyArrow check skipped), and 38 imported
VR checks passed (four optional browser/WebSocket checks skipped). New checks
exercise right-grip authority edges, takeover while carrying, trigger matching,
stale/missing tracking, reconnect behavior, camera reuse and label admission.
Runtime Quest/Isaac operation has not been tested for this integration. The v1
policy's earlier headless performance was measured before adding VR.
