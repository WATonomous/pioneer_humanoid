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

Wilson’s latest compatible input/transport updates are included: 1-euro pose
smoothing, mapping anchored at takeover to avoid accumulated drift, bounded
inter-packet prediction, and active-headset-tab ownership. The cargo headset
camera panels are retained. Upstream’s new seven-joint MuJoCo model and mink
IK are not substituted for this trained six-joint Isaac policy; see
[the import record](../../src/teleop/quest_controller_teleop/IMPORTED_FROM.md).

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
8444. Startup waits with physics and recording held while the browser preview
remains live. Enter VR with head and both controllers tracked and both grip
buttons released; then the trained policy starts automatically and runs until
you intervene. Close a previous
Isaac viewer before starting this command. The task-worker flag carries over
the VR branch's startup workaround; its native-runtime behavior has not been
retested for this integration.

## 3. Headset and controls

Open **`http://localhost:8444/` in Meta Browser**, refresh, then select **Enter
VR**. Start with both grip buttons released, so button state can synchronize.
The page/HUD identifies `POLICY`, `HUMAN` and pause/tracking-wait states.
Until the first ready connection it displays `WAITING FOR VR`; the terminal
prints `VR ready; starting control` when startup is released. This is a
one-time startup wait; subsequent policy-mode disconnect behavior is unchanged.

| Input | Cargo HG-DAgger behavior |
|---|---|
| Hold **right grip button** (middle finger) | Take over from the policy |
| Release right grip | Return to policy; keep corrections unsaved until B or Y |
| Press **right B** | Save all unsaved human correction segments, then reset to the next seed |
| Move/rotate **left controller** | Move the left arm while you have human authority |
| **Left index trigger** | Proportional gripper closure after matching the current held grip |
| Hold **X** | Fine translation/rotation at 25% increments |
| Hold left thumbstick click, rotate controller, release | Pause arm input and fit wrist-pivot calibration |
| **Left Y** | Discard all unsaved corrections and reset; release right grip to start the next attempt |
| GUI keyboard **H** | Alternative takeover/return toggle; does not save in VR mode |
| GUI keyboard **P** | Freeze/resume physics |
| GUI keyboard **N** / **Delete** | Save / discard all unsaved correction segments |
| GUI keyboard **B** | Reset/discard for the next attempt |
| GUI **Esc**, or terminal **Ctrl+C** | Exit; discard all unsaved frames, including released interventions |

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

Each uninterrupted intervention remains a separate episode; policy-only execution
is excluded and action chunks never bridge a policy-controlled interval. Releasing
right grip stages the segment in memory without writing it. Right B explicitly
saves all staged segments plus any active human segment, then resets. Saving
finishes before the reset; the saved segments retain the original attempt seed.
Left Y discards all
unsaved segments for the attempt and resets, including when right grip is held.
After either button, physics waits for right-grip release so reset cannot create
stray frames. B also resets an empty attempt without creating a recording. The
HUD always shows both the total saved segment count and the unsaved frame count.
If B and Y arrive together, Y wins. Existing saved recordings are not deleted.

The VR buffer holds at most `--max-segment-frames` human frames (default 750,
30 simulated seconds of human control). At the limit, physics pauses and the HUD
asks for B to save or Y to discard; it never automatically saves a failed attempt.
`--episodes 50` counts saved intervention segments, not full tasks or B presses.
One B press can save multiple segments if you intervened more than once.
Existing output directories require `--resume`; choose a new round directory
for a new collection round.

For VR-only demonstrations using this same data contract, use `--mode teleop
--teleop vr`, omit `--policy`, and choose a new dataset directory. That mode
starts in HUMAN authority; right B saves/resets, left Y discards/resets, and releasing
right grip does not save. Append `--headless` for headset-only operation; GUI
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

The cargo suite covers explicit-save and reset behavior in the container. Regression checks cover
right-grip authority without automatic saves, explicit B saving (including
held-button suppression), Y discard/reset, simultaneous B+Y with Y priority,
post-reset grip-release waiting, bounded pending buffers, and exclusion of
policy actions. The actual collection loop is exercised without an Isaac
launcher. The imported VR stack also has protocol, smoothing, target-mapping
and prediction tests; optional browser/asset tests depend on installed tools.
A live headless check verified the waiting server and headset camera endpoint.
Runtime Quest/controller feel still requires the operator's headset check.


## If right B appears to do nothing

Restart the collector with `--resume` if its output directory already exists.
Exit immersive VR, reload `http://localhost:8444/` in Meta Browser, and enter VR
again so the headset runs the current page. An old page can still stream motion
and grips while omitting the new B field; the HUD now explicitly requests a
page refresh in that case. Received presses are logged as `RIGHT B received`.
A successful write shows `SAVED ...`; an empty buffer shows `NO NEW CORRECTION
TO SAVE`, together with the unchanged total saved count. B resets in both cases.
The `/status` response includes `right_b_presses`, `unsaved_frames`, and
`vr_client_has_save_button` and `saved_segments` for diagnostics. A brief tracking dropout does not
discard an already-received save-button event; pose commands still require
fresh tracking. Focus the Isaac window and press keyboard N as an alternative
explicit save. Save pending work before exiting the collector.
