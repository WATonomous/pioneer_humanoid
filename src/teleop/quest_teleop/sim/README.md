# Quest 2 Bimanual Arm Teleop

Hand-tracking data from a Quest 2 headset drives both arms in Isaac Sim 5.1
in real time. Everything runs inside a **single `simulation_isaac` container** —
no separate teleop container is needed.

```
Quest 2 headset (WebXR hand tracking)
    │  HTTPS port 8443  (adb reverse tunnel)
    ▼
webxr_server.py  ──────────────────────────────┐
                                               │ self-signed cert
    │  WSS port 9090  (adb reverse tunnel)     │  (must be trusted on Quest
    ▼                                          │   for BOTH ports 8443 + 9090)
quest_teleop_node  (ROS 2 WebSocket bridge)    │
    │  /quest_teleop  (ROS 2 topic)            │
    ▼                                          │
run_quest_bimanual_teleop.py                   │
    │  DifferentialIKController (DLS) → joint  │
    │  cmds, per arm, fingertip-tip tracked     │
    ▼                                          │
Isaac Sim 5.1  (rendered on host monitor) ◄────┘
```

All four processes run inside `simulation_isaac`. The Quest browser connects
back to the host via `adb reverse` USB tunnels — no Wi-Fi or external network
needed.

---

## Architecture notes

### Why this is a single container now

The old architecture used two containers:
- `teleop` — ran `quest_teleop_node` + `webxr_server` + a UDP relay bridge
- `simulation_isaac` — received hand data over UDP sockets, had no ROS 2

The UDP bridge was only needed because `simulation_isaac` had no ROS 2. We
eliminated it by building ROS 2 Humble from source inside `simulation_isaac` and
having `run_quest_bimanual_teleop.py` subscribe to `/quest_teleop` directly.

### Why ROS 2 is built from source (not apt)

Isaac Sim 5.1 bundles its own Python 3.11. The apt ROS 2 packages (`rclpy`,
etc.) are compiled against Python 3.10 and **will not load** in Isaac Sim's
interpreter. Building from source against deadsnakes Python 3.11 produces
`cpython-311` ABI-tagged extensions that load in both the system Python 3.11
and Isaac Sim's bundled Python 3.11.

### The two Python 3.11 installs

There are intentionally **two** separate Python 3.11 installations in the
final image:

| Python | Path | Purpose |
|--------|------|---------|
| Isaac Sim bundled | `/workspace/isaaclab/_isaac_sim/kit/python/bin/python3.11` | Runs Isaac Sim, IK script, loads rclpy at runtime |
| deadsnakes (apt) | `/usr/bin/python3.11` | Used **only** by `colcon build` — never at runtime |

The deadsnakes install exists purely to satisfy cmake import paths baked into
`std_msgs`/`geometry_msgs` during the ROS 2 build stage. Its numpy version
must be unpinned (latest 2.x) to match the builder stage's numpy, since
numpy ≥ 2.0 uses `numpy/_core/include` and numpy < 2.0 uses `numpy/core/include`
— cmake export files bake this path in at build time.

**Never pip-install ROS 2 build tools into Isaac Sim's own Python (`$PYTHON`).
It has no C headers in its pip_prebundle paths and will cause packaging version
conflicts with lerobot/isaaclab-rl.**

---

## One-time setup

### 1. Build the image

The Dockerfile installs deadsnakes Python 3.11, builds ROS 2 Humble from
source, and bakes `colcon build` into `.bashrc`. Run this once after any
Dockerfile change:

```bash
# From repo root on the host
ACTIVE_MODULES="simulation_isaac" ./watod build
```

Takes 10–20 minutes the first time. Subsequent builds use Docker layer cache
and are much faster unless the ROS 2 build layer changes.

### 2. SSL certificates

The WebXR HTTPS server needs a self-signed cert. Generate once on the host:

```bash
mkdir -p src/teleop/quest_teleop/certs
openssl req -x509 -newkey rsa:4096 -keyout src/teleop/quest_teleop/certs/key.pem \
    -out src/teleop/quest_teleop/certs/cert.pem -days 3650 -nodes \
    -subj "/CN=localhost"
```

The compose file bind-mounts this directory into the container at `/certs`.

### 3. Quest developer mode

The Quest must have **developer mode** enabled in the Meta mobile app so that
`adb` can connect over USB.

### 4. gnirehtet (optional — Quest internet)

gnirehtet lets the Quest use the host's internet connection over USB. Not
required for teleop to work, but useful so the Quest can load pages.

```bash
# Download to ~/gnirehtet if not already there
~/gnirehtet/gnirehtet-rust-linux64/gnirehtet install
```

---

## Running the pipeline

### Check what's already running first

`quest_teleop_node`, `webxr_server.py`, and the Isaac Sim IK script can all
be left running across sessions — don't assume you're starting from a clean
slate. Blindly re-launching a process that's already up fails with
`Address already in use` (for the ROS node / WebXR server) or just wastes
Isaac Sim's ~1-3 minute boot time (for the sim script).

```bash
# From the host — is the container already up?
docker ps --filter "name=simulation_isaac"

# Inside that container — are the ROS node / WebXR server / Isaac Sim
# script already running?
docker exec <container_name> bash -c "ps aux | grep -E 'quest_teleop_node|webxr_server|run_quest_bimanual_teleop' | grep -v grep"

# On the host — are the adb tunnels already set up?
adb devices
adb reverse --list
```

If `quest_teleop_node` / `webxr_server.py` are already running, skip
Terminals 3 and 4 below and go straight to Terminal 2 (adb) and Terminal 5
(Isaac Sim). If Isaac Sim is already running too, you likely don't need to
launch anything at all — just check `adb reverse --list` shows both ports
and put the headset on.

Start the container first, then open the terminals below for whatever
isn't already running:

```bash
ACTIVE_MODULES="simulation_isaac" ./watod up -d
```

### Terminal 1 — Host: Internet sharing (optional)

```bash
~/gnirehtet/gnirehtet-rust-linux64/gnirehtet run
```

Wait for repeating `TcpConnection` log lines. Leave running.

### Terminal 2 — Host: USB port forwarding

Plug in the Quest over USB, then:

```bash
adb reverse tcp:8443 tcp:8443
adb reverse tcp:9090 tcp:9090
```

No output = success. Run `adb devices` first to confirm the Quest is detected.
This terminal is done.

### Terminal 3 — simulation_isaac: ROS 2 hand tracking node

```bash
cd ~/Documents/Wato/humanoid && ./watod -t simulation_isaac
```

`.bashrc` automatically runs `colcon build` when you shell in. Wait for:

```
Summary: 2 packages finished
```

Then source the workspace and run the node:

```bash
source /opt/ros/humble/setup.bash
source /root/ament_ws/install/setup.bash
ros2 run quest_teleop quest_teleop_node
```

Wait for `[INFO] WSS server listening on port 9090`. Leave running.

> **Note:** If `ros2` command is not found, the image predates the ros2cli
> addition. Use the direct binary instead:
> `/root/ament_ws/install/quest_teleop/lib/quest_teleop/quest_teleop_node`
> Then rebuild the image to get `ros2` CLI support.

### Terminal 4 — simulation_isaac: HTTPS WebXR page server

```bash
cd ~/Documents/Wato/humanoid && ./watod -t simulation_isaac
```

Wait for `Summary: 2 packages finished`, then:

```bash
python3 /workspace/humanoid/src/teleop/quest_teleop/bridge/scripts/webxr_server.py
```

Wait for `Serving at https://0.0.0.0:8443`. Leave running.

### Terminal 5 — simulation_isaac: Isaac Sim + IK script

```bash
cd ~/Documents/Wato/humanoid && ./watod -t simulation_isaac
```

Wait for `Summary: 2 packages finished`, then:

```bash
cd /workspace/humanoid/src/teleop/quest_teleop/sim
./run_quest_bimanual_teleop.sh
```

Isaac Sim takes **1–3 minutes** to load. Wait for:

```
[Quest] Ready. Waiting for /quest_teleop messages.
```

before putting on the headset.

### Quest headset

The self-signed cert must be trusted separately for each port the browser
connects to. Missing either step will silently break the WebSocket connection
and `/quest_teleop` will receive no data.

1. Put on the headset
2. Open **Meta Browser** (not the Isaac Sim app)
3. Navigate to `https://localhost:9090` — click **Advanced → Proceed** to trust the WSS cert
4. Navigate to `https://localhost:8443/` — click **Advanced → Proceed** to trust the HTTPS cert
5. Press **Start**

Both arms should begin following your wrists. The Isaac Sim terminal will
print `[Quest] Left wrist tracked — homed at ...` / `[Quest] Right wrist
tracked — homed at ...` as each wrist is first tracked.

---

## Controls

| Action | Effect |
|--------|--------|
| Move right wrist | Right arm follows |
| Move left wrist | Left arm follows |
| Pinch right thumb + index | Right gripper closes/opens |
| Pinch left thumb + index | Left gripper closes/opens |
| `R` (Isaac Sim window focused) | Recalibrate — re-homes both arms to your current wrist pose |

Pass `--gain VALUE` (default `1.0`) to `run_quest_bimanual_teleop.sh` to scale
how far the arm moves per metre of real wrist motion.

The scene always includes a white lightbox enclosure + table around the arm
stand — same geometry as `BimanualPushBlockSceneCfg` in
`humanoid_rl_tasks/push_block/push_env_cfg.py` (plain emissive-white
`CuboidCfg` walls, lit by the scene's existing dome light; no RectLight, no
external USD/asset dependency; no CLI flag needed). This replaced the older
`bimanual_arm_lightbox.usd` stage (Ramy's, removed in
"RL-push-task-switch-to-bimanual-arm") which baked in a RectLight rig,
PerceptionCamera, and wrist-mounted RealSense D455s — none of that is
reproduced here, just the studio-wall + table backdrop.

### Recalibrating mid-session

Homing happens automatically the instant each wrist is first tracked, which
can catch your hand mid-raise before it settles into a comfortable pose —
producing a persistent offset between where your hand feels like it is and
where the arm starts tracking from. Hold your wrists in a comfortable
neutral pose and press **R** (with the Isaac Sim window focused) to re-home
both arms to that pose at any time, including immediately after Start.

---

## Debugging

Check what's publishing on the hand pose topic (requires sourced workspace):

```bash
source /opt/ros/humble/setup.bash
source /root/ament_ws/install/setup.bash
ros2 topic echo /quest_teleop
```

You should see `QuestHandPose` messages at ~30 Hz when the Quest is active.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `colcon build` fails silently on shell entry | Image wasn't rebuilt after Dockerfile changed | `ACTIVE_MODULES="simulation_isaac" ./watod build` |
| `Cannot run the interpreter /usr/bin/python3.11` | Same as above | Rebuild image |
| `ros2: command not found` | Image predates ros2cli addition | Rebuild image, or use direct binary path (see Terminal 3 note) |
| `quest_teleop_node: No such file or directory` | Build didn't finish yet | Wait for `Summary: 2 packages finished` |
| WebSocket closed immediately on Quest | `quest_teleop_node` not running | Start Terminal 3 first |
| Quest browser shows connection refused on port 8443 | `webxr_server.py` not running or `adb reverse` not done | Start Terminal 4 and re-run Terminal 2 |
| Quest browser shows `localhost terminated connection` | SSL issue or server not running | Confirm Terminal 4 is running; check cert files exist at `src/teleop/quest_teleop/certs/` |
| Quest browser shows security warning | Self-signed cert | Expected — click Advanced → Proceed |
| `/quest_teleop` topic empty after pressing Start | WSS cert not trusted for port 9090 | In Quest browser, navigate to `https://localhost:9090` and accept the warning, then return to port 8443 and press Start again |
| Arms don't move after pressing Start | No hand data reaching IK script | Check Terminal 5 prints `[Quest] Left/Right wrist tracked — homed at ...`; check Terminal 3 is running and topic is publishing |
| Isaac Sim numpy broken after container restart | Container was manually patched (not image-built) | Run `./watod build` to make fixes permanent; restart container |
| `ModuleNotFoundError: No module named 'em'` during colcon | `empy` installed into wrong Python (Isaac Sim's, not deadsnakes) | Rebuild image — Dockerfile now uses `/usr/bin/python3.11` explicitly |
| `packaging` version conflict warnings | `catkin_pkg` was pip-installed into Isaac Sim's Python | Fixed in current Dockerfile — rebuild image |
| adb reverse shows no output | Quest not detected | Check USB cable, check developer mode is on, run `adb devices` |
| adb reverse entries disappear | Quest locked or USB disconnected | Unlock Quest, replug USB, re-run Terminal 2 |

---

## Tuning

### Translation scale

`--gain` (default `1.0`) scales how far the IK target moves per metre of real
wrist motion. Increase if the arms feel sluggish to reach positions; decrease
if they overshoot.

### IK tracking speed

Tracking aggressiveness is controlled by the DLS damping `_DLS_LAMBDA` /
`_DLS_LAMBDA_RIGHT` in `run_quest_bimanual_teleop.py`, passed as `lambda_val`
to each arm's weighted-DLS solve. Lower damping converges faster but is less
stable near singular configurations; higher damping is more stable but
converges more slowly.

Current values (`0.1` left, `0.175` right) were lowered from `0.2 / 0.35`
once the canonical arm config started spawning the arm with the elbows flexed —
the higher damping had been compensating for the ill-conditioned
full-extension spawn pose, which no longer occurs. Lower damping also cuts
wrist-twist leakage into the shoulder (see `_IK_JOINT_COST`). Raise both back
toward `0.2 / 0.35` if an arm feels jittery near singularities.

Since the Quest wrist stream only updates at ~30 Hz against a 100 Hz solve
loop, low damping directly translates hand-tracking noise into visible
jitter — retune live in sim rather than assuming this number is final.

**If the arm undershoots specifically on upward reach** (tracks fine
horizontally but doesn't lift as far as the real wrist moves), check
`effort_limit_sim` on the shoulder/elbow `ImplicitActuatorCfg` blocks in
`pioneer_humanoid.bimanual_arm` before touching IK gains — lifting the arm against
gravity at extension needs more torque than any other direction, and a cap
set to the motor's *rated* torque (not *peak*) will saturate there first and
visibly lag the commanded target. Shoulder/elbow are currently set to peak
(53 Nm / 22 Nm); if it's still undershooting, the next suspect is an actual
kinematic reach limit.

Actuator responsiveness (how fast joints track the IK output) is set in
`src/pioneer_humanoid/pioneer_humanoid/bimanual_arm.py`
via `stiffness` and `damping` on each `ImplicitActuatorCfg`. Current values
are raised ~25–50% above the original static-hold tuning for faster tracking
of a moving IK target; the torque caps stay at the real motor peak.

### Coordinate frame mapping

Both position and orientation are now mapped camera-relative, through
`_QUEST_TO_CAM_LOCAL` / `_CAM_LOCAL_*` and the runtime `camera_tilt_quat`, so
sideways hand motion stays sideways regardless of head tilt. `_QUEST_TO_WORLD`
is retired — kept only because `quest_to_world_quat` is still derived from it.

`_AXIS_SIGN_LEFT` / `_AXIS_SIGN_RIGHT` are per-arm, per-axis sign vectors
`[sx, sy, sz]` applied to the position delta. Use them to flip an axis per
arm if it feels backwards (currently `[1, 1, 1]` for both). Don't negate a
row of the mapping matrix — that flips its determinant and breaks orientation.

### Wrist orientation alignment

**Observed issue:** rotating the wrist in one physical axis drove the EE in a
different axis in sim (~30° misalignment on the right arm, ~90° on the left
arm in testing).

**Root cause:** the Quest wrist tracking frame and the robot's link6/link6l
URDF frame have different axis conventions. After the Quest→World→Base frame
remap, the rotation axes do not 1:1 correspond between the physical wrist and
the EE.

**Current knob:** `_WRIST_ORIENT_OFFSET_LEFT` / `_WRIST_ORIENT_OFFSET_RIGHT`
(in the constants block near the top of `run_quest_bimanual_teleop.py`) are
independent `(w, x, y, z)` quaternions, one per arm, applied as a conjugation
on the world-frame orientation delta before converting to base frame. They're
separate because the measured misalignment differs per arm (~30° right, ~90°
left) — a single shared offset can't correct both simultaneously. Identity
`[1, 0, 0, 0]` = no correction. Both are identity now: armWithStand's joint6l
rotates about a different axis than the old asset's, so the old non-identity
values don't transfer and need live in-headset retuning. Common trial values:

| Value | Meaning |
|-------|---------|
| `[1, 0, 0, 0]` | No correction (identity) |
| `[0, 0, 0, 1]` | 180° about world Z |
| `[0.707, 0, 0, 0.707]` | 90° about world Z |
| `[0.707, 0, 0.707, 0]` | 90° about world Y |
| `[0.707, 0.707, 0, 0]` | 90° about world X |

**Automatic calibration (removed 2026-07-18):** a per-arm auto-calibration
was added that recomputed `wrist_orient_offset` at homing time from
`ee_quat_w * q_home_w^-1` (EE's world orientation at rest vs. the Quest
wrist's world orientation at the homing instant), applied as a conjugation
on every subsequent orientation delta. Verified in sim (headless DLS
convergence test, both arms, synthetic wrist rotations) that this formula is
invalid: `ee_quat_w` and `q_home_w` are two orientations with no physical
relationship (an arbitrary fixed robot rest pose vs. whatever way the
operator's wrist happened to be pointing at homing), so the "offset" it
computes is a large, essentially arbitrary rotation. Conjugating a delta by
it preserves the delta's rotation *angle* but not its *axis* — so it
converged to a stable but ~50-60° **wrong** orientation (confirmed via a
400-step/4s trend, not a convergence-speed issue), and destabilized tip
position noticeably worse on the right arm (~0.11m residual vs. ~0.06m left,
for an identical commanded rotation) while doing so. That right/left skew is
what read as "the right arm doesn't track as well" — it wasn't an actuator
or Jacobian bug (those tested symmetric/fine), it was this calibration
feeding both arms a bad target and the right arm being less stable against
it. Removed; `wrist_orient_offset` is now just the static
`_WRIST_ORIENT_OFFSET_LEFT`/`_RIGHT` constant from the table above (identity
by default — re-tune those manually if rotation axes still feel misaligned,
same as before this feature existed). A mathematically sound auto-cal would
need multiple independent rotation samples to solve for the actual
sensor/body axis-convention correction, not a single home-pose snapshot.

---

## Real hardware bridge (left arm only)

`run_quest_bimanual_teleop.py --publish-real-left-arm` publishes the LEFT
arm's live DLS-solved joint targets to `/arm/joint_targets` (ROS 2), so
`joint_command_node` drives the real physical left arm over CAN in lockstep
with your Quest hand tracking. **Off by default** — normal runs are sim-only
and completely unaffected by this flag.

### Right arm is NOT supported yet

`src/interfacing/joint_command/config/arm_calibration.yaml` only has a
`left:` section — there are no CAN IDs documented anywhere in this repo for
the right arm's motors, and `joint_command_node` is single-arm per instance
(`arm_side` param, currently hardcoded to `"left"` in
`joint_command.yaml`/`joint_command.launch.py`), not something that routes
both arms off one topic. `--publish-real-left-arm` only ever builds and
sends the left arm's `ArmPose`; there is no equivalent right-arm flag because
there's nothing on the other end to receive it yet. Adding it requires:

1. The right arm's actual motor CAN IDs (physical/hardware knowledge, not
   something to guess — sending a position command to the wrong physical
   motor ID is exactly the kind of mistake that damages hardware).
2. A `right:` section in `arm_calibration.yaml` with those IDs.
3. A second `joint_command_node` instance (`arm_side: "right"`, its own
   `input_topic`/`motor_cmd_topic` params so it doesn't collide with the left
   instance) and the matching publish call in the teleop script.

### Message mapping

Field layout and units are copied exactly from the existing sim-to-real
precedent, `src/teleop/task_space_controller/task_space_ik.py`
(`publish_joint_pos`) — same `LEFT_ARM_JOINTS` order
(`joint1L, joint2l, joint3l, joint4l, joint5l, joint6l`) maps 1:1 to:

| ArmPose field | Source joint | Meaning |
|---|---|---|
| `shoulder.position[0]` | `joint1L` | flexion |
| `shoulder.position[1]` | `joint2l` | abduction |
| `shoulder.position[2]` | `joint3l` | rotation |
| `elbow.position[0]` | `joint4l` | flexion |
| `elbow.position[1]` | `joint5l` | forearm rotation |
| `wrist.position[0]` | `joint6l` | extension |

`joint_pos_des` from the DLS solver is radians (Isaac Lab convention);
`arm_calibration.yaml`'s limits and the CAN `PositionDeg` signal expect
degrees, so `_publish_real_left_arm_pose` converts before publishing.

### Safety: the 5-second startup delay

`joint_command_node` seeds its rate-limiter from live motor feedback on the
first `ArmPose` and rate-limits every message including that one (position
clamp → velocity limit → delta limit → low-pass), so the arm ramps from its
actual pose rather than snapping. `_REAL_ARM_PUBLISH_START_DELAY_S = 5.0`
holds off publishing for 5s after the flag is enabled — **operator prep time:
get a hand on the e-stop, and position the real arm near the sim's pose so the
ramp is short.** Not a technical formality; don't skip it.

Publish rate is throttled to `_REAL_ARM_PUBLISH_PERIOD_S = 0.02` (50Hz,
matching `joint_command_node`'s `control_rate_hz`) using the same
held-off-then-throttled accumulator pattern as `task_space_ik.py`, so the
delay can't build up a backlog and burst-publish once it ends. Actual
observed rate can be lower than 50Hz if the sim itself runs below real-time
(seen at ~10Hz in one `--device cpu` test with no GPU) — this isn't a bug,
`joint_command_node`'s safety layer doesn't require exactly 50Hz, just
degrades gracefully to whatever rate actually arrives.

### Bring-up checklist

1. Connect the CANable adapter (`/dev/canable`, `can0` @ 1Mbps).
2. `ros2 launch can can.launch.py`
3. `ros2 launch joint_command joint_command.launch.py`
4. E-stop armed and within reach, power supply on.
5. **A human physically present, hand on the e-stop, for the entire first
   test** -- not optional.
6. Run the teleop script with `--publish-real-left-arm` added:
   ```bash
   ./run_quest_bimanual_teleop.sh --publish-real-left-arm
   ```
7. Watch for:
   ```
   [Quest][REAL HARDWARE] Left arm will start publishing to /arm/joint_targets in 5s. ...
   ```
   During those 5 seconds, manually position the real left arm near wherever
   the sim's rest pose currently has it.
8. After the delay:
   ```
   [Quest][REAL HARDWARE] Publishing left arm to /arm/joint_targets now.
   ```
   The real arm should now track the sim (and therefore your Quest hand
   tracking) in real time.

### Do you need a physical RealSense on the real arm?

No, not for this. The stereo camera feed documented earlier in this README
(the "Both-Arms POV" stereo pair, `pov_left.jpg`/`pov_right.jpg`) is captured
entirely from the Isaac Sim viewport — it's a property of the simulated
scene, has no connection to the real-hardware bridge above, and doesn't
require or benefit from a physical camera. For the first real-hardware test,
having a human physically present (already required, see the checklist
above) watching the real arm directly is sufficient. A physical RealSense
would only be needed later if you want the same in-VR camera view *of the
real arm* instead of the simulated one -- that needs an actual D455, a real
camera-capture pipeline (not `capture_viewport_to_file`), and is unrelated to
today's joint-command bridge.
