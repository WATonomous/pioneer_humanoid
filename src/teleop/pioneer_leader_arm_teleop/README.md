# pioneer_leader_arm_teleop

The 7-servo leader arm (STS3215, torque always off) drives the Pioneer left arm joint to joint,
with no IK, in any registered scene: Isaac Sim (default) or plain MuJoCo (`--target mujoco`, CPU).
Optional recording in the shared `dataset_schema_pioneer_v1.yaml` format, from either simulator.

Angles map **one-to-one** from a shared zero: arm hanging straight down under gravity, gripper open.
That is the URDF zero, the real arm's calibrated zero, and the leader's (`calibrate_leader.py`).

| Servo | Bus ID | Sim joint | Default sign |
|-------|--------|-----------|--------------|
| A | 2 | `joint1L` shoulder flexion | +1 |
| B | 3 | `joint2l` shoulder abduction | −1 |
| C | 1 | `joint3l` shoulder rotation | −1 |
| D | 5 | `joint4l` elbow flexion | +1 |
| E | 4 | `joint5l` forearm rotation | +1 |
| F | 7 | `joint6l` wrist | −1 |
| G | 6 | gripper (41.5° open → 0° closed) | +1 |

## Calibrate (once per leader)

```bash
python calibrate_leader.py     # hang the leader straight down, gripper open, rotation joints on their marks; Enter
```

Writes `leader_calibration.json` here (zero encoder counts per servo); teleop refuses to start without it.
Redo it after re-mounting a servo.

## Run (Isaac)

The leader plugs in over USB (`/dev/ttyACM0`); the `simulation_isaac` container is privileged and mounts `/dev`. On the host you need the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and in).

```bash
# on the host, once: build and start the Isaac container
./watod build simulation_isaac        # rebuild after pulling Dockerfile or package changes
./watod up -d
./watod -t simulation_isaac           # shell inside the container

# inside the container
cd /workspace/humanoid/src/teleop/pioneer_leader_arm_teleop
$PYTHON encoder_test.py                   # first time: all 7 servos respond? (--ids 2,3 --raw for a subset)
$PYTHON calibrate_leader.py                # first time: store the hanging-pose zero

/workspace/isaaclab/isaaclab.sh -p pioneer_leader_arm_teleop.py --scene push
/workspace/isaaclab/isaaclab.sh -p pioneer_leader_arm_teleop.py --scene vial_rack --record
```

Scenes: `bare` (default), `push`, `vial_rack`, or any scene registered in `humanoid_isaac_scenes` (an unknown name lists them).

## Run (MuJoCo)

CPU only. Either in the `simulation_mj` container (needs the host's X display; it reserves a GPU for mjlab):

```bash
xhost +local:                              # on the host, once per login: let the container use the display
./watod build simulation_mj && ./watod up -d simulation_mj
./watod -t simulation_mj
cd /workspace/humanoid/src/teleop/pioneer_leader_arm_teleop
python3 pioneer_leader_arm_teleop.py --target mujoco --scene peg_insert [--record]
```

or on any machine with a display, no Docker (`dialout` group as above):

```bash
pip install mujoco feetech-servo-sdk                  # teleop
pip install torch "lerobot @ git+https://github.com/huggingface/lerobot.git@e670ac5daf9b76" pynput   # + --record (and ffmpeg with libsvtav1)
python pioneer_leader_arm_teleop.py --target mujoco --scene peg_insert   # macOS: mjpython
```

Scenes: `bare`, `peg_insert`, `zip_tie`, or any scene in `humanoid_mujoco_scenes` (see `src/simulation/mujoco_scenes/`).

- **Home:** the arm starts at home (elbow bent 90°, forearm forward, gripper open) and follows the leader only once every leader joint is within 3° of home and the gripper is open. The status line lists the joints still off.
- **Directions:** move one leader joint at a time. If a sim joint goes the wrong way, restart with that entry flipped in `--signs` (order A..G, default `1,-1,-1,1,1,-1,1`).
- **R:** reset the arm and every object in the scene; the arm waits at home until the leader is back at home. During a take, it also discards the take.
- **`--record`:** `S` start · `N` save (then auto-reset) · `D` discard → `<repo>/datasets/pioneer_v1_left_arm/sim/` · `--cameras ego,wrist_left` / `none`.
  In the MuJoCo viewer these keys (and R) also toggle display flags (shadows, reflections, …); harmless.
- Other flags: `--port`, `--baud`, `--calibration` (default `leader_calibration.json`), `--filter-alpha` (target smoothing, default 0.35).

Targets are clamped to the arm's URDF limits. Wrist damping is lowered to 2.5 in this teleop only, so the sim wrist keeps up with the leader.

## Files

| file | role |
|------|------|
| `pioneer_leader_arm_teleop.py` | entry; `--target` picks the backend before any simulator import |
| `isaac_sim.py` / `mujoco_sim.py` | backends |
| `leader_mapping.py` | shared: 1:1 mapping, home check, filter, gripper closure, leader read, wall-clock pacing |
| `servo_leader.py` / `arm_limits.py` | servo bus reader / clamp + gripper fraction |
| `calibrate_leader.py` | stores the hanging-pose zero |
| `encoder_test.py` | servo bring-up, no simulator |
