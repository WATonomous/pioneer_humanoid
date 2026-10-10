# slam_sim_mj — RTAB-Map SLAM against a simulated RealSense D455

A simulation testbed for the real-camera pipeline in `../slam-quickstart.md`. A
simulated D455 walks a corridor loop in MuJoCo and RTAB-Map builds a floor plan from
it — the same thing as walking a building with the real camera, without the building
or the camera.

It exists so the SLAM side can be developed and regression-tested without hardware:
the room is known exactly, the trajectory is repeatable, and a full run takes about
three minutes on a laptop with no GPU.

Topic names, encodings and frame ids match what `realsense2_camera` publishes, so the
SLAM half cannot tell simulation from hardware. The launch file runs the *same*
`imu_filter_madgwick` node with the same parameters as the real-hardware procedure —
a fix proven here is a fix for the real rig.

## Quick start

```bash
python3 -m venv --system-site-packages .venv     # at the repo root, once
.venv/bin/pip install mujoco imageio imageio-ffmpeg

src/perception/perception/slam/slam_sim_mj/run.sh             # 2 laps, 100 s each
RECORD=1 src/perception/perception/slam/slam_sim_mj/run.sh    # ...and a walkthrough MP4
```

`--system-site-packages` is not optional: Ubuntu 24.04 refuses a bare `pip install`
(PEP 668), and a plain venv cannot see apt's `rclpy`. The point is one interpreter
holding both ROS and MuJoCo — ROS 2 Jazzy and the MuJoCo wheel are both Python 3.12,
so the renderer and `rclpy` run in a single process with no bridge between them.

The scene is **generated, not committed**: `run.sh` builds `scene/room.xml` and its 59
textures on first use, deterministically from a fixed seed, so every checkout gets an
identical room. Running the publisher directly on a fresh checkout errors with the
command to run.

Outputs land in `outputs/slam_mj/` at the repo root (gitignored, matching the
convention used by `outputs/rl/` and `outputs/train/`): `rtabmap.db`, `map_2d.png`,
`rtabmap.log`, and `demo.mp4` when recording. Recording is off by default because
encoding runs inside the publish loop and costs about 1% of the frame rate. The
ffmpeg binary comes from the `imageio-ffmpeg` wheel, so no system package is needed.

### Interactively, with RViz

```bash
# terminal 1
source /opt/ros/jazzy/setup.bash
ros2 launch src/perception/perception/slam/slam_sim_mj/launch/slam_sim_mj.launch.py

# terminal 2
source /opt/ros/jazzy/setup.bash
MUJOCO_GL=egl .venv/bin/python \
    src/perception/perception/slam/slam_sim_mj/mj_slam_publisher.py --laps 2
```

Same shape as the terminals in the real-hardware quickstart.

### Exploring the room by hand

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco.viewer \
    --mjcf=src/perception/perception/slam/slam_sim_mj/scene/room.xml
```

Free camera, no ROS attached. Use `glfw`, not `egl` — `egl` is the headless offscreen
path the publisher uses and will not open a window.

## Layout

```
mj_slam_publisher.py           sim half: renders the D455 and publishes RGB-D + IMU
launch/slam_sim_mj.launch.py   SLAM half: madgwick + static TFs + rtabmap
mj_camera.py                   MuJoCo offscreen RGB-D with real D455 intrinsics
orbit_path.py                  camera trajectory maths — pure numpy
imu.py                         synthetic IMU from the trajectory — pure numpy
scene/build_room.py            generates the corridor room, furniture and textures
check_map.py                   read the database: closures, gravity links, z drift
export_map.py                  save the 2D occupancy grid as a PNG
run.sh                         one unattended run end to end
tests/                         149 tests, ~0.3 s, no GPU needed
```

The two halves talk only over ROS topics and share no code. Swapping
`mj_slam_publisher.py` for a real camera driver leaves the SLAM half untouched.

## The room

`scene/build_room.py` generates an office corridor circuit: painted walls,
baseboards, doors with frames and handles, notice boards, locker banks, a tiled floor
and ceiling light panels.

```bash
.venv/bin/python src/perception/perception/slam/slam_sim_mj/scene/build_room.py --preview
```

`--preview` renders four points of view around the loop and prints the ORB feature
count at each. **Use it as the gate on any scene change.** Below roughly 150 features
per frame visual odometry starts failing the way the real-hardware quickstart
describes ("Not enough inliers"); this room sits at 300–500.

Two properties are load-bearing for SLAM rather than for looks:

- **Nothing repeats along a corridor.** Every wall bay, door and notice board gets its
  own generated texture. MuJoCo's builtin `checker` is a *repeating* pattern, which is
  worse than no texture at all: identical patches metres apart make visual odometry
  match the wrong one and loop closure fire on places the camera has never been.
- **Geometry carries the features, not paint.** Real corridor walls are nearly
  featureless, which is the documented failure mode on the real rig. Door frames,
  baseboards, handles and locker edges give a corner detector real corners, which is
  what allows realistic flat wall colour. This is measured, not assumed: replacing
  high-contrast noise textures with realistic paint plus furniture *halved* the
  feature count and yet increased loop closures and reduced z drift. Distinctive
  landmarks beat raw texture density.

Paint colour is chosen once per wall run rather than per bay, so a corridor reads as
one building instead of patchwork.

### Swapping in a different room

Point `--scene` at another MJCF. Nothing else in the pipeline knows about the scene
file; `mj_camera.py` includes whatever it is given and attaches the camera to it.

Importing a downloaded room is possible but not as easy as it sounds: **MuJoCo binds
one texture per mesh**, so a multi-material OBJ/GLTF arrives untextured unless it is
split per material first. There is no collision requirement — the camera is
kinematically posed and nothing is simulated — so visual-only geometry is enough.
Two things to check on any new scene:

- **Does the walk loop fit?** A path through a wall renders a blank camera and a junk
  map. Measure rather than guess; many real floor plans have no viable rectangular
  loop at all, and those that do often have exactly one placement.
- **Is it lit?** MuJoCo point lights attenuate, so a long corridor lit only from the
  ceiling renders near-black a few metres out and drops ORB to ~40 features per frame.
  The generated room leans on a bright ambient headlight for this reason; a scene with
  baked lighting will not need to.

## Invariants — violating these breaks things quietly

- **Intrinsics are real.** `fx=383.682, fy=383.165` at 640×480, taken from a real
  D455 recording, so the simulated camera matches the hardware. Focal length in pixels
  **scales with resolution**, so `--width` changes sampling, not field of view.
- **`CameraInfo` is derived from the model**, never from the constructor arguments, so
  published intrinsics cannot drift from the images they describe.
- **`znear`/`zfar` are fractions of `model.stat.extent`, not metres.** Setting them in
  metres clips the near plane several metres out; two thirds of every frame comes back
  invalid and the walls vanish. It looks like a broken scene, not a bad number.
- **Depth is distance to the image plane** (pinned by test), not radial range, which
  would bow the map. Out of range → `0`, RTAB-Map's convention for "no reading".
- **The path advances by simulation time**, and the loop is paced against an
  **absolute** schedule. Per-iteration `sleep(dt - elapsed)` compounds timer
  granularity — measured 28.4 Hz against a 30 Hz target, which reads to the IMU filter
  as the camera turning 5% slower than the images show.
- **All three image messages share one stamp** — one render of one camera. That is what
  lets the launch file set `approx_sync:=false`, and `approx_rgbd_sync:=false`
  separately, which it must.
- **Output goes to `outputs/slam_mj/rtabmap.db`, never `~/.ros/rtabmap.db`.** The
  launch file passes `-d`, which *deletes* the database it is pointed at, and that
  default path is where real-hardware recordings accumulate.
- **The repo root is found by walking up to `.git`**, not by counting `../..`. A fixed
  depth resolves silently to the wrong directory as soon as the package moves, and a
  wrong root means `-d` deletes a database somewhere unexpected.

## Verifying a run

Read the database, not the log — RTAB-Map does not print "loop closure" at default
verbosity, so grepping reports zero on a run that is closing loops constantly.

```bash
python3 src/perception/perception/slam/slam_sim_mj/check_map.py outputs/slam_mj/rtabmap.db
```

A healthy 2-lap run at the default 100 s/lap shows:

- **loop closures non-zero** — `GlobalClosure` and usually some `LocalSpaceClosure`.
  Zero means pure odometry with nothing correcting it.
- **one `Gravity` link per node** — confirms the IMU prior is reaching rtabmap.
- **z drift well under `Grid/MaxGroundHeight` (0.15 m)**, on trajectories that are
  flat by construction.
- **`free fraction of known cells` around 90%** from `export_map.py`. A collapsed map
  reads near 50% and its PNG shows thick black smears instead of walls.

### Walk slowly

`--period` sets seconds per lap; the 100 s default is about 0.38 m/s over the 37.9 m
loop. Odometry degrades **sharply** rather than gradually above roughly 0.5 m/s — a
run fast enough to fail can log almost no registration errors and still climb metres
in z, so the logs will not warn you. Raise `--period` before suspecting anything else.

Run-to-run variance at a fixed speed is significant, so **one run is not evidence**;
re-run before concluding that a change helped. This matches the real-hardware
quickstart's advice to move slower when tracking stalls.

`check_map.py` reports drift on the **raw odometry** poses stored per node, not the
optimised graph the map is built from. It measures how hard the pose graph had to
work, not the final map's error.

**`Gravity` links are not loop closures.** Every node carries exactly one gravity
constraint, so counting them as closures reports success on a run that closed nothing.
Keep `LINK_TYPES` in sync with
`/opt/ros/jazzy/include/rtabmap-0.22/rtabmap/core/Link.h` — the values differ between
rtabmap versions.

**One lap does not close loops.** The revisit happens only in the final seconds, with
no time to detect it. Two laps is the minimum that proves anything.

**Getting a map out after a run needs both** TRANSIENT_LOCAL durability *and* a call to
`/rtabmap/rtabmap/publish_map`. RTAB-Map assembles the occupancy grid lazily and stops
once data stops arriving, so the latched topic stays empty however correct the QoS is.
`export_map.py` does both.

## Why there is a synthetic IMU

RTAB-Map uses an IMU as a gravity prior in RGB-D mode, which pins roll and pitch. A
real D455 has one; without it, odometry drifts in z on a path that is flat by
construction, and since `Grid/MaxGroundHeight` is 0.15 m that drift destroys
ground/obstacle classification and the 2D map degenerates into a blob with no border.

A simulated IMU cannot simply be attached to this camera: physics-based IMU sensors
derive acceleration by differencing rigid-body velocities, and this camera is
kinematically posed rather than dynamically simulated, so such a sensor reads zero.
`imu.py` differentiates the known trajectory instead, which is exact.

The accelerometer convention is the easy thing to get wrong: it measures **specific
force**, so at rest it reads +9.8 along its own up, not zero. `tests/test_imu.py` pins
that, along with free-fall reading zero and gravity following the sensor as it tilts.

`check_map.py` reports z drift directly. **Measure it before touching any `Grid/*`
parameter** — those parameters are correct; what breaks is the datum they measure
against.

## Known limitations

- **Run-to-run variance is significant** and not yet understood: identical inputs
  give noticeably different closure counts. Slowing down narrows it but does not
  remove it.
- **z drift sits above the 0.15 m ground threshold** on a typical run, though the
  optimised map is clean regardless. More distinctive scene geometry reduces it.
- **`camera_pitch` for `--path orbit` is untested.** Not an issue for the default
  `--path corridor`, which gazes level and runs with `camera_pitch:=0.0`.
- **No planning.** The occupancy grid is produced but nothing consumes it; Nav2 is not
  wired up.
