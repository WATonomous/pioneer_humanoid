# Quest 2 controller teleoperation

Drive the Pioneer bimanual rig's **7-DoF + 1-DoF-gripper arms**
([`pioneer_bimanual_arm_8dof`](../../../assets/pioneer_bimanual_arm_8dof/README.md)) in MuJoCo with Meta Quest 2 Touch
controllers.

- **No grip button:** arm motion is always on while tracking is valid.
- **IK:** a [mink](https://github.com/kevinzakka/mink) QP (`mink_control.py`), the IK library OpenArm's VR teleop uses.
- **Isaac:** not supported yet, because `pioneer_bimanual_arm_8dof` has no USD.

## Run (no GPU, runs on a laptop)

Needs `pip install -r src/teleop/quest_controller_teleop/requirements.txt` (plus `numpy`, `Pillow`, `pyyaml`). Plug the Quest into the same machine:

```bash
mjpython src/teleop/quest_controller_teleop/mujoco/mujoco_sim.py --scene bimanual_sorting
adb reverse tcp:8444 tcp:8444
```

1. Open `http://localhost:8444/` in Meta Browser and select **Enter VR**.
2. **Headset:** shows the ego camera and both wrist cameras.
3. **MuJoCo window:** shows the scene on the laptop. With it focused, **6** resets the scene, like left Y.

The scene can be any one in `humanoid_mujoco_scenes`. `--headless` drops the window and runs under plain `python`; `mjpython` is only needed for the window on macOS.

**Tuning flags:**
- Gains: `--gain`, `--rotation-gain`.
- Speed caps: `--joint-speed` (shoulder; also the elbow unless `--elbow-speed` is set), `--wrist-speed`.
- Lead limits: `--position-lead`, `--angular-lead`.
- IK: `--ik-orientation-weight`, `--body-keepout`.
- Input filter: the flags in the table below.

`--help` lists the defaults.

**Joint limits are the URDF's**, the owner's joint chart, plus one rule: the elbow stays at least 10° bent, away from the straight-arm singularity.
- **No hand-picked ranges:** with 7 joints, the IK's pull toward the home pose uses the spare joint to keep the elbow natural. In simulation the elbow came near straight 1% of the time without the rule, and 0% with it.
- **Torso keep-out:** `--body-keepout` keeps the upper arm and forearm off an approximate torso box. It's off by default because the stand has no torso.

MuJoCo uses `--controller-frame base` by default (hand up = wrist up). Headset cameras render on their own thread (`mujoco/mujoco_cameras.py`), so the 10 ms control loop never waits on them.

## Controls

| Input | Action |
| --- | --- |
| Controller motion | Move that arm automatically while tracked |
| Index trigger | Proportional gripper closure |
| A / X | Hold for 25% translation and rotation increments |
| Left Y | Reset the scene, arms, and grippers once per press |
| Thumbstick click | Hold to pause that arm and collect wrist-pivot calibration; release to fit and resume |

**Tracking loss:** input older than 200 ms pauses movement. When tracking returns, the target re-anchors at the measured arm pose, so there's no jump. Reset and calibration re-anchor the same way.

**Pivot calibration:**
1. Hold the thumbstick click.
2. Rotate the controller about several axes, keeping your wrist pivot still.
3. Release after 3–5 s.

Offsets are saved in `wrist_calibration.json`, which is git-ignored.

## Motion mapping and IK

The control path is `index.html` → `transport.py` → `scene_driver.py` → `mink_control.py` → joint targets. `scene_driver.py` applies the input filter and `AnchoredTarget` from `article_control.py`. `control.py` supplies the URDF kinematics, rotation utilities and pivot fitting.

**Input filter** (`input_filter.py`, on by default): a 1-euro filter on each controller pose, the scheme from OpenArm's VR teleop, re-tuned on our arm.
- **Hand still:** strong smoothing. In simulation, wrist jitter dropped from 2.9 to 1.1 mm.
- **Hand moving:** little lag, about 10 ms.
- **Tracking jumps:** the speed caps spread a one-frame jump over several frames.
- **Resets:** on tracking loss, scene reset and calibration.

| Flag | Default | What it does |
|---|---|---|
| `--input-filter-cutoff` | 3 Hz | Minimum cutoff; 0 turns the filter off |
| `--input-filter-beta` | 50 s/m | How fast the position cutoff rises with hand speed |
| `--input-filter-beta-rot` | 8 s/rad | How fast the rotation cutoff rises with turn speed |
| `--max-hand-speed` | 1 m/s | Speed cap, in controller space, before `--gain` |
| `--max-hand-turn-rate` | 10 rad/s | Turn-rate cap |

**Targets** are anchored to the hand (`AnchoredTarget`).
- **What's sent:** the pose at engagement plus the gain times the hand's motion since then.
- **Lead caps:** 8 cm and 0.6 rad ahead of the measured arm. They only limit what the IK gets each frame, so the arm lines up with the hand again once it catches up.
- **Precision (A/X):** pressing or releasing re-bases the reference, with no jump.

**mink IK** (`mink_control.py`), one QP per tick:
- **Cost:** position error at the wrist pivot, plus gripper orientation error (weight 0.5), plus a pull toward the home pose (0.02), plus damping.
- **Hard constraint per joint:** the joint limit and this tick's speed cap, as one box. A joint left outside its limit, e.g. after a reset, is driven back at its speed cap.
- **Why, from the IK benchmark on this arm** (simulation only): against a sequential-DLS IK, it solved 82% vs 51% of random poses, returned to the start after wandering with 0.4° vs 28° of joint drift, and held hand orientation about 3× better.

Default gains and speed caps: translation gain 1.5, rotation gain 1, 0.9 rad/s on joints 1–4 and 3.6 rad/s on joints 5–7.
