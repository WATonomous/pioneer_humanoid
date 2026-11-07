# Bimanual arm v3: 7-DOF arms (wrist pitch + yaw on AK45-10, gripper on AK40-10)

`urdf/pioneer_bimanual_arm_v3.urdf` is the bimanual arm with the redesigned lower arm. The v2 URDF
`urdf/pioneer_bimanual_arm.urdf`, its USD and all the code that uses it are unchanged. v3 shares v2's `meshes/` folder;
its new links are the `v3_*.STL` files there.

| | v2 (current) | v3 |
|---|---|---|
| wrist | one joint, GL40 (0.25 / 0.73 N m) | **pitch + yaw**, two CubeMars **AK45-10** (2.5 / 7 N m) |
| gripper motor | GL40 | CubeMars **AK40-10** (1.3 / 4.1 N m) |
| gripper | MGN9 rail, TPU fin-ray fingers, crank + 2 links | the same gripper, with a new horn on the AK40's output (same link pins) |
| joints per arm | 6 + 2 claws | **7** + 2 claws |

Joints 1-5 and the links base..link4 are v2's, unchanged. Two things differ on joint 5:

- its origin is now on the AK80-9 output face for both arms (z -0.1075 below link4, as the right arm already was);
- the left arm's joint 5 sat 61.7 mm lower along the same axis, which gives the same motion.

## Joints

| joint (right / left) | what | axis (link frame) | range |
|---|---|---|---|
| `joint6` / `joint6l` | wrist pitch, AK45-10 | -y / +y (as v2's joint6) | +-90 deg |
| `joint7` / `joint7l` | **wrist yaw, AK45-10 (new)** | +x | +-40 deg |
| `joint8` / `joint8l` | claw, -0.041..0 m (v2's joint7) | -y / +y | closes negative |
| `joint9` / `joint9l` | claw, 0..0.041 m (v2's joint8) | -y / +y | closes positive |

- Joints 5, 6 and 7 meet at one point, the wrist centre, 130 mm below the AK80 output. That makes a spherical wrist,
  like OpenArm's: the hand's orientation decouples from its position.
- `link7` / `link7l` is the hand, with its origin at the wrist centre. The wrist cameras ride on it (`arm_params_v3.CAMERAS_WRIST`),
  in the same pose relative to the claws as v2.
- One motor drives both claws, so `joint9 = -joint8`, as before.

**Masses.** Motors are at their datasheet weights, prints at 0.65 g/cm3, steel 7.85; each claw keeps v2's 63.4 g.

| link | mass | what it holds |
|---|---|---|
| link5 | 421 g | forearm + pitch motor |
| link6 | 302 g | knuckle + yaw motor |
| link7 | 369 g | wrist frame + AK40 + rail plate, rail, links, horn, camera |
| link8 / link9 (each claw: finger + mount + carriage) | 63.4 g | |

Below the elbow the arm is 1.22 kg, about 0.5 kg more than v2's 0.70 kg.

## Use it

- **MuJoCo:** `pioneer_humanoid.mujoco_bimanual_arm.arm_spec(cameras, version="v3")`. The default is still `"v2"`.
  Joint lists, gains, the gripper and the cameras are in `pioneer_humanoid/arm_params_v3.py`.
- **Isaac:** the USD has to be converted from the v3 URDF first. Use Isaac Lab's `convert_urdf.py`, with the same
  options as `usd/config.yaml`. That conversion is not done here.

**Checked in MuJoCo (2 ms step):**

- It loads: 18 joints, 18 actuators and 3 cameras.
- It holds the home pose under gravity.
- The wrists track pitch / yaw targets of +-1.2 / +-0.5 rad on both arms.
- The claws close to +-0.041.

**A note from the check:** v3 uses v2's arm and gripper gains, with the wrist group scaled to the AK45's torque.

- **Joint chatter.** With the default Euler integrator, joint 5 chatters at the home pose (about 4 rad/s, v2 about
  2.5 rad/s). That comes from the arm gains' damping on a light joint. `spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST`
  makes both v2 and v3 hold still to 2 mrad.

## Where it comes from

The CAD is Prometheus `Backend/cad_bench/projects/pioneer_wrist/`. It is checked by the CAD harness: interference over
the full pitch / yaw ranges, FEA on the printed frames, the bolts, and motor torque. `make_urdf.py` there writes this
URDF and the `v3_*` meshes. The team's current gripper was taken from this folder's own meshes (`link6_print`,
`link6_steel`, `link7/8_*`), because no STEP of it is in the repo.

## Not done yet

- The **real-arm configs** (`src/interfacing/joint_command/config`, `src/interfacing/can/config/mit_profiles.yaml`)
  need the AK45-10s and the AK40-10: CAN ids, MIT ranges and limits. Bench-check them before enabling.
- The **Isaac USD**: see "Use it" above.
- **Cable routing** through the wrist is not modelled yet. It needs a service loop over pitch, behind the knuckle, and
  one over yaw, under the yaw motor.
