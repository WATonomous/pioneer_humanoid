# humanoid_hands_sim

The Pioneer humanoid with the tendon hands (the URDF from Prometheus,
`Backend/cad_bench/projects/humanoid_urdf/pioneer_humanoid_mujoco.urdf`) in plain MuJoCo, CPU only, doing the
`matcha` task from `humanoid_mujoco_scenes` with both hands: scoop the matcha into the cup, pour the water, whisk
it (the right hand holds the cup still), serve the cup on the tray.

```bash
pip install mujoco scipy pillow
pip install -e src/pioneer_humanoid -e src/simulation/mujoco_scenes -e src/simulation/mujoco_humanoid_hands
# the URDF (206 MB of meshes) stays in Prometheus; point at it, or check Prometheus out next to this repo
export PIONEER_HUMANOID_URDF_DIR=/path/to/Prometheus/Backend/cad_bench/projects/humanoid_urdf

python -m humanoid_hands_sim.matcha_demo --out matcha.npy                  # ~75 s simulated, ~10 min
MUJOCO_GL=egl python -m humanoid_hands_sim.render matcha.npy matcha.mp4 --size 360,640 --fps 10   # quick look
MUJOCO_GL=egl python -m humanoid_hands_sim.render matcha.npy matcha.mp4 --size 720,1280 --shadow  # full
```

Tuned on the URDF as of Prometheus `dd5a01c9` (PR #23: the hand links written un-posed, so a finger joint at 0
is a straight finger hinging on its real pivots). An older URDF still runs, but its whisking falls short.

The demo prints each step as the scene's checks latch it, and where every tool ends up:

```
[ 17.6s] ladle: step 1/4  matcha in cup 8/8  water 0/24  whisk path 0.00 m
[ 35.7s] pitcher: step 2/4  matcha in cup 8/8  water 23/24  whisk path 0.00 m
[ 60.2s] whisked: step 3/4  matcha in cup 8/8  water 23/24  whisk path 0.32 m
[ 77.6s] served: step 4/4  matcha in cup 8/8  water 23/24  whisk path 0.33 m
```

Then check that nothing passed through anything (the arms collide with nothing in the sim, see below):

```bash
python -m humanoid_hands_sim.check matcha.npy   # closest: left forearm 0.8 mm from the scoop (against the forearm's
                                                # convex hull, larger than the forearm) for 0.5 s; everything else >= 9.9 mm
```

## Files

| file | what |
|---|---|
| `robot.py` | loads the URDF into an `MjSpec`, stands it at the counter (base fixed), adds actuators, grasp sites, the grasp-assist welds; `Arm`: damped-least-squares IK of one arm's 7 joints on its grasp site |
| `matcha_hands.py` | the scene: `humanoid_mujoco_scenes.matcha`'s balls, cup and checks, tools re-laid out for hands (below) |
| `control.py` | `Sim`: Cartesian moves (IK every 10 ms, PD joint targets), finger closure, `attach`/`detach` |
| `matcha_demo.py` | the scripted task |
| `render.py` | qpos record to MP4 (parallel chunks) or one PNG |
| `check.py` | arm links vs the torso and thighs, tools vs arm links, over a whole record: closest approach and when |

## What is real and what is not

Physics that is simulated: the arms are driven only through their joint PD actuators; the fingers close
physically and collide (each hand link's own visual mesh, as a convex hull); the matcha and water are the
scene's free 5 mm balls, poured, spilled and counted by the scene's own checks; the whisk path is measured
in the cup; the cup is carried and set down.

Not real, or assumed:

- **Grasp assist.** Once the fingers have closed on a handle *and* the scene confirms the grip (a finger plus
  the thumb or the palm touching it), a weld holds the tool in the palm until it is let go. Without it the
  hull-collided fingers hold a light tool only now and then (a sweep of grasp offsets and finger torques:
  some held through a 90 deg roll, neighbouring settings dropped it). Fingertip pads, the real hand's
  strength and soft contact would be needed to drop this.
- **Scripted, not learned.** Every motion is a hand-written waypoint script; it is closed loop only on where
  the tool sits in the hand (re-aiming the pour and the set-down on the measured tool pose). The run is
  deterministic: the same code gives the same run. Small changes (a waypoint, a contact setting) can change
  which balls land or whether a tool topples.
- **Joint ranges widened.** The URDF's arm limits are placeholders (+-90 deg). Here the elbow bends 0-145 deg
  and the forearm rolls +-150 deg (a jug pour palm-down needs ~140 deg). Not from the CAD. The hand ranges are
  the URDF's (exact, from `finger_drive.py`); the wrist deviation range is often the binding limit.
- **Gains and strength guessed.** Arm PD kp 400 (60 N m cap), wrist 60, fingers kp 10 with a 1.5 N m cap;
  gravity compensated on every robot body; base fixed (the legs and inertia are placeholders in the URDF, so
  the robot stands on a fixed base, it does not balance).
- **No arm, torso or leg collision.** The URDF's collision is each link's bounding box (the forearm's is
  ~10 cm square and drags on the table), so only the hands collide. Instead the IK keeps each arm (capsules,
  50 mm left, 60 mm right) out of three boxes fitted to the torso and pelvis, and the layout keeps the
  forearms off the tools; `check.py` measures what is left on the real meshes. A first version tucked the
  left elbow into the torso while pouring: that is what this is for.

## matcha_hands vs matcha (gripper)

Same balls, cup size, checks and step order. Changed for a 1.28 m robot with hands:

- **Counter at 0.78 m** (gripper scene 0.705 m): the elbow cannot go below 0.96 m, so at 0.705 m a handshake
  grip on a handle needs the wrist bent past its range.
- **Jug handles.** The gripper scene's handles are posts on the rim, sticking up, for jaws from above. A hand
  holding one curls its fingers into the container. Here the scoop ("ladle") and the pitcher have a post
  30 mm off the wall on a base plate, like a jug handle: the fingers wrap it through the gap, and rolling the
  forearm tips the container over its far lip. The cup is a mug (same cup, a handle on its -Y side): the plain
  66 mm cup is shorter than the hand's finger span, and only the little finger reaches it.
- **Layout** spread out for the hand's footprint (a hand on a post covers ~24 cm along the fingers) and so the
  pouring forearm passes over no tool; the cup where both hands reach it without the elbows meeting the body;
  the tray on the right for the right hand. No `reset` randomisation yet.
