# pioneer_whole_body — whole-body Pioneer v2 scenes (MuJoCo, CPU)

Long-horizon manipulation demos for the **whole-body Pioneer v2 humanoid** (graphite helmet, taller torso, hands D,
1010 legs; 69 joints) in plain MuJoCo: the robot walks up to a workbench, looks at what it works on, and assembles
things with its two 22-joint hands. Every demo is a scripted plan (like the arm's synthetic teleop), run in physics
and recorded to a video.

```bash
pip install mujoco pillow numpy scipy
pip install -e src/simulation/whole_body_scenes
# run a demo, record its tape, render the video (CPU: MUJOCO_GL=osmesa or egl)
MUJOCO_GL=osmesa python -m pioneer_whole_body.run_demo --scene bearing_block --out out/bearing_block --render
```

`--out PREFIX` writes `PREFIX.json` (the result: steps done, success, notes), `PREFIX.npz` (the tape: the simulation
state at 30 fps, the step captions, the gaze target) and, with `--render`, `PREFIX.mp4` (1280x720, `--size`), drawn
afterwards in `--workers` processes (CPU rendering is ~0.6 s a frame, the physics ~0.1 s per simulated second).

## Scenes

| name | what the robot does |
|------|---------------------|
| `bearing_block` | 7 steps, ~87 s. Walks up to the bench; picks the bearing block by its boss (left hand), lines it up over the base plate (position, level, yaw: look, correct, look again) and sets it in the plate's pocket; picks a knob bolt from its rack (left hand), stands it upright over the hole, lowers it until the thread catches and screws it home (grip, turn, let go, turn back, regrasp: ~5 strokes); the same with the right bolt and right hand; presses the jig's clamp release with its thumb; takes the bolted assembly by both bolt knobs, side-steps 0.45 m carrying it and sets it in the outbound fixture (its latch snaps over the plate). |
| `battery_module` | 5 steps, ~155 s. Walks up with its hands raised; side-steps to the cell rack, takes a cell in each hand, side-steps back and loads them into the holder's sockets (twice: four cells); side-steps to the lid stand, lifts the lid by its two captive screws with both hands, carries it back, lines it up and sets it on the holder's pillars (the screws catch their threads); screws both down at once, one per hand; side-steps to the tester and presses its button: the indicator flips up. |
| `wheel_hub` | 4 steps, ~77 s. Walks up with its hands raised; side-steps to the wheel stand, lifts the wheel by its two handle knobs with both hands, carries it back and lowers it over the axle onto its collar; takes the hub nut from its rack (right hand), lowers it down over the axle until its thread catches and screws it home; spins the wheel a full turn by pushing a handle round with its index finger. |
| `tote_to_shelf` | 3 steps, ~68 s. Walks up with its hands raised; takes the three cells from the rack one at a time (right hand), side-steps along the bench to each slot of the tote and stands the cell in it; lifts the loaded tote by its two handle knobs with both hands, side-steps 0.9 m to the low shelf carrying it, crouches and leans, and sets it in the shelf's fixture (its latch snaps), the cells still in it. |
| `service_panel` | 8 steps, ~71 s. Walks up; pulls the parts cabinet's drawer out by its knob (left hand) and takes the cartridge out of it; slides the panel's hatch open (right hand); stands the cartridge in its bay; slides the hatch shut; takes the knob bolt from its rack, puts it through the hatch's hole until it catches the thread and screws it home; pushes the drawer shut with its fist; presses the power button with its thumb: the lamp flag comes up. |
| `gear_train` | 6 steps, ~73 s. Walks up to the gearbox; turns round, walks to the parts table behind and takes a gear in each hand by its hub knob; walks back and turns round carrying both; lowers gear 1 over the right shaft onto its collar, then gear 2 over the left one, its teeth in gear 1's; runs the train in by twisting gear 1's hub (strokes): gear 2 turns the other way; presses the test button: the lamp comes up. |
| `grasp_test` | A bolt in a rack and a plate with one threaded hole (tuning bench for the grasps). |
| `bare` | The studio and an empty workbench. |

Each scene registers `step` (mechanics: threads, clamp, button), `progress` (the current step and its instruction,
latched in order as in `humanoid_mujoco_scenes`) and `plan` (the demo).

## How it works

| file | what it is |
|------|------------|
| `robot.py` | The robot: `assets/pioneer_v2_humanoid` (the Pioneer v2 URDF from the Prometheus CAD, meshes decimated to 18 MB by `convert.py` there) with its CAD colours, a movable pelvis (x, y, z, yaw, pitch at the hips), a position actuator per joint (the DIP joints follow the PIPs: URDF mimic), gravity compensation, palm / fingertip sites and a head camera. Arms and hands collide with the world, not with each other; the legs are visual. |
| `controller.py` | `WholeBody`: every motion is a timed blend of targets turned into joint commands at 100 Hz by IK on a scratch `MjData` — palms along straight lines (7-joint arm IK, damped least squares, a posture in the null space), fingertips (position only), the gaze (neck yaw / pitch at a point), the pelvis, and the legs solved each tick so the feet stay planted (`step`, `walk`: quasi-static steps; balance isn't simulated). Turning on the spot (`turn`: the feet step round the pelvis, the hands turn with it when they carry something). Finger synergies (`SHAPES`), solved for the knobs and cells by fitting fingertip positions. |
| `skills.py` | What plans are written in: `pick_knob`, `carry`, `align_held` (hover, level and turn the held part, correct by looking), `lower_until`, `release`, `insert_bolt`, `tighten` (release / turn back / regrasp strokes), `press`, and two-handed `pick_two`, `align_two`, `move_two`, `tighten_two`, `let_go`. A failed skill raises `TaskFailed`. |
| `parts.py` | Knob bolts and threaded holes. A thread is a nut body in the part with a turn hinge and a feed slide coupled rigidly (pitch 12 mm, quick-acting); a bolt whose tip is in the hole's mouth, upright and centred gets welded onto it, from then on it only turns and feeds. Seated bolts stay tight. |
| `studio.py` | The room: floor, lights, workbench. |
| `video.py`, `shots.py` | The tape, and the renderer: a camera that follows the robot's gaze target (smoothed), pulls back while it walks, stays in front of it when it turns round, the head camera inset, a caption bar with the current step. |

## Physics vs. scripting

- Contact physics: every object moves only because fingers, palms or other parts push it; grasps hold by friction
  (no attachment welds); bolts turn because the fingertips twist the knobs; a part put down a few mm off settles
  into its pocket on its chamfers; a cell dropped into a socket stands in it.
- Not physics: the thread (a rigid helix coupling, see `parts.py`), the jig clamp (a weld that the button
  releases), the outbound and shelf fixtures' latches (welds made when the part is set flat in them), the drawer's and hatch's detents and soft-close (a small force near the end of travel), the lid's captive
  screws (welded to the lid until they catch their threads), the gear mesh (the second gear's turn is coupled to the first's, teeth in phase; a gear set on its shaft is welded to the shaft's pivot), the robot's balance (the pelvis is moved; the legs are solved so the feet don't slide), the tester's
  indicator. Object poses come from the simulator (as a perfect perception stack would give them).
- The plans are scripted, not learned; they check their results and retry where a person would (a part not
  seated, a lid whose screws didn't catch). Long scenes are sensitive to small changes: re-run the demo after
  editing a scene or a skill.
