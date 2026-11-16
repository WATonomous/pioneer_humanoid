# Soft TPU Fin Ray finger (MuJoCo)

The gripper's fin-ray finger (`assets/pioneer_bimanual_arm/meshes/link7_finger.STL`) as a finger that bends, instead of
the rigid boxes the arm models use. It runs faster than real time.

```bash
pip install -r requirements.txt
MUJOCO_GL=egl python squeeze.py cyl --video cyl.mp4   # also: box, card
```

`squeeze.py` closes two fingers on a fixed object, holds, and prints the grip force and how far the contact face is
pushed back along its length.

## Model (`finray.py`)

The profile (the STL cut at mid-width) is skeletonised into its walls and ribs. Each one is a chain of 3 mm rigid links
on hinges about the finger's width axis, with bending springs from beam theory (`k = E·I/l`, `I = W·t³/12`, `t` the local
thickness). The ribs close their loops with welds, and the top 15 mm is clamped to the carriage. 252 links per finger.

```
mesh    = skeleton(profile) -> walls + ribs as chains of 3 mm links    # 252 hinges per finger
spring  = E · BEAM_E · W · t³ / 12 / l  per hinge                       # t from the profile, BEAM_E = 0.625
loops   = weld(rib end, wall)                                           # solimp 0.999
step    = 10 ms, implicitfast, Newton                                   # 5x real time
```

`BEAM_E = 0.625` is the one fitted number. It makes the beam network match a 2D finite-element model of the same profile
(TPU 95A, E = 20 MPa) on the cylinder: beam theory overstates stiffness where ribs meet the thicker walls.

## Checked against the finite-element model

`reference_fem.py cyl|box` (about 4 min each) is that FEM model: the profile on a 0.75 mm pixel mesh as a MuJoCo flex
(plane-stress TPU), 1 ms steps. It is only the validation reference, not a second finger.

Squeeze to about 10.6 mm carriage travel, then hold. "Face pushed back" is the contact face's motion relative to the
carriage, at heights from the tip (-65 mm) up to the clamp.

| object | model | grip | face pushed back (mm), z = -65 / -50 / -35 / -20 / -5 / 10 / 25 / 40 mm | speed |
|---|---|---|---|---|
| cylinder (fitted) | FEM | 3.65 N | 3.4 / 7.0 / 9.8 / 10.7 / 6.9 / 3.8 / 1.7 / 0.2 | 0.014x |
| | beam network | 3.65 N | 3.3 / 7.0 / 9.5 / 10.1 / 7.0 / 4.2 / 1.9 / 0.3 | 5.5x |
| box (not fitted) | FEM | 4.57 N | 4.2 / 7.6 / 10.2 / 11.1 / 8.5 / 4.8 / 2.1 / 0.3 | 0.013x |
| | beam network | 4.77 N | 3.8 / 7.3 / 9.8 / 10.6 / 8.9 / 5.5 / 2.6 / 0.4 | 5.0x |

The FEM's thin card case isn't a valid check: its contact nodes pass through the 0.8 mm card.

## Limits

- In-plane bending only: no twist, and the finger can't bend across its width.
- TPU stiffness is a datasheet value, not measured on our printed fingers.
- The hinges carry large rotor inertia (the springs' period is about 12 steps), which suits grasping, not fast impacts.
- Only the two fingers on a fixed object here; not yet on the arm.
