# Task: solve the Tower of Hanoi with the bimanual robot, camera-only

You control a two-armed robot (Pioneer 8-DOF arms, parallel grippers) in a physics simulation. The simulation is paused
while you think; each command runs to completion and returns JSON.

The toy: a wooden base with three vertical pegs A, B, C and 3 coloured rings with a hole in the middle, stacked on
peg A (largest at the bottom). Goal: move the whole stack onto peg C. Classic rules: one ring at a time, never a larger
ring on a smaller one. Besides the pegs, a ring may rest only on the round hand-over tray T, one ring at a time and only
while passing it from one arm to the other.

You do NOT get ring positions. `state` gives the fixed layout (peg and tray positions, peg height, board height, ring
diameters and thickness) and the robot's own fingertip points and grippers. Where the rings are, and which ring is on
which peg, you must find out from camera images.

Commands (run from the repo root):
- `python3 src/simulation/agent_control/agent_cli.py state`
- `python3 src/simulation/agent_control/agent_cli.py move <left|right> x y z` - fingertip point to (x, y, z) in metres,
  world frame, in a straight line, gripper pointing down
- `python3 src/simulation/agent_control/agent_cli.py grip <left|right> <open|close>` - `holding: true` if something is
  between the fingers
- `python3 src/simulation/agent_control/agent_cli.py look <cams> look_<n>` - cams from film, top, wrist_left,
  wrist_right (comma separated). The reply lists the image files (open them to see them) and each camera's calibration
  (position, image_right / image_down / forward unit vectors in world, fx, fy, cx, cy in pixels), so you can turn
  pixels into rays. The wrist cameras look down past the fingers and are the best way to centre the gripper on a ring
  or a peg.
- `python3 src/simulation/agent_control/agent_cli.py caption <text>` - one short line shown in the video: say what you
  are doing and why
- `python3 src/simulation/agent_control/agent_cli.py done` - only when finished; it ends the recording

Hints: each arm can only point straight down on its own side (left arm works at y > ~+0.03, right at y < ~-0.03, both
near y = 0). Grip a ring with the fingertip point about 1 cm below its top surface. A carried ring hangs about 1.2 cm
below the fingertip point and has to clear the peg tops. A ring only slides down a peg if it is centred on it to within
a few millimetres. Moves fail with an error code (IK_UNREACHABLE, PATH_UNREACHABLE, BELOW_TABLE) instead of doing
something unsafe: read it and adapt. Look after each placement to check it worked; if a ring fell, find it and recover.

Rules for you as the agent: use only the commands above. Do not read hanoi_server.py, log.jsonl or anything else under
outputs/agent_control except the images you capture. Finish with all rings on peg C, largest at the bottom, then run
`done`.
