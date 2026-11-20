# `task_space_ik.py`

Task-space (Cartesian) IK for the Pioneer bimanual arm 8dof's **left arm** (7 joints). A cube in the
viewport is the absolute tool-point (TCP) pose target — drag/rotate it and the arm follows via the
Quest teleop's mink IK (`quest_controller_teleop/mink_control.py`).

```bash
# inside the simulation_isaac watod container:
cd /workspace/humanoid/src/teleop/task_space_controller
/workspace/isaaclab/isaaclab.sh -p task_space_ik.py
```

Sim only. The real arm is still the old 6-DoF arm: it is driven by
`pioneer_leader_arm_teleop/real_arm_teleop.py` until the 8dof real arm exists.
