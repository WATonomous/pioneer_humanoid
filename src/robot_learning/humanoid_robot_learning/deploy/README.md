### Run

1. **Bring up the arm**: follow [ARM_BRINGUP.md](../../../interfacing/ARM_BRINGUP.md) §0–3 (can_node, calibrate, joint_command).
2. **Open the policy container:**
   ```bash
   ./watod -t simulation_mj
   cd /workspace/humanoid/src/robot_learning
   ```
3. **Dry run** (nothing moves). Check the printed target and measured angles agree:
   ```bash
   python3 -m humanoid_robot_learning.deploy.real_arm --policy <run>/checkpoints/last/pretrained_model
   ```
4. **Live:** add `--live`, then type `live` at the prompt:
   ```bash
   python3 -m humanoid_robot_learning.deploy.real_arm --policy <run>/checkpoints/last/pretrained_model --live
   ```
5. **Stop:** press Ctrl-C and the arm holds where it is. Support the arm, then press Ctrl-C again.
   10 s later joint_command sets kp = 0 and the arm goes limp.
