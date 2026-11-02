#!/bin/bash
# Puts the emulator's configs where joint_command and the arm scripts look, then runs the command.
set -e
source /opt/ros/humble/setup.bash
source /opt/watonomous/setup.bash
mkdir -p "${HOME}" "${ROS_HOME}"
cd /workspace/humanoid
python3 -m arm_emulator.mit_native > /dev/null
python3 -m arm_emulator.sim_config /opt/joint_command_config
cp /opt/joint_command_config/arm_calibration.yaml /calibration/arm_calibration.yaml
cp /opt/joint_command_config/arm_actuators.yaml /opt/joint_command_config/arm_calibration.yaml \
   /opt/watonomous/joint_command/share/joint_command/config/
exec "$@"
