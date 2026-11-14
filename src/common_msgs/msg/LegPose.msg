std_msgs/Header header

string name

bool is_left

# Degrees, command frame, same as ArmPose.
#   hip.position   = [pitch, roll, yaw]
#   knee.position  = [pitch]
#   ankle.position = [pitch, roll]
common_msgs/JointState hip
common_msgs/JointState knee
common_msgs/JointState ankle
