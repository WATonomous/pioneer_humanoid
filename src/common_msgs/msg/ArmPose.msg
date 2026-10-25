std_msgs/Header header

string name

bool is_left

common_msgs/JointState shoulder
common_msgs/JointState elbow
common_msgs/JointState wrist

# Optional Overall Hand Pose
bool include_hand_pose
common_msgs/HandPose hand_pose

# Gripper position: 0 = open .. 1 = closed. joint_command drives the gripper only while
# include_gripper is true; otherwise it holds the gripper where it is.
bool include_gripper
float64 gripper_position
