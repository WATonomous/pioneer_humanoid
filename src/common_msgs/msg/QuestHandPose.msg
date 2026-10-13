geometry_msgs/Pose left_wrist
geometry_msgs/Pose right_wrist
float32[] left_hand_joints
float32[] right_hand_joints
geometry_msgs/Pose head_pose
# Quest 2 controller trigger, 0 (released) .. 1 (fully pulled). Drives that arm's gripper.
# Only set when the page tracks controllers; 0 under hand tracking.
float32 left_trigger
float32 right_trigger
# Quest 2 left-controller Y button held (WebXR xr-standard button 5). On its rising edge the sim
# discards the episode being recorded (if --record) and resets the scene.
bool reset_button
# Quest 2 right-controller B button held (xr-standard button 5). On its rising edge the sim saves
# the episode being recorded (if --record) and resets the scene.
bool save_button
