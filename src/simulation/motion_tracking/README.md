# Motion Tracking

**Repository:** [motion_tracking_watonomous](https://github.com/WATonomous/Humanoid_motion_tracking)

A pipeline that takes a recording of a human performing a motion and trains the WATonomous (Wato) humanoid to reproduce it in physics simulation.

## What it does

1. **Convert:** reads a human motion capture file (BVH).
2. **Retarget:** maps the human motion onto the Wato robot's joints using [GMR](https://github.com/YanjieZe/GMR), producing a motion the robot's body can follow.
3. **Train:** trains a physics-aware reinforcement learning policy that makes Wato perform the motion while balancing under gravity, contacts and its real motor limits, using [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) in Isaac Sim 5.1 / Isaac Lab 2.3.
4. **Evaluate:** plays the trained policy through the whole motion and records a real-time video.

The whole conversion (BVH → retargeted motion → training-ready file) runs with one command, `convert_motion.sh`. Training and evaluation are one command each.

## Supported motions

Any motion can be used: walking, dancing, boxing and so on. The pipeline has been tested end to end with **Xsens** motion capture (BVH exported in 3DSM format). GMR supports other input formats, but they haven't been tested with Wato yet.

## To Do

- [ ] Test other input formats (only Xsens BVH has been tested so far)
- [ ] Update the motor specs in the Wato robot config to match the current hardware
- [ ] Create a simplified collision URDF (simple capsule/box hitboxes instead of the CAD meshes) 

**Currently for training self-collisions is turned off, as the current collision model self-collides in isaacsim**

## Getting started

See the repository's [README](https://github.com/WATonomous/Humanoid_motion_tracking#readme) for installation, the full pipeline command, and example training and evaluation commands.
