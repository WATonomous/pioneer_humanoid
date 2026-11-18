#!/usr/bin/env python3
"""Send a few LegPose messages to joint_command_node. That is all it does.

Run it where joint_command_node is running (with `legs` enabled)::

    source /opt/watonomous/setup.bash
    python3 /opt/humanoid_scripts/legpose_test_script.py

Each pose is ABSOLUTE command-frame degrees (0 = the calibrated zero), in LegPose order:
    [hip.pitch, hip.roll, hip.yaw, knee.pitch, ankle.pitch, ankle.roll]

joint_command_node does the safe part: it clamps to the joint limits, limits the speed and starts from where
the joint really is, so a pose far from the current angle is approached gradually, not jumped to. Only joints
with `active: true` in leg_actuators.yaml move.

When this script ends, joint_command keeps the last pose for about 10 s, then lets the joints go slack.
Ctrl-C stops sending. Keep the E-stop within reach.
"""

import time

import rclpy
from rclpy.node import Node

from common_msgs.msg import LegPose

HOLD_SECONDS = 3.0  # how long to hold each pose
RATE_HZ = 50.0      # how often to publish

# [hip.pitch, hip.roll, hip.yaw, knee.pitch, ankle.pitch, ankle.roll]
POSES = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, -5, 0, 0],
    [0, 0, 0, -10, 0, 0],
    [0, 0, 0, -5, 0, 0],
    [0, 0, 0, 0, 0, 0],
]


def make_leg_pose(angles):
    msg = LegPose()
    msg.is_left = True
    msg.hip.position = [float(a) for a in angles[0:3]]
    msg.knee.position = [float(angles[3])]
    msg.ankle.position = [float(a) for a in angles[4:6]]
    return msg


def main():
    rclpy.init()
    node = Node("legpose_test_script")
    publisher = node.create_publisher(LegPose, "/leg/left/joint_targets", 10)
    time.sleep(1.0)  # let the connection to joint_command come up
    try:
        for number, angles in enumerate(POSES, 1):
            print(f"pose {number}/{len(POSES)}: {angles}", flush=True)
            end = time.monotonic() + HOLD_SECONDS
            while time.monotonic() < end:
                publisher.publish(make_leg_pose(angles))
                time.sleep(1.0 / RATE_HZ)
        print("done")
    except KeyboardInterrupt:
        print("stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
