"""Start foxglove_bridge so Foxglove can show live topics (a replayed bag, or the live camera and map).

Usage:
    ros2 launch terrain_mapping_bringup foxglove.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            parameters=[{"address": "0.0.0.0"}],
        ),
    ])
