"""Start foxglove_bridge so Foxglove can show live topics (a replayed bag, or the live camera and map).

Usage:
    ros2 launch terrain_mapping_bringup foxglove.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node

# Only the topics the terrain layout uses, so the bridge doesn't stream every raw camera image.
TOPIC_WHITELIST = [
    "^/elevation_mapping_node/elevation_map_(raw|filter)$",
    "^/camera/depth/points$",
    "^/tf(_static)?$",
    "^/camera/camera/imu$",
    "^/camera/camera/color/(image_raw|camera_info)$",
    "^/camera/camera/aligned_depth_to_color/(image_raw|camera_info)$",
]


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            parameters=[{"address": "0.0.0.0", "topic_whitelist": TOPIC_WHITELIST}],
        ),
    ])
