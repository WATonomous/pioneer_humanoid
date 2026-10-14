from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "topic", default_value="/camera/camera/depth/image_rect_raw"
        ),
        DeclareLaunchArgument(
            "msg_type", default_value="sensor_msgs/msg/Image"
        ),
        
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            parameters=[{"address": "0.0.0.0"}],
        ),
        
        Node(
            package="perception",
            executable="latency_logger_node",
            name="latency_logger",
            parameters=[{
                "topic": LaunchConfiguration("topic"),
                "msg_type": LaunchConfiguration("msg_type"),
            }],
        ),
    ])