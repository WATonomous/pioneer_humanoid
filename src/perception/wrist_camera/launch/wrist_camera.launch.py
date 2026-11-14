from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import launch_ros.actions
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        # /dev/v4l/by-id/usb-...-video-index0 survives reboots and replugging; /dev/videoN may not.
        DeclareLaunchArgument('device', default_value='/dev/video0'),
        DeclareLaunchArgument('fps', default_value='30'),
        launch_ros.actions.Node(
            name='wrist_cam',
            package='wrist_camera',
            executable='wrist_camera_node',
            parameters=[{
                # Typed: launch args are strings, and fps is declared as an int in the node.
                'device': ParameterValue(LaunchConfiguration('device'), value_type=str),
                'fps': ParameterValue(LaunchConfiguration('fps'), value_type=int),
            }],
            output='screen',
            emulate_tty=True),
    ])
