from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # $CAN_CONFIG_DIR (docker-compose.interfacing.yaml: a directory bind mount of the repo's
    # can/config) wins over the installed copy, as for mit_profiles.yaml in can_node.
    config_dir = os.environ.get('CAN_CONFIG_DIR') or os.path.join(
        get_package_share_directory('can'), 'config')
    config_file = os.path.join(config_dir, 'params.yaml')

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to the CAN node configuration YAML file'
    )

    # Create the CAN node
    can_node = Node(
        package='can',
        executable='can_node',
        name='can_node',
        parameters=[LaunchConfiguration('config_file')],
        output='screen'
    )

    # Return the launch description
    return LaunchDescription([
        config_file_arg,
        can_node,
    ])
