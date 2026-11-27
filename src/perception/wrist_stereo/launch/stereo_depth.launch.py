from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('wrist_stereo')

    # Launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(pkg_share, 'config', 'stereo_config.yaml'),
        description='Path to stereo configuration file'
    )

    urdf_file_arg = DeclareLaunchArgument(
        'urdf_file',
        default_value='',
        description='Path to robot URDF for forward kinematics'
    )

    # Stereo depth node
    stereo_node = Node(
        package='wrist_stereo',
        executable='stereo_depth_node',
        name='stereo_depth_node',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'urdf_file': LaunchConfiguration('urdf_file'),
            'left_image_topic': '/wrist_left/image_raw',
            'right_image_topic': '/wrist_right/image_raw',
            'left_intrinsics': os.path.join(pkg_share, 'config', 'left_intrinsics.yaml'),
            'right_intrinsics': os.path.join(pkg_share, 'config', 'right_intrinsics.yaml'),
            'left_hand_eye': os.path.join(pkg_share, 'config', 'left_hand_eye.yaml'),
            'right_hand_eye': os.path.join(pkg_share, 'config', 'right_hand_eye.yaml'),
            'base_link': 'base_link',
            'left_wrist_link': 'link6l',
            'right_wrist_link': 'link6',
            'min_depth': 0.05,
            'max_depth': 0.5,
            'publish_points': True,
        }]
    )

    return LaunchDescription([
        config_file_arg,
        urdf_file_arg,
        stereo_node,
    ])
