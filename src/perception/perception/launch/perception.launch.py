import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    config = os.path.join(get_package_share_directory('perception'), 'config', 'perception_config.yaml')
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_tracknet', default_value='false',
            description='Start the TrackNet detector node (needs a GPU and models/TrackNet_best.pt)'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Log level for the dummy publisher node'
        ),
        Node(
            package='perception',
            executable='dummy_publisher_node',
            name='dummy_publisher_node',
            output='screen',
            arguments=[
                '--ros-args',
                '--log-level',
                LaunchConfiguration('log_level')
            ],
            remappings=[
                # Add any topic remappings here if needed
            ]
        ),
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_node',
            output='screen',
            parameters=[{
                'depth_module.depth_profile': '640x480x30',
                'pointcloud.enable': True,
                'enable_rgbd': True,
                'enable_sync': True,
                'align_depth.enable': True,
                'enable_color': True,
                'enable_depth': True
            }]
        ),
        Node(
            package='perception',
            executable='tracknet_node',
            name='tracknet_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_tracknet')),
        ),
        Node(
            package='perception',
            executable='dummy_tracknet_output',
            name='dummy_tracknet_output',
            output='screen',
        ),
        Node(
            package='perception',
            executable='trajectory_EKF_prediction_node',
            name='trajectory_EKF_prediction_node',
            output='screen',
            parameters=[config],
        ),
    ])
