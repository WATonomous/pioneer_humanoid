import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = os.path.join(get_package_share_directory('perception'), 'config', 'perception_config.yaml')
    realsense_launch = PathJoinSubstitution([FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])

    use_detector = LaunchConfiguration('use_detector')
    use_dummy = LaunchConfiguration('use_dummy')
    use_camera = LaunchConfiguration('use_camera')
    sim_time = {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_detector', default_value='false',
            description='Start the YOLO shuttlecock detector (needs ultralytics and /data/weights)'
        ),
        DeclareLaunchArgument(
            'use_dummy', default_value='false',
            description='Start the dummy publisher and dummy TrackNet nodes; leave false when using the real camera and detector'
        ),
        DeclareLaunchArgument(
            'use_camera', default_value='true',
            description='Start the D455 driver; false when replaying a bag'
        ),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true when replaying a bag with --clock'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Log level for the detector and dummy publisher node'
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch),
            condition=IfCondition(use_camera),
            launch_arguments={
                'enable_color': 'true',
                'enable_depth': 'true',
                'rgb_camera.color_profile': '640,480,30',
                'depth_module.depth_profile': '640,480,30',
                'align_depth.enable': 'true',
                'enable_sync': 'true',
            }.items()
        ),
        Node(
            package='perception',
            executable='shuttlecock_detector_node',
            name='shuttlecock_detector_node',
            output='screen',
            parameters=[config, sim_time],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            condition=IfCondition(use_detector),
        ),
        Node(
            package='perception',
            executable='shuttlecock_localizer_node',
            name='shuttlecock_localizer_node',
            output='screen',
            parameters=[config, sim_time]
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
            ],
            condition=IfCondition(use_dummy),
        ),
        Node(
            package='perception',
            executable='dummy_tracknet_output',
            name='dummy_tracknet_output',
            output='screen',
            condition=IfCondition(use_dummy),
        ),
        Node(
            package='perception',
            executable='trajectory_EKF_prediction_node',
            name='trajectory_EKF_prediction_node',
            output='screen',
        ),
    ])
