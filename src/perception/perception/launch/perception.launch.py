from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
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
            executable='dummy_tracknet_output',
            name='dummy_tracknet_output',
            output='screen',
        ),
        Node(
            package='perception',
            executable='shuttlecock_localizer_node',
            name='shuttlecock_localizer_node',
            output='screen',
        ),
        Node(
            package='perception',
            executable='pose_estimation_node',
            name='pose_estimation_node',
            output='screen',
        ),
        Node(
            package='perception',
            executable='trajectory_EKF_prediction_node',
            name='trajectory_EKF_prediction_node',
            output='screen',
            remappings=[
                ('/shuttle_states', '/perception/shuttlecock/detections_3d'),
            ]
        ),
    ])
