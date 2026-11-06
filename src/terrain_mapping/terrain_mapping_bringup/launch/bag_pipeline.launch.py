"""Replay a D455 bag and build a GPU elevation map from it, all on the bag's clock.

    depth image ──> depth_image_proc ──> /camera/depth/points ──> elevation_mapping_cupy
    color + aligned depth + IMU (Madgwick) ──> rtabmap rgbd_odometry ──> TF odom -> camera_link

Usage:
    ros2 launch terrain_mapping_bringup bag_pipeline.launch.py
    ros2 launch terrain_mapping_bringup bag_pipeline.launch.py bag:='/bags/001-D455(360_turn_v1)'
"""
import math
import os

import numpy as np
import rosbag2_py
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage

BAG_TOPICS = [
    '/tf_static',
    '/camera/camera/imu',
    '/camera/camera/depth/image_rect_raw',
    '/camera/camera/depth/camera_info',
    '/camera/camera/color/image_raw',
    '/camera/camera/color/camera_info',
    '/camera/camera/aligned_depth_to_color/image_raw',
    '/camera/camera/aligned_depth_to_color/camera_info',
]
SIM_TIME = {'use_sim_time': True}
IMU_TOPIC = '/camera/camera/imu'
STILL_SAMPLES = 50  # 0.25 s at 200 Hz; the camera is still at the start of each bag


def quaternion_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_tilt(bag: str) -> tuple:
    """Roll and pitch [rad] of camera_link from gravity, averaged over the bag's first IMU samples.

    rgbd_odometry (rtabmap 0.23.13) aligns odom with gravity only from IMU samples older than its
    first frame. A frame arrives ~38 ms after capture and an IMU sample ~1 ms after, so the first
    frame always predates the first IMU sample and odom keeps the camera's starting tilt (measured
    5.6 deg on bag 002). Passing the tilt as initial_pose aligns odom from the first frame.
    """
    if not os.path.isdir(bag):
        raise FileNotFoundError(f'bag folder not found: {bag} (the container mounts TERRAIN_BAGS_DIR at /bags)')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=''), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/tf_static', IMU_TOPIC]))
    parents, accel, imu_frame = {}, [], None
    while reader.has_next() and (len(accel) < STILL_SAMPLES or not parents):
        topic, data, _ = reader.read_next()
        if topic == '/tf_static':
            for t in deserialize_message(data, TFMessage).transforms:
                parents[t.child_frame_id] = (t.header.frame_id, quaternion_matrix(t.transform.rotation))
        elif len(accel) < STILL_SAMPLES:
            msg = deserialize_message(data, Imu)
            imu_frame = msg.header.frame_id
            accel.append((msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z))

    rotation, frame = np.eye(3), imu_frame
    while frame != 'camera_link':
        frame, parent_rotation = parents[frame]
        rotation = parent_rotation @ rotation
    # Median, not mean: bag 001 starts with two glitched accelerometer samples while the camera is still.
    up = rotation @ np.median(accel, axis=0)  # an accelerometer at rest reads +g along "up"
    up /= np.linalg.norm(up)
    return math.atan2(up[1], up[2]), math.asin(-up[0])


def odometry_node(context):
    roll, pitch = camera_tilt(LaunchConfiguration('bag').perform(context))
    return [Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        parameters=[{
            'frame_id': 'camera_link',
            'odom_frame_id': 'odom',
            'publish_tf': True,
            'initial_pose': f'0 0 0 {roll:.5f} {pitch:.5f} 0',  # x y z roll pitch yaw
            'wait_imu_to_init': True,
            'approx_sync': False,  # aligned depth carries the color frame's exact stamp
            **SIM_TIME,
        }],
        remappings=[
            ('rgb/image', '/camera/camera/color/image_raw'),
            ('rgb/camera_info', '/camera/camera/color/camera_info'),
            ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('imu', '/imu/data'),
        ],
    )]


def generate_launch_description():
    bringup_share = get_package_share_directory('terrain_mapping_bringup')
    emc_share = get_package_share_directory('elevation_mapping_cupy')
    config = os.path.join(bringup_share, 'config')

    bag = LaunchConfiguration('bag')
    rate = LaunchConfiguration('rate')

    # TODO(robot): placeholder. The bags have no robot, so base_link is pinned to the camera.
    # On the robot, odom -> base_link comes from the state estimator and base_link -> camera_link
    # from the URDF camera mount; delete this publisher then.
    base_link = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_placeholder',
        arguments=['--frame-id', 'camera_link', '--child-frame-id', 'base_link'],
        parameters=[SIM_TIME],
    )

    # The D455 IMU publishes no orientation; rgbd_odometry needs one to use the IMU for motion prediction.
    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        parameters=[{'use_mag': False, 'publish_tf': False, 'world_frame': 'enu', **SIM_TIME}],
        remappings=[('imu/data_raw', '/camera/camera/imu'), ('imu/data', '/imu/data')],
    )

    point_cloud = ComposableNodeContainer(
        name='point_cloud_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='depth_image_proc',
                plugin='depth_image_proc::PointCloudXyzNode',
                name='point_cloud_xyz',
                remappings=[
                    ('image_rect', '/camera/camera/depth/image_rect_raw'),
                    # Humble derives the info topic from the unremapped name ('/camera_info'), so remap it too.
                    ('camera_info', '/camera/camera/depth/camera_info'),
                    ('points', '/camera/depth/points'),
                ],
                parameters=[SIM_TIME],
            ),
        ],
    )

    elevation_mapping = Node(
        package='elevation_mapping_cupy',
        executable='elevation_mapping_node.py',
        name='elevation_mapping_node',
        output='screen',
        parameters=[
            os.path.join(emc_share, 'config', 'core', 'core_param.yaml'),
            os.path.join(config, 'd455_elevation.yaml'),
            {'plugin_config_file': os.path.join(config, 'd455_plugins.yaml'), **SIM_TIME},
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(bringup_share, 'rviz', 'terrain_mapping.rviz')],
        parameters=[SIM_TIME],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    # Start playback once the nodes are up so they see /tf_static and the first frames.
    play = TimerAction(period=5.0, actions=[ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag, '--clock', '--rate', rate, '--topics', *BAG_TOPICS],
        output='screen',
    )])

    return LaunchDescription([
        DeclareLaunchArgument('bag', default_value='/bags/002-D455(360_turn_office)'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        base_link,
        imu_filter,
        OpaqueFunction(function=odometry_node),
        point_cloud,
        elevation_mapping,
        rviz,
        play,
    ])
