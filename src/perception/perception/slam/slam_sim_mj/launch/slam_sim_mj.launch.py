"""RTAB-Map against the simulated D455 published by mj_slam_publisher.py.

    source /opt/ros/jazzy/setup.bash
    ros2 launch src/perception/perception/slam/slam_sim_mj/launch/slam_sim_mj.launch.py

Launched by path rather than by package name, so it needs no ament workspace built.

This is deliberately the SAME graph as the real-hardware procedure in
../slam-quickstart.md -- camera -> madgwick -> rtabmap, with the same odometry tuning
-- so that a fix proven here is a fix for the real rig. The publisher synthesises gyro
and accel, so the IMU half of that graph is exercised rather than stubbed out.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _repo_root() -> Path:
    """Repo root, found by walking up to the .git marker.

    NOT parents[N]. A fixed depth resolves silently to the wrong directory as soon as
    the package moves, and it has moved twice already (autonomy/ -> src/, and a
    proposed deeper home under perception/). Wrong root means the database is written
    somewhere unexpected -- and this launch file passes "-d", which deletes the
    database it is pointed at.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(f"no repo root (.git) above {__file__}")


def generate_launch_description() -> LaunchDescription:
    use_rviz = LaunchConfiguration("rviz")
    rtabmap_args = LaunchConfiguration("rtabmap_args")
    camera_pitch = LaunchConfiguration("camera_pitch")
    camera_height = LaunchConfiguration("camera_height")

    # NOT ~/.ros/rtabmap.db, which is rtabmap's default. That path holds the
    # real-hardware demo recording, and this launch file passes "-d" -- which deletes
    # the database on start. Sim output goes to the repo's outputs/ instead, where it
    # is also easy to find and diff against a previous run.
    default_db = _repo_root() / "outputs" / "slam_mj" / "rtabmap.db"
    default_db.parent.mkdir(parents=True, exist_ok=True)

    rtabmap_launch = PythonLaunchDescriptionSource(
        [FindPackageShare("rtabmap_launch"), "/launch/rtabmap.launch.py"]
    )

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("database_path", default_value=str(default_db)),
        DeclareLaunchArgument(
            "camera_height", default_value="1.5",
            description="Camera height above the FLOOR (m). Must match the publisher's "
                        "--height_m. If this is 0, base_link sits at eye level, the "
                        "floor lands at z = -1.5 in the map frame, and every Grid/* "
                        "height threshold is measured from the wrong datum -- which "
                        "produces a black blob with no free space and no border.",
        ),
        DeclareLaunchArgument(
            "camera_pitch", default_value="0.0",
            description="Camera depression angle (rad). 0 for --path corridor, which "
                        "gazes level. For --path orbit it is "
                        "atan2(height_m - target_height, radius) = 0.3805 at the "
                        "defaults -- and NOTE the sign of that is untested, so if the "
                        "orbit map comes out tilted, try -0.3805 before looking "
                        "anywhere else.",
        ),
        DeclareLaunchArgument(
            "rtabmap_args",
            # These Grid/* parameters are what turn the 2D map from an unreadable blob
            # into a floor plan. Without them the occupancy grid has NO height
            # filtering, so ceiling and high-wall points project straight down as
            # obstacles and the rays fan out into radial streaks with no outer border.
            #
            #   RayTracing         mark the space BETWEEN camera and hit as free. This
            #                      is what draws a border -- without it you get
            #                      obstacles floating in unknown space.
            #   3D false           accumulate directly into a 2D grid.
            #   MaxObstacleHeight  ignore the ceiling.
            #   MaxGroundHeight    below this is floor, not obstacle.
            #   NormalsSegmentation false  use those height cutoffs instead of surface
            #                      normals, which mis-classify on the clean, low-noise
            #                      depth a renderer produces.
            #   RangeMax           long depth rays are the noisiest and cause most of
            #                      the streaking.
            #
            # These are verified correct and are recorded in the database's
            # Info.parameters. If the 2D map looks wrong, measure odometry z-drift
            # with check_map.py before touching them -- the usual cause is the datum
            # they measure against, not these thresholds.
            default_value=(
                "-d "
                "--Grid/RayTracing true "
                "--Grid/3D false "
                "--Grid/MaxObstacleHeight 2.0 "
                "--Grid/MaxGroundHeight 0.15 "
                "--Grid/NormalsSegmentation false "
                "--Grid/RangeMax 6.0 "
                "--Grid/CellSize 0.05"
            ),
            description="'-d' starts a fresh map; pass rtabmap_args:='' to keep the "
                        "existing one.",
        ),

        # Same node, same parameters as Terminal 2 of the real-hardware quickstart.
        # It takes raw gyro+accel and outputs an orientation quaternion; RTAB-Map uses
        # that purely as a gravity prior, which is what pins roll and pitch and stops
        # the odometry sinking in z.
        Node(
            package="imu_filter_madgwick",
            executable="imu_filter_madgwick_node",
            name="imu_filter",
            parameters=[{"use_mag": False, "publish_tf": False}],
            remappings=[
                ("/imu/data_raw", "/camera/camera/imu"),
                ("/imu/data", "/imu/data_filtered"),
            ],
            output="screen",
        ),

        # The publisher stamps images and IMU in the optical frame, so this fixed
        # rotation is what connects them to the rest of the TF tree. Without it
        # RTAB-Map receives data in a frame it cannot resolve and the map comes out
        # sideways. (x y z qx qy qz qw) -- the standard body->optical rotation:
        # optical +Z is body +X, optical +X is body -Y, optical +Y is body -Z.
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="camera_link_to_optical",
            arguments=["0", "0", "0", "-0.5", "0.5", "-0.5", "0.5",
                       "camera_link", "camera_color_optical_frame"],
            output="screen",
        ),

        # base_link is the gravity-levelled frame the map is built in. Using
        # camera_link (the camera itself) as frame_id makes RTAB-Map initialise its
        # odom frame at the first camera pose, so the whole map inherits the camera's
        # tilt and the 2D projection comes out skewed by exactly that angle.
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="base_link_to_camera_link",
            arguments=["--x", "0", "--y", "0", "--z", camera_height,
                       "--roll", "0", "--pitch", camera_pitch, "--yaw", "0",
                       "--frame-id", "base_link", "--child-frame-id", "camera_link"],
            output="screen",
        ),

        IncludeLaunchDescription(
            rtabmap_launch,
            launch_arguments={
                "database_path": LaunchConfiguration("database_path"),
                "rgb_topic": "/camera/camera/color/image_raw",
                "depth_topic": "/camera/camera/depth/image_rect_raw",
                "camera_info_topic": "/camera/camera/color/camera_info",
                "frame_id": "base_link",
                "imu_topic": "/imu/data_filtered",
                # Safe here BECAUSE there is an IMU. With no IMU publishing this
                # blocks forever and presents as a hang rather than an error.
                "wait_imu_to_init": "true",
                # Colour, depth and camera_info come from one render of one camera and
                # carry identical stamps, so there is nothing to approximately match.
                "approx_sync": "false",
                "rgbd_sync": "true",
                # approx_rgbd_sync is a SEPARATE argument from approx_sync and defaults
                # to true -- setting only approx_sync leaves the rgbd_sync node
                # approximately matching streams that are already exactly stamped, and
                # it then pairs colour with depth from a frame or two earlier.
                "approx_rgbd_sync": "false",
                "rtabmap_viz": "false",
                "rviz": use_rviz,
                # Same odometry tuning as the real-camera quickstart. Renders are
                # cleaner than a real sensor, but corner rotation is still where
                # frame-to-frame overlap is smallest and tracking fails first.
                "odom_args": "--Vis/MinInliers 10 --Vis/MaxFeatures 1500 --Odom/ResetCountdown 5",
                "rtabmap_args": rtabmap_args,
            }.items(),
        ),
    ])
