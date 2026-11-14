"""Does preprocess() put points where the surfaces really are in odom?

A made-up scene (a floor and a wall) is rendered into a depth image from a known camera
pose, so the right answer is known exactly and no dataset is needed: these run in CI.
The same check on real TUM data is in test_tum.py.
"""
import numpy as np
from slam_core.types import Frame

from terrain_core.preprocess.pipeline import PreprocessConfig, preprocess

WALL_X = 2.5                     # the scene: floor at z = 0, wall at x = 2.5 m (odom: x forward, z up)
K = np.array([[100.0, 0, 40.0], [0, 100.0, 30.0], [0, 0, 1]])
H, W = 60, 80
CONFIG = PreprocessConfig(z_min=0.3, z_max=3.0, voxel_size=0.02)


def camera_pose(x=0.0, y=0.0, height=1.2, pitch_deg=30.0):
    """T_odom_cam for a camera at (x, y, height) looking along +x, tilted down by pitch_deg."""
    # Camera optical axes in odom when looking along +x: x right = -y, y down = -z, z forward = +x
    R_level = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    p = np.radians(pitch_deg)
    R_pitch = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])   # tilt forward down
    T = np.eye(4)
    T[:3, :3] = R_pitch @ R_level
    T[:3, 3] = [x, y, height]
    return T


def render_depth(T_odom_cam):
    """Depth image (metres) of the floor + wall scene, seen by a camera at T_odom_cam."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # Ray through each pixel in camera coordinates, scaled so its z (depth) is 1
    ray_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], axis=-1)
    ray = ray_cam @ T_odom_cam[:3, :3].T                 # same rays, in odom directions
    origin = T_odom_cam[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        to_floor = -origin[2] / ray[..., 2]              # how many "depth = 1" steps to reach z = 0
        to_wall = (WALL_X - origin[0]) / ray[..., 0]     # ... and to reach x = WALL_X
    to_floor[to_floor <= 0] = np.inf                     # behind the camera: never hit
    to_wall[to_wall <= 0] = np.inf
    depth = np.minimum(to_floor, to_wall)                # the first surface each ray hits
    depth[~np.isfinite(depth)] = 0.0                     # no hit: no reading, like the real camera
    return depth.astype(np.float32)


def distance_to_scene(points):
    """For each point, how far it is from the nearest true surface (floor or wall), in metres."""
    return np.minimum(np.abs(points[:, 2]), np.abs(points[:, 0] - WALL_X))


def cloud(T_render, T_given):
    """Render from the true pose T_render, then preprocess using T_given (maybe a wrong pose)."""
    frame = Frame(stamp=1.0, image=np.zeros((H, W), np.uint8), depth=render_depth(T_render), K=K)
    return preprocess(frame, T_given, CONFIG).points


def test_right_pose_puts_points_on_the_surfaces():
    T = camera_pose()
    points = cloud(T, T)
    assert len(points) > 100
    assert distance_to_scene(points).max() < 1e-3        # float32 rounding only


def test_two_camera_poses_agree():
    # Same scene from two places: both clouds land on the same surfaces in odom
    for T in [camera_pose(), camera_pose(x=0.4, y=-0.3, height=1.0, pitch_deg=40.0)]:
        assert distance_to_scene(cloud(T, T)).max() < 1e-3


def test_wrong_pose_moves_points_off_the_surfaces():
    # Controls: if the check above could not fail, it would prove nothing.
    T = camera_pose()
    shifted = camera_pose(x=0.05, y=0.05, height=1.25)  # 5 cm off in every direction
    tilted = camera_pose(pitch_deg=32.0)                # 2 degrees off
    assert np.median(distance_to_scene(cloud(T, shifted))) > 0.03
    assert np.median(distance_to_scene(cloud(T, tilted))) > 0.03
