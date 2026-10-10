"""Scripted camera trajectory for the SLAM sandbox scene.

Deliberately free of simulator imports: the rotation conventions here are easy to
get backwards, and a wrong sign is invisible in a headless run — it shows up much
later as a map that is silently upside down or mirrored. Keeping this module pure
numpy means `tests/test_orbit_path.py` proves it on any machine in milliseconds,
rather than behind a renderer and a full pipeline run.

Everything here uses the ROS optical convention, matching what
`Camera.set_world_poses(..., convention="ros")` expects and what the images are
stamped with (`camera_color_optical_frame`):

    +Z  boresight (out of the lens)
    +X  right in the image
    +Y  down in the image

Note this is NOT the ROS body convention (x-forward, y-left, z-up) that
`camera_link` uses — the fixed rotation between the two is published by
`src/perception/perception/slam/slam_sim_mj/launch/slam_sim_mj.launch.py`, not applied here.

Quaternions are (w, x, y, z), matching MuJoCo's ordering.
"""

from __future__ import annotations

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])


def look_at_quat(
    eye: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray = WORLD_UP,
) -> np.ndarray:
    """Orientation for an optical frame at `eye` looking at `target`.

    Returns a unit quaternion (w, x, y, z) rotating camera-local axes into world
    axes. Roll is resolved by keeping the image's "up" as close to `world_up` as
    possible, which is what makes the rendered image look upright.

    There is no `pitch` argument on purpose. Pitch is the classic trap here:
    rotating about +Y by a positive angle swings the boresight *down*, so an
    argument named "pitch" reads as the opposite of what it does. Aiming at an
    explicit point has no such ambiguity.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    world_up = np.asarray(world_up, dtype=float)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("look_at_quat: eye and target coincide, direction undefined")
    forward = forward / norm

    # Image "down" is world-down projected perpendicular to the boresight.
    down = -world_up
    down = down - np.dot(down, forward) * forward
    down_norm = np.linalg.norm(down)
    if down_norm < 1e-6:
        # Boresight is parallel to world up (looking straight up or straight down):
        # roll is genuinely undefined, so pick a deterministic fallback rather than
        # dividing by ~0 and emitting NaNs into the pose stream.
        fallback = np.array([1.0, 0.0, 0.0])
        down = fallback - np.dot(fallback, forward) * forward
        down_norm = np.linalg.norm(down)
    down = down / down_norm

    # Right-handed optical frame: x_c cross y_c == z_c, i.e. right = down x forward.
    right = np.cross(down, forward)

    # Columns map camera-local axes into world axes.
    rotation = np.column_stack((right, down, forward))
    return _quat_from_matrix(rotation)


def orbit_pose(
    t: float,
    center: np.ndarray,
    radius: float = 2.0,
    height: float = 1.4,
    period: float = 60.0,
    target: np.ndarray | None = None,
    phase: float = 0.0,
    look_ahead: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose on a circular orbit at time `t`.

    Two aiming modes, and the choice changes what the run is *for*:

    `look_ahead == 0` (default) — **inward**, staring at `target` (the room centre).
    Every lap re-observes the same features from the same viewpoints, so loop
    closure either fires or visibly does not, unlike a wandering path where a
    failure to close is ambiguous. This is a SLAM *test* trajectory.

    `look_ahead > 0` — **forward**, aiming at a point that far ahead (radians) along
    the orbit, at eye level. This is what a person walking with a handheld camera
    actually does: you look where you are going, sweeping walls as they come into
    view rather than fixating on one spot. Realistic, and it maps the walls far
    better — but loop closure is genuinely harder, because a place is only
    recognised when you approach it along the same heading.

    NOTE: forward mode gazes level, so the launch file's gravity-levelling
    `camera_pitch` must be **0**. The default 0.3805 rad exists to cancel the
    downward tilt of inward mode; leaving it set here tips the whole map.

    `t` must be *simulation* time, not wall-clock. If rendering runs slower than
    real time, a wall-clock path teleports the camera between frames and visual
    odometry loses tracking immediately.

    Returns (position, quaternion_wxyz).
    """
    center = np.asarray(center, dtype=float)

    theta = phase + 2.0 * np.pi * (t / period)
    eye = center + np.array(
        [radius * np.cos(theta), radius * np.sin(theta), height]
    )

    if look_ahead:
        # Aim at where we will be, not at the middle. Reuses look_at_quat rather
        # than building a heading quaternion by hand: its right-handedness and
        # image-upright guarantees are already covered by tests.
        ahead = theta + look_ahead
        aim = center + np.array(
            [radius * np.cos(ahead), radius * np.sin(ahead), height]
        )
    else:
        aim = center if target is None else np.asarray(target, dtype=float)

    return eye, look_at_quat(eye, aim)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion. Columns are the mapped axes."""
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """(w, x, y, z) from a rotation matrix, via the branch with the largest divisor.

    The branching is not premature optimisation: the single-formula version loses
    precision (and can take a sqrt of a negative) when the trace is near -1.
    """
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _rect_point(s: float, half_x: float, half_y: float, corner: float) -> np.ndarray:
    """Point at loop fraction `s` (wraps) around a rounded rectangle, xy only.

    A rectangle, not a circle, because a corridor circuit is what a person actually
    walks: long straights that build a floor plan, and corners that force the ~90
    degree heading changes loop closure has to survive. Corners are rounded so
    heading stays continuous -- a hard corner steps yaw discontinuously between
    frames and visual odometry loses tracking exactly there.

    Counter-clockwise, starting at the bottom of the +x side.
    """
    hx, hy = float(half_x), float(half_y)
    r = min(float(corner), hx, hy)

    lx = 2.0 * (hx - r)          # each horizontal straight
    ly = 2.0 * (hy - r)          # each vertical straight
    arc = 0.5 * np.pi * r        # each quarter turn
    perim = 2.0 * (lx + ly) + 4.0 * arc

    d = (float(s) % 1.0) * perim

    # +x side, heading +y
    if d < ly:
        return np.array([hx, -(hy - r) + d])
    d -= ly
    if d < arc:                                   # top-right corner, 0 -> 90 deg
        a = d / r
        return np.array([hx - r + r * np.cos(a), hy - r + r * np.sin(a)])
    d -= arc
    if d < lx:                                    # top side, heading -x
        return np.array([hx - r - d, hy])
    d -= lx
    if d < arc:                                   # top-left corner, 90 -> 180
        a = np.pi / 2 + d / r
        return np.array([-(hx - r) + r * np.cos(a), hy - r + r * np.sin(a)])
    d -= arc
    if d < ly:                                    # -x side, heading -y
        return np.array([-hx, hy - r - d])
    d -= ly
    if d < arc:                                   # bottom-left corner, 180 -> 270
        a = np.pi + d / r
        return np.array([-(hx - r) + r * np.cos(a), -(hy - r) + r * np.sin(a)])
    d -= arc
    if d < lx:                                    # bottom side, heading +x
        return np.array([-(hx - r) + d, -hy])
    d -= lx
    a = 3 * np.pi / 2 + d / r                     # bottom-right corner, 270 -> 360
    return np.array([hx - r + r * np.cos(a), -(hy - r) + r * np.sin(a)])


def rect_perimeter(half_x: float, half_y: float, corner: float) -> float:
    """Path length of one lap, so runs can be described in metres and m/s."""
    hx, hy = float(half_x), float(half_y)
    r = min(float(corner), hx, hy)
    return 2.0 * (2.0 * (hx - r) + 2.0 * (hy - r)) + 2.0 * np.pi * r


def corridor_pose(
    t: float,
    center: np.ndarray,
    half_x: float = 6.0,
    half_y: float = 4.0,
    corner: float = 1.2,
    height: float = 1.5,
    period: float = 90.0,
    look_ahead: float = 0.04,
    scan_yaw: float = 0.0,
    scan_pitch: float = 0.0,
    scan_period: float = 11.0,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose walking a rounded-rectangle corridor circuit, looking ahead.

    This is the sim analogue of walking a building loop once with a handheld
    camera: long straights, four corners, back to the start so loop closure has a
    real revisit to find.

    `look_ahead` is a fraction of the loop to aim in front of the walker.
    `scan_yaw` / `scan_pitch` (radians) add a slow head sweep on top of the
    heading. A person does not walk with their gaze nailed forward -- they glance
    around, and that is what fills in walls, floor and ceiling instead of a narrow
    band at eye level. The two use different periods so the motion does not look
    mechanically circular.

    Returns (position, quaternion_wxyz).
    """
    center = np.asarray(center, dtype=float)
    s = phase + (t / period)

    xy = _rect_point(s, half_x, half_y, corner)
    eye = center + np.array([xy[0], xy[1], height])

    ahead_xy = _rect_point(s + look_ahead, half_x, half_y, corner)
    aim = center + np.array([ahead_xy[0], ahead_xy[1], height])

    if scan_yaw or scan_pitch:
        forward = aim - eye
        dist = np.linalg.norm(forward[:2])
        heading = np.arctan2(forward[1], forward[0])
        # Different periods (and a phase offset) keep yaw and pitch from tracing a
        # circle, which reads as a mechanical wobble rather than someone looking around.
        yaw = heading + scan_yaw * np.sin(2.0 * np.pi * t / scan_period)
        pitch = scan_pitch * np.sin(2.0 * np.pi * t / (scan_period * 1.7) + 1.1)
        aim = eye + np.array([
            dist * np.cos(yaw),
            dist * np.sin(yaw),
            dist * np.tan(pitch),
        ])

    return eye, look_at_quat(eye, aim)
