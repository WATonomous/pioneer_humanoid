"""Tests for terrain_tools.bag_reader.
"""
import numpy as np
import pytest
from mcap.writer import CompressionType
from mcap_ros2.writer import Writer

from terrain_tools.bag_reader import read_bag
from slam_core.types.pose import is_pose

HEADER = """\
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

POINTCLOUD2_DEF = """\
std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
================================================================================
MSG: sensor_msgs/PointField
uint8 INT8=1
uint8 UINT8=2
uint8 INT16=3
uint8 UINT16=4
uint8 INT32=5
uint8 UINT32=6
uint8 FLOAT32=7
uint8 FLOAT64=8
string name
uint32 offset
uint8 datatype
uint32 count
""" + HEADER

TF_DEF = """\
geometry_msgs/TransformStamped[] transforms
================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform
================================================================================
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
""" + HEADER

JOINT_STATE_DEF = """\
std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
""" + HEADER

FLOAT32 = 7


def header(sec, nanosec, frame):
    return {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": frame}


def cloud_msg(points, frame="livox_frame", sec=1, extra_field=False):
    """PointCloud2 of float32 x y z. With extra_field, an intensity float sits after z,
    like real lidar clouds, so point_step is bigger than the xyz part."""
    points = np.asarray(points, dtype=np.float32)
    fields = [{"name": n, "offset": 4 * i, "datatype": FLOAT32, "count": 1}
              for i, n in enumerate("xyz")]
    if extra_field:
        fields.append({"name": "intensity", "offset": 12, "datatype": FLOAT32, "count": 1})
        intensity = np.full((len(points), 1), 7.0, dtype=np.float32)
        points = np.hstack([points, intensity])
    step = 4 * points.shape[1]
    return {
        "header": header(sec, 0, frame),
        "height": 1, "width": len(points),
        "fields": fields, "is_bigendian": False,
        "point_step": step, "row_step": step * len(points),
        "data": points.tobytes(), "is_dense": True,
    }


def tf_msg(parent, child, t, q, sec=1):
    return {"transforms": [{
        "header": header(sec, 0, parent),
        "child_frame_id": child,
        "transform": {
            "translation": dict(zip("xyz", t)),
            "rotation": dict(zip("xyzw", q)),
        },
    }]}


def joints_msg(names, positions, sec=1):
    return {"header": header(sec, 0, ""), "name": names, "position": positions,
            "velocity": [], "effort": []}


@pytest.fixture
def write_bag(tmp_path):
    """write_bag([(topic, kind, message, log_time_ns), ...]) -> path of a new .mcap."""
    defs = {"cloud": ("sensor_msgs/msg/PointCloud2", POINTCLOUD2_DEF),
            "tf": ("tf2_msgs/msg/TFMessage", TF_DEF),
            "joints": ("sensor_msgs/msg/JointState", JOINT_STATE_DEF)}

    def _write(records):
        path = tmp_path / "test.mcap"
        with open(path, "wb") as f:
            writer = Writer(f, compression=CompressionType.NONE)
            schemas = {}
            for topic, kind, message, log_time in records:
                if kind not in schemas:
                    schemas[kind] = writer.register_msgdef(*defs[kind])
                writer.write_message(topic=topic, schema=schemas[kind], message=message,
                                     log_time=log_time, publish_time=log_time)
            writer.finish()
        return path

    return _write


# Point clouds -----------------------------------------------------------------


def test_cloud_points_come_back_exactly(write_bag):
    pts = [[1.0, 2.0, 3.0], [-0.5, 0.25, -0.9]]
    path = write_bag([("/livox/lidar", "cloud", cloud_msg(pts), 10)])
    [m] = list(read_bag(path))
    assert m.kind == "cloud" and m.topic == "/livox/lidar"
    assert m.data.shape == (2, 3)
    assert np.allclose(m.data, pts)


def test_cloud_ignores_extra_fields(write_bag):
    pts = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    path = write_bag([("/livox/lidar", "cloud", cloud_msg(pts, extra_field=True), 10)])
    [m] = list(read_bag(path))
    assert np.allclose(m.data, pts)


def test_cloud_drops_nan_points(write_bag):
    pts = [[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [4.0, 5.0, 6.0]]
    path = write_bag([("/livox/lidar", "cloud", cloud_msg(pts), 10)])
    [m] = list(read_bag(path))
    assert np.allclose(m.data, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_cloud_keeps_header_frame_and_stamp(write_bag):
    path = write_bag([("/livox/lidar", "cloud", cloud_msg([[0, 0, 0]], frame="livox_frame", sec=5), 10)])
    [m] = list(read_bag(path))
    assert m.frame_id == "livox_frame"
    assert m.stamp_ns == 5_000_000_000


# TF ---------------------------------------------------------------------------


def test_tf_identity_rotation_keeps_translation(write_bag):
    path = write_bag([("/tf", "tf", tf_msg("odom", "base_link", (1.0, 2.0, 3.0), (0, 0, 0, 1)), 10)])
    [m] = list(read_bag(path))
    [e] = m.data
    assert (e.parent, e.child) == ("odom", "base_link")
    assert np.allclose(e.T, [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])


def test_tf_yaw_90_turns_forward_into_left(write_bag):
    s = np.sqrt(0.5)   # quaternion (x, y, z, w) for +90 deg about z
    path = write_bag([("/tf", "tf", tf_msg("odom", "base_link", (0, 0, 0), (0, 0, s, s)), 10)])
    [e] = next(read_bag(path)).data
    assert np.allclose(e.T[:3, :3] @ [1, 0, 0], [0, 1, 0])
    assert is_pose(e.T)


# Joint states -------------------------------------------------------------------


def test_joint_names_map_to_positions(write_bag):
    path = write_bag([("/joint_states", "joints",
                       joints_msg(["left_knee", "right_knee"], [0.3, -0.1]), 10)])
    [m] = list(read_bag(path))
    assert m.kind == "joints"
    assert m.data == {"left_knee": 0.3, "right_knee": -0.1}


# Reading the bag ------------------------------------------------------------------


def test_messages_come_out_in_log_time_order(write_bag):
    path = write_bag([
        ("/joint_states", "joints", joints_msg(["knee"], [0.0]), 30),
        ("/livox/lidar", "cloud", cloud_msg([[0, 0, 0]]), 10),
        ("/tf", "tf", tf_msg("odom", "base_link", (0, 0, 0), (0, 0, 0, 1)), 20),
    ])
    assert [m.log_time_ns for m in read_bag(path)] == [10, 20, 30]


def test_topic_filter_reads_only_those_topics(write_bag):
    path = write_bag([
        ("/livox/lidar", "cloud", cloud_msg([[0, 0, 0]]), 10),
        ("/tf", "tf", tf_msg("odom", "base_link", (0, 0, 0), (0, 0, 0, 1)), 20),
    ])
    assert [m.topic for m in read_bag(path, topics=["/tf"])] == ["/tf"]
