"""Read terrain mapping inputs from an MCAP bag, without ROS.

Turns ROS 2 messages into plain numpy so the offline runner (TR-13) can feed
them to terrain_core:
  sensor_msgs/PointCloud2  -> (N, 3) float64 xyz in the message's frame
  tf2_msgs/TFMessage       -> one 4x4 T_parent_child per transform
  sensor_msgs/JointState   -> {joint name: position}

Messages come out in the order they were recorded (log time).
"""
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from slam_core.types.pose import make_T, quat_to_rot

POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"
TF = "tf2_msgs/msg/TFMessage"
JOINT_STATE = "sensor_msgs/msg/JointState"

# sensor_msgs/PointField datatype codes -> numpy types
_POINTFIELD_TYPES = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


@dataclass
class BagMessage:
    topic: str
    kind: str          # "cloud", "tf" or "joints"
    log_time_ns: int   # when the recorder logged it
    stamp_ns: int      # the message's own header stamp (TF: the first transform's)
    frame_id: str      # header frame (TF: "" since each transform has its own)
    data: Any          # cloud: (N, 3) array, tf: list of TfEntry, joints: dict


@dataclass
class TfEntry:
    parent: str
    child: str
    stamp_ns: int
    T: np.ndarray      # T_parent_child: p_parent = T @ p_child


def _stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def pointcloud2_to_xyz(msg):
    """(N, 3) float64 x, y, z from a PointCloud2. Points with NaN or inf are dropped."""
    fields = {f.name: f for f in msg.fields}
    missing = {"x", "y", "z"} - fields.keys()
    if missing:
        raise ValueError(f"PointCloud2 has no {sorted(missing)} field(s)")
    order = ">" if msg.is_bigendian else "<"
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [np.dtype(_POINTFIELD_TYPES[fields[n].datatype]).newbyteorder(order)
                    for n in ("x", "y", "z")],
        "offsets": [fields[n].offset for n in ("x", "y", "z")],
        "itemsize": msg.point_step,
    })
    n_points = msg.width * msg.height
    raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=n_points)
    xyz = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64)
    return xyz[np.all(np.isfinite(xyz), axis=1)]


def tf_to_entries(msg):
    """One TfEntry per transform in a TFMessage."""
    entries = []
    for t in msg.transforms:
        tr, q = t.transform.translation, t.transform.rotation
        T = make_T(quat_to_rot([q.x, q.y, q.z, q.w]), [tr.x, tr.y, tr.z])
        entries.append(TfEntry(t.header.frame_id, t.child_frame_id, _stamp_ns(t.header.stamp), T))
    return entries


def joint_state_to_dict(msg):
    """{joint name: position}."""
    if len(msg.position) != len(msg.name):
        raise ValueError(f"JointState has {len(msg.name)} names but {len(msg.position)} positions")
    return dict(zip(msg.name, (float(p) for p in msg.position)))


def read_bag(path, topics: Optional[Iterable[str]] = None) -> Iterator[BagMessage]:
    """Yield the bag's point clouds, TF and joint states in log-time order.

    topics: only read these topics (None reads all). Topics of other message types
    are skipped, so a full bag can be passed in as-is.
    """
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, record, msg in reader.iter_decoded_messages(topics=topics):
            name, topic, log_time = schema.name, channel.topic, record.log_time
            if name == POINTCLOUD2:
                yield BagMessage(topic, "cloud", log_time, _stamp_ns(msg.header.stamp),
                                 msg.header.frame_id, pointcloud2_to_xyz(msg))
            elif name == TF:
                entries = tf_to_entries(msg)
                stamp = entries[0].stamp_ns if entries else log_time
                yield BagMessage(topic, "tf", log_time, stamp, "", entries)
            elif name == JOINT_STATE:
                yield BagMessage(topic, "joints", log_time, _stamp_ns(msg.header.stamp),
                                 msg.header.frame_id, joint_state_to_dict(msg))
