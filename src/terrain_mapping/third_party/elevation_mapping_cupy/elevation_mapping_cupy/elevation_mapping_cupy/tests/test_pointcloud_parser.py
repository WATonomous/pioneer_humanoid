import struct

import numpy as np
import pytest
from sensor_msgs.msg import PointCloud2, PointField

from elevation_mapping_cupy.elevation_mapping_node import _pointcloud2_xyz_f32


def _padded_cloud() -> PointCloud2:
    message = PointCloud2()
    message.height = 2
    message.width = 2
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = 40
    message.is_dense = True
    data = bytearray(message.row_step * message.height)
    points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0), (10.0, 11.0, 12.0)]
    for index, point in enumerate(points):
        row, col = divmod(index, message.width)
        offset = row * message.row_step + col * message.point_step
        struct.pack_into("<ffff", data, offset, *point, 100.0 + index)
    message.data = bytes(data)
    return message


def test_parser_honors_organized_cloud_row_padding():
    points = _pointcloud2_xyz_f32(_padded_cloud())

    np.testing.assert_array_equal(
        points,
        np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        ),
    )


def test_parser_rejects_truncated_organized_cloud():
    message = _padded_cloud()
    message.data = message.data[:-1]

    with pytest.raises(ValueError, match="expected at least"):
        _pointcloud2_xyz_f32(message)
