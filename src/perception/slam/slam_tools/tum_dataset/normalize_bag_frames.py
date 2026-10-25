#!/usr/bin/env python3

"""Strip ROS 1-style leading slashes from frame IDs in a ROS 2 bag."""

import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

from rosbags.rosbag2 import Reader, StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore


def normalize_frames(value: object) -> int:
    """Normalize all frame_id and child_frame_id fields in a ROS message."""
    changed = 0
    if is_dataclass(value):
        for field in fields(value):
            item = getattr(value, field.name)
            if field.name in {"frame_id", "child_frame_id"} and isinstance(item, str):
                normalized = item.lstrip("/")
                if normalized != item:
                    setattr(value, field.name, normalized)
                    changed += 1
            else:
                changed += normalize_frames(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            changed += normalize_frames(item)
    return changed


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} INPUT_ROS2_BAG OUTPUT_ROS2_BAG")

    source, destination = map(Path, sys.argv[1:])
    if destination.exists():
        raise SystemExit(f"Error: output directory already exists: {destination}")

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    changed = 0
    with Reader(source) as reader, Writer(
        destination, version=9, storage_plugin=StoragePlugin.MCAP
    ) as writer:
        connections = {
            connection.id: writer.add_connection(
                connection.topic,
                connection.msgtype,
                typestore=typestore,
                serialization_format=connection.ext.serialization_format,
                offered_qos_profiles=connection.ext.offered_qos_profiles,
            )
            for connection in reader.connections
        }

        for connection, timestamp, raw in reader.messages():
            message = typestore.deserialize_cdr(raw, connection.msgtype)
            edits = normalize_frames(message)
            changed += edits
            data = typestore.serialize_cdr(message, connection.msgtype) if edits else raw
            writer.write(connections[connection.id], timestamp, data)

    print(f"Normalized {changed} frame ID fields.")


if __name__ == "__main__":
    main()
