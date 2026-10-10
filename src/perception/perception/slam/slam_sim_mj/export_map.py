#!/usr/bin/env python3
"""Save RTAB-Map's 2D occupancy grid as a PNG, while rtabmap is still running.

    python3 export_map.py [out.png]

Two things are needed to get a map out AFTER a run has finished, and missing either
one presents identically as "rtabmap publishes no map":

  * TRANSIENT_LOCAL durability on the subscription. /rtabmap/map is latched; a
    default (VOLATILE) subscriber attaching after the last update receives nothing.
  * Calling /rtabmap/rtabmap/publish_map first. RTAB-Map assembles the occupancy
    grid lazily -- only when a subscriber exists AND new sensor data arrives. Once
    the publisher stops there are no more updates, so the latched topic stays empty
    however correct the QoS is. The service forces one assembly and publish.

Grey is unknown, white is free space, black is obstacle -- the same reading as the
RTAB-Map Database Viewer's graph view.
"""

import sys

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from PIL import Image
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.srv import PublishMap

OUT = sys.argv[1] if len(sys.argv) > 1 else "map_2d.png"


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("export_map")
    got = {}

    def on_map(msg: OccupancyGrid) -> None:
        got["msg"] = msg

    node.create_subscription(
        OccupancyGrid, "/rtabmap/map", on_map,
        QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                   durability=DurabilityPolicy.TRANSIENT_LOCAL,
                   history=HistoryPolicy.KEEP_LAST, depth=1),
    )

    client = node.create_client(PublishMap, "/rtabmap/rtabmap/publish_map")
    if client.wait_for_service(timeout_sec=5.0):
        req = PublishMap.Request()
        req.global_map = True
        req.optimized = True   # the graph AFTER loop closures, not raw odometry
        req.graph_only = False
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
    else:
        print("warning: /rtabmap/rtabmap/publish_map not available; "
              "relying on whatever is already latched")

    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.1)
        if got:
            break

    if not got:
        print("no map received on /rtabmap/map -- is rtabmap still running?")
        return 1

    msg = got["msg"]
    grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    img = np.full(grid.shape, 128, np.uint8)   # unknown
    img[grid == 0] = 255                       # free
    img[grid > 0] = 0                          # obstacle
    img = np.flipud(img)                       # ROS grids are y-up, images are y-down
    Image.fromarray(img).save(OUT)

    known = int((grid >= 0).sum())
    print(f"{OUT}: {msg.info.width}x{msg.info.height} @ {msg.info.resolution:.3f} m/cell")
    print(f"  free {int((grid == 0).sum()):6d}   obstacle {int((grid > 0).sum()):6d}   "
          f"unknown {int((grid < 0).sum()):6d}")
    if known:
        print(f"  free fraction of known cells: {(grid == 0).sum() / known:.1%}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
