"""Renders the scene's cameras in their own process, from the pose on /arm_emulator/qpos.

A separate process, because rendering would otherwise hold Python's interpreter lock against the drives'
ROS callbacks: enough, with CPU rendering, for joint_command's 0.2 s feedback watchdog to halt the arm.

    python3 -m arm_emulator.camera --scene rubiks_cube --cameras wrist_left,ego     # run_emulator starts this
"""
from __future__ import annotations

import argparse
import os
import threading
import time

from arm_emulator import config  # noqa: F401 -- sys.path for the scenes

# Same topics as bridge_node's docstring: the wrist camera on its real topic.
CAMERA_TOPICS = {"wrist_left": "/wrist_cam/image_rect", "ego": "/sim/ego/image_raw"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="rubiks_cube")
    p.add_argument("--cameras", default="wrist_left,ego")
    p.add_argument("--image-size", default="480x640", help="HxW")
    p.add_argument("--hz", type=float, default=30.0)
    args = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco
    import rclpy
    from humanoid_mujoco_scenes import make_model
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float64MultiArray

    height, width = (int(v) for v in args.image_size.split("x"))
    names = args.cameras.split(",")
    # Only poses matter here: the actuators don't change what a camera sees.
    model = make_model(args.scene, cameras={name: (height, width) for name in names})
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)

    rclpy.init()
    node = Node("arm_emulator_cameras")
    # Images are big and only the newest matters.
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=2)
    pubs = {name: node.create_publisher(Image, CAMERA_TOPICS[name], qos) for name in names}
    lock = threading.Lock()
    latest: list[list[float] | None] = [None]

    def on_qpos(msg: Float64MultiArray) -> None:
        with lock:
            latest[0] = list(msg.data)

    node.create_subscription(Float64MultiArray, "/arm_emulator/qpos", on_qpos, 2)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    period, next_frame = 1.0 / args.hz, time.monotonic()
    try:
        while rclpy.ok():
            next_frame += period
            with lock:
                qpos = latest[0]
            if qpos is not None and len(qpos) == model.nq:
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                for name, pub in pubs.items():
                    renderer.update_scene(data, name)
                    image = renderer.render()
                    out = Image()
                    out.header.stamp = node.get_clock().now().to_msg()
                    out.header.frame_id = name
                    out.height, out.width = height, width
                    out.encoding = "rgb8"
                    out.step = width * 3
                    out.data = image.tobytes()
                    pub.publish(out)
            delay = next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame = time.monotonic()  # slower than the rate: don't try to catch up
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
