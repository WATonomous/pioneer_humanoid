import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
import numpy as np
from std_msgs.msg import Bool


class Shuttle:
    def __init__(self, velocity, position):
        self.velocity = np.array(velocity, dtype=float)
        self.position = np.array(position, dtype=float)


class DummyTrackNetOutput(Node):

    def __init__(self):
        super().__init__('dummy_tracknet_output')

        self.shuttle = None
        self.dt = 0.02  # 50Hz
        self.history_position = []
        self.noise = 0.04  # m

        # Camera intrinsics (typical RealSense D455 at 640x480)
        self.fx = 380.0
        self.fy = 380.0
        self.cx = 320.0
        self.cy = 240.0
        self.img_width = 640
        self.img_height = 480

        # This is the L-value
        self.aerodynamic_characteristic_length = 3.4

        # Publishes 2D pixel coordinates (like YOLO detector)
        self.pub = self.create_publisher(
            PointStamped,
            '/perception/shuttlecock/detections_2d',
            10
        )
        self.true_pub = self.create_publisher(
            PointStamped,
            '/shuttle_states_true',
            10
        )
        self.new_spawn = self.create_publisher(
            Bool,
            '/shuttle_spawned',
            10
        )
        self.visible_pub = self.create_publisher(
            Bool,
            '/perception/shuttlecock/detection_visible',
            10
        )

        self.timer = self.create_timer(self.dt, self.step)

    def spawn_shuttle(self):

        # units in meters
        # (0,0) is center of court
        # position based on dimensions of a badminton court, 13.4m long
        # x spread is based on width of court, y spread is length,
        # z spread is based on net height to max shuttlecock height
        msg = Bool()
        msg.data = True
        self.new_spawn.publish(msg)
        position = np.array([
            np.random.uniform(-3.05, 3.05),   # x spread
            np.random.uniform(3.35, 10.05),   # y spread
            # zspread (net height to max height of shuttlecock)
            np.random.uniform(1.53, 3)
        ])
        dir_to_origin = np.array([0.0, 0.0, 0.0]) - position
        noise = np.random.uniform(-0.5, 0.5, size=3)
        # invert z direction to ensure shuttle is launched upwards
        dir_to_origin[2] *= -1
        noisy_dir = dir_to_origin + noise

        noisy_dir = noisy_dir / np.linalg.norm(noisy_dir)

        velocity = np.random.uniform(20, 70) * noisy_dir

        self.shuttle = Shuttle(velocity, position)

        self.get_logger().info(
            f"Spawned shuttle | pos={position} vel={velocity}"
        )

    def step(self):

        # AUTO SPAWN IF EMPTY
        if self.shuttle is None:
            self.spawn_shuttle()

        cur_pos = self.shuttle.position
        cur_vel = self.shuttle.velocity
        # using quadratic air drag for shuttlecock trajectory
        # dv/dt = g - ||v||*v/L
        speed = np.linalg.norm(cur_vel)

        drag = (speed * cur_vel / self.aerodynamic_characteristic_length)
        gravity = np.array([0.0, 0.0, -9.81])

        dvdt = gravity - drag

        new_vel = cur_vel + dvdt * self.dt
        new_pos = cur_pos + new_vel * self.dt

        self.shuttle.velocity = new_vel
        self.shuttle.position = new_pos
        self.history_position.append(new_pos)

        # Publish TRUE position (ground truth for debugging)
        true_msg = PointStamped()
        true_msg.header.stamp = self.get_clock().now().to_msg()
        true_msg.header.frame_id = "camera_link"
        true_msg.point.x = float(new_pos[0])
        true_msg.point.y = float(new_pos[1])
        true_msg.point.z = float(new_pos[2])
        self.true_pub.publish(true_msg)

        # Add noise to position
        noise = np.random.uniform(-self.noise, self.noise, size=3)
        noisy_pos = new_pos + noise

        # Project 3D to 2D pixel coordinates
        # Only visible if in front of camera (z > 0) and within image bounds
        visible = False
        u, v = 0.0, 0.0

        if noisy_pos[2] > 0.1:  # Must be in front of camera
            u = self.fx * (noisy_pos[0] / noisy_pos[2]) + self.cx
            v = self.fy * (noisy_pos[1] / noisy_pos[2]) + self.cy

            # Check if within image bounds
            if 0 <= u < self.img_width and 0 <= v < self.img_height:
                visible = True

        # Publish visibility
        vis_msg = Bool()
        vis_msg.data = visible
        self.visible_pub.publish(vis_msg)

        # Publish 2D detection (pixel coordinates)
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.point.x = float(u)  # pixel u
        msg.point.y = float(v)  # pixel v
        msg.point.z = 0.0       # unused for 2D

        self.pub.publish(msg)

        # RESET ON GROUND HIT
        if new_pos[2] < 0:
            self.shuttle = None
            self.get_logger().debug("Shuttle landed → respawning next tick")
            self.get_logger().debug(
                f"Shuttle trajectory history: {self.history_position}")
            self.history_position = []


def main(args=None):
    rclpy.init(args=args)
    node = DummyTrackNetOutput()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()