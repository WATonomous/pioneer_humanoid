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

        # This is the L-value
        self.aerodynamic_characteristic_length = 3.4

        self.pub = self.create_publisher(
            PointStamped,
            '/shuttle_states',
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

        velocity = np.random.uniform(70, 130) * noisy_dir

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
        true_msg = PointStamped()
        true_msg.header.stamp = self.get_clock().now().to_msg()
        true_msg.header.frame_id = "robot_base_link"
        true_msg.point.x = float(new_pos[0])
        true_msg.point.y = float(new_pos[1])
        true_msg.point.z = float(new_pos[2])
        self.true_pub.publish(true_msg)

        noise = np.random.uniform(-self.noise, self.noise, size=3)

        # publish noisy data
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "robot_base_link"
        msg.point.x = float(new_pos[0]+noise[0])
        msg.point.y = float(new_pos[1]+noise[1])
        msg.point.z = float(new_pos[2]+noise[2])

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
