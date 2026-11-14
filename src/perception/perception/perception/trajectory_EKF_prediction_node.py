import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Pose, PointStamped
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool
from common_msgs.msg import PerceptionImpactEstimation
import numpy as np


class EKFPredictionNode(Node):
    def __init__(self):
        super().__init__('trajectory_EKF_prediction_node')
        self.dt = 0.01
        # ---------------- STATE ----------------
        # [px, py, pz, vx, vy, vz]
        self.x = np.zeros(6)
        self.true_pos = np.zeros(3)
        self.true_pos_valid = False
        self.increments = 0

        # ---------------- COVARIANCE ----------------
        self.P = np.eye(6) * 0.1
        self.P[3:6, 3:6] = np.eye(3) * 50.0   # high uncertainty on vx, vy, vz
        self.R = (0.04 ** 2) * np.eye(3)  # measurement noise
        self.process_noise_pos = 1e-2
        self.process_noise_vel = 1e-2

        # ---------------- CONSTANTS ----------------
        # Gravity depends on the frame of the incoming measurements: 'z_up' (robot convention, the default) or
        # 'optical' (camera frame: x right, y down, z forward, with the camera tilted up by camera_pitch_deg)
        self.gravity_frame = self.declare_parameter('gravity_frame', 'z_up').value
        pitch = np.radians(self.declare_parameter('camera_pitch_deg', 0.0).value)
        optical = self.gravity_frame == 'optical'
        self.g = 9.81 * (np.array([0.0, np.cos(pitch), -np.sin(pitch)]) if optical else np.array([0.0, 0.0, -1.0]))
        self.frame_warned = False
        self.L = 3.4  # aerodynamic characteristic length (L-value)
        self.look_ahead_time = 2  # seconds
        self.bounding_sphere_radius = 2.0  # meters

        # ---------------- TRACK RULES ----------------
        # Seconds without a measurement before the track is dropped: short while the shuttle should still be in the
        # camera's view (it was probably lost), longer once it has left the view, so the impact estimate keeps
        # being published as the shuttle flies out of the picture just before reaching the robot
        self.track_timeout = self.declare_parameter('track_timeout', 0.5).value
        self.coast_timeout = self.declare_parameter('coast_timeout', 1.0).value
        # A measurement is used if it is within gate_distance (m) of the prediction, or inside the
        # filter's own uncertainty ellipse (squared Mahalanobis distance <= gate_chi2; 16.27 = 99.9%, 3 dof)
        self.gate_distance = self.declare_parameter('gate_distance', 1.5).value
        self.gate_chi2 = self.declare_parameter('gate_chi2', 16.27).value
        # Consecutive rejected measurements after which the track is considered lost and restarted
        self.max_rejections = self.declare_parameter('max_rejections', 5).value
        self.rejections = 0
        self.last_meas_clock = None  # node-clock time (s) the last measurement arrived

        # ---------------- ROS ----------------
        self.pub = self.create_publisher(Pose, '/ekf_predicted_states', 10)
        self.sub = self.create_subscription(
            PointStamped, '/shuttle_states', self.shuttle_callback, 10)
        self.true_sub = self.create_subscription(
            PointStamped,
            '/shuttle_states_true',
            self.shuttle_true_callback,
            10
        )
        self.new_spawn = self.create_subscription(
            Bool, '/shuttle_spawned', self.shuttle_spawned_callback, 10)
        self.impact_estimate_pub = self.create_publisher(
            PerceptionImpactEstimation, '/ekf_impact', 10)
        # True once each time a track is lost (no measurements for too long, or they keep missing the prediction)
        self.lost_pub = self.create_publisher(Bool, '/ekf_track_lost', 10)
        # Camera calibration of the frame the measurements are in, to tell whether the shuttle is still in view
        # (only used with gravity_frame 'optical')
        self.intrinsics = None  # fx, fy, cx, cy, width, height
        camera_info_topic = self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info').value
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)

        self.latest_meas = None
        self.latest_meas_time = None
        self.new_meas_available = False
        self.low_uncertainty = False

        # ---------------- VELOCITY SEEDING ----------------
        self.prev_meas = None
        self.velocity_initialized = False
        self.prev_meas_time = None

        self.timer = self.create_timer(self.dt, self.step)

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def check_frame(self, frame_id):
        """Warn once if the measurements' frame doesn't look like the one gravity was set for."""
        if not self.frame_warned and ('optical' in frame_id) != (self.gravity_frame == 'optical'):
            self.get_logger().warn(f"Measurements are in '{frame_id}' but gravity_frame is '{self.gravity_frame}': gravity may point the wrong way")
            self.frame_warned = True

    def camera_info_callback(self, msg):
        self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5], msg.width, msg.height)

    def in_view(self, p):
        """Whether position p projects into the camera image. False if that can't be known (not the optical
        frame, or no camera info yet), which gives the longer timeout."""
        if self.gravity_frame != 'optical' or self.intrinsics is None or p[2] <= 0:
            return False
        fx, fy, cx, cy, width, height = self.intrinsics
        u, v = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
        return 0 <= u < width and 0 <= v < height

    # ---------------- CALLBACK ----------------
    def shuttle_callback(self, msg):
        self.check_frame(msg.header.frame_id)

        if self.latest_meas_time is not None:
            dt_elapsed = (
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                - (
                    self.latest_meas_time.sec
                    + self.latest_meas_time.nanosec * 1e-9
                )
            )
            # a long gap between measurements (including a time jump): assume a new shuttle
            if dt_elapsed > self.coast_timeout:
                self.shuttle_spawned_callback(Bool(data=True))

        self.latest_meas = msg
        self.latest_meas_time = msg.header.stamp
        self.last_meas_clock = self.now_seconds()
        self.new_meas_available = True

        z = np.array([msg.point.x, msg.point.y, msg.point.z])

        if not self.velocity_initialized:
            # Always initialize position from the first measurement so
            # published state/impact estimation aren't based on (0, 0, 0).
            if self.prev_meas is None:
                self.x[0:3] = z
                self.prev_meas = z
                self.prev_meas_time = msg.header.stamp
                return

            dt_elapsed = (
                self.latest_meas_time.sec
                + self.latest_meas_time.nanosec * 1e-9
                - (
                    self.prev_meas_time.sec
                    + self.prev_meas_time.nanosec * 1e-9
                )
            )
            # Keep position synced to the latest measurement while we wait
            # to seed velocity.
            self.x[0:3] = z

            # guard: only seed if elapsed time is not like zero or too small
            if dt_elapsed > 0.005:
                v_init = (z - self.prev_meas) / dt_elapsed
                self.x[3:6] = v_init
                self.velocity_initialized = True
                self.get_logger().info(f"Velocity seeded: {v_init}")

            self.prev_meas = z
            self.prev_meas_time = msg.header.stamp

    def shuttle_true_callback(self, msg):
        self.true_pos = np.array([
            msg.point.x,
            msg.point.y,
            msg.point.z
        ])
        self.true_pos_valid = True

    def shuttle_spawned_callback(self, msg):
        if msg.data:
            self.get_logger().info("Shuttle spawned → resetting EKF state")
            self.reset_track()

    def drop_track(self, reason):
        """Reset the track and tell consumers that the shuttle was lost."""
        self.get_logger().info(f"{reason} → dropping track")
        self.lost_pub.publish(Bool(data=True))
        self.reset_track()

    def reset_track(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.1
        self.P[3:6, 3:6] = np.eye(3) * 50.0
        self.increments = 0
        self.latest_meas = None
        self.new_meas_available = False
        self.latest_meas_time = None
        self.last_meas_clock = None
        self.rejections = 0
        self.true_pos = np.zeros(3)
        self.true_pos_valid = False
        self.low_uncertainty = False
        self.prev_meas = None
        self.velocity_initialized = False
        self.prev_meas_time = None

    # ---------------- MODEL ----------------
    def f(self, x):
        p = x[0:3]
        v = x[3:6]

        speed = np.linalg.norm(v)

        dp = v
        dv = self.g - (speed / self.L) * v

        p_new = p + dp * self.dt
        v_new = v + dv * self.dt

        return np.hstack((p_new, v_new))

    # ---------------- JACOBIAN ----------------
    def F_jacobian(self, x):
        v = x[3:6]
        speed = np.linalg.norm(v) + 1e-12

        F = np.eye(6)

        # dp/dv = I * dt
        F[0:3, 3:6] = np.eye(3) * self.dt

        # dv/dv nonlinear drag term
        identity = np.eye(3)

        # derivative of (|v| v)
        outer = np.outer(v, v) / speed

        drag_jac = (1 / self.L) * (speed * identity + outer)

        F[3:6, 3:6] = identity - drag_jac * self.dt

        return F

    # ---------------- IMPACT ESTIMATION ----------------
    def estimate_impact(self):
        # Avoid publishing a false "impact at origin" before the filter is initialized.
        if self.latest_meas is None or not self.velocity_initialized:
            return

        future_x = self.x.copy()
        impact = False
        time_to_impact = -1
        position_of_impact = np.zeros(3)
        hitback_direction = np.zeros(3)

        already_impacted = (
            future_x[0]**2 + future_x[1]**2 + future_x[2]**2
            <= self.bounding_sphere_radius**2
        )
        if already_impacted:
            impact = True
            time_to_impact = 0.0
            position_of_impact = future_x[0:3]
            vel = future_x[3:6]
            vel_norm = np.linalg.norm(vel)
            hitback_direction = (
                -(vel / vel_norm) if vel_norm > 1e-9 else np.zeros(3)
            )
        else:
            for i in range(int(self.look_ahead_time / self.dt)):
                future_x = self.f(future_x)
                if (
                    future_x[0]**2 + future_x[1]**2 + future_x[2]**2
                    <= self.bounding_sphere_radius**2
                ):
                    impact = True
                    vel = future_x[3:6]
                    vel_norm = np.linalg.norm(vel)
                    hitback_direction = (
                        -(vel / vel_norm) if vel_norm > 1e-9
                        else np.zeros(3)
                    )
                    time_to_impact = (i + 1) * self.dt
                    position_of_impact = future_x[0:3]
                    break
        if impact:
            self.get_logger().debug(
                "Impact estimated in "
                f"{time_to_impact:.2f} seconds at position "
                f"{position_of_impact} with hit back racket direction "
                f"{hitback_direction}"
            )
        else:
            self.get_logger().debug(
                f"No impact estimated in the next {self.look_ahead_time} seconds"
            )
            return

        # Publish the impact estimate
        impact_msg = PerceptionImpactEstimation()
        impact_msg.time_to_impact = float(time_to_impact)

        impact_msg.position_of_impact.x = float(position_of_impact[0])
        impact_msg.position_of_impact.y = float(position_of_impact[1])
        impact_msg.position_of_impact.z = float(position_of_impact[2])

        impact_msg.hitback_direction.x = float(hitback_direction[0])
        impact_msg.hitback_direction.y = float(hitback_direction[1])
        impact_msg.hitback_direction.z = float(hitback_direction[2])

        impact_msg.uncertainty = float(np.trace(self.P))
        impact_msg.header.stamp = self.get_clock().now().to_msg()
        impact_msg.header.frame_id = (
            self.latest_meas.header.frame_id
            if self.latest_meas is not None
            else "robot_base_link"
        )

        self.impact_estimate_pub.publish(impact_msg)

    # ---------------- MEASUREMENT GATE ----------------
    def within_gate(self, y, S):
        """y: measurement minus prediction, S: its covariance. True if the measurement is plausible."""
        if np.linalg.norm(y) <= self.gate_distance:
            return True
        return float(y @ np.linalg.solve(S, y)) <= self.gate_chi2

    # ---------------- STEP ----------------

    def step(self):
        # No track (nothing measured yet, or the last one was dropped): nothing to predict or publish
        if self.latest_meas is None:
            return
        # The shuttle stopped being measured: drop the track instead of predicting it forever
        timeout = self.track_timeout if self.in_view(self.x[:3]) else self.coast_timeout
        if self.last_meas_clock is not None and self.now_seconds() - self.last_meas_clock > timeout:
            self.drop_track(f"No measurement for {timeout} s")
            return

        self.increments += 1
        if (
            self.true_pos_valid
            and (
                self.true_pos[0]**2
                + self.true_pos[1]**2
                + self.true_pos[2]**2
                <= self.bounding_sphere_radius**2
            )
        ):
            self.get_logger().debug(
                f"Shuttle has impacted at true position {self.true_pos}")
        if (np.trace(self.P) < 2):
            if (not self.low_uncertainty):
                self.get_logger().info(
                    "Low uncertainty → reducing process noise"
                )
            self.process_noise_pos = 1e-5
            self.process_noise_vel = 1e-5
            self.low_uncertainty = True
        else:
            self.process_noise_pos = 1e-2
            self.process_noise_vel = 1e-2

        # ===== PREDICT =====
        F = self.F_jacobian(self.x)   # old state
        self.x = self.f(self.x)

        Q = np.eye(6)
        Q[0:3, 0:3] *= self.process_noise_pos
        Q[3:6, 3:6] *= self.process_noise_vel
        self.P = F @ self.P @ F.T + Q

        # ===== UPDATE =====
        if self.latest_meas is not None and self.new_meas_available:
            z = np.array([
                self.latest_meas.point.x,
                self.latest_meas.point.y,
                self.latest_meas.point.z
            ])

            H = np.zeros((3, 6))
            H[:, 0:3] = np.eye(3)

            y = z - H @ self.x

            S = H @ self.P @ H.T + self.R

            # Gate: only use measurements that are plausible given the prediction (the velocity
            # must be seeded first: until then the prediction is just the previous measurement)
            if not self.velocity_initialized or self.within_gate(y, S):
                self.rejections = 0
                PHt = self.P @ H.T
                K = np.linalg.solve(S.T, PHt.T).T
                self.x = self.x + K @ y

                identity = np.eye(6)
                I_KH = identity - K @ H
                self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
            else:
                self.rejections += 1
                self.get_logger().debug(
                    f"Measurement rejected ({self.rejections}/{self.max_rejections}): "
                    f"{np.linalg.norm(y):.2f} m from the prediction")
                if self.rejections >= self.max_rejections:
                    self.drop_track("Measurements keep missing the predicted track")
                    return

        # ==== IMPACT ESTIMATION (post-update) ====
        self.estimate_impact()

        # ===== PUBLISH =====
        msg = Pose()
        msg.position.x = float(self.x[0])
        msg.position.y = float(self.x[1])
        msg.position.z = float(self.x[2])
        msg.orientation.w = 1.0
        if (
            self.true_pos_valid
            and self.latest_meas is not None
            and self.new_meas_available
        ):
            self.get_logger().debug(
                f"Total Error: {np.linalg.norm(self.true_pos - self.x[0:3])}")

        self.get_logger().debug(f'Predicted Position: {self.x[0:3]}')
        self.new_meas_available = False

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFPredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
