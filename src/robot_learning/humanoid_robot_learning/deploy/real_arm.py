"""RealArmPolicyRunner: run a trained policy on the real left arm through joint_command.

A dry run by default: it reads motor feedback and cameras, runs the policy and prints its targets
next to the measured pose, but never creates a publisher. --live publishes each action as ArmPose
on /arm/joint_targets once the preflight passes and you type `live`. The path from there to the
motors:

    policy action (7,) rad + gripper 0..1      dataset schema = sim joint1L..joint6l, joint7l closure
      -> vector_to_arm_pose: deg, ArmPose slots  slot i = joint_command jointPaths()[i]
      -> /arm/joint_targets -> joint_command     clamp, low-pass, velocity/delta limit, calibration,
                                                 seed from feedback, MIT watchdogs, 10 s staleness
      -> /interfacing/motorCMD -> can_node       MIT / servo frames -> CAN bus -> drives
      -> /interfacing/motorFeedback              -> ArmFeedbackState -> observation.state

Cases, checked every tick before anything is published:

    case               trigger                                         behaviour
    no data            a motor or camera has nothing newer than 0.5 s  hold the last measured pose, stop
    bad action         not 7 finite values                             hold, stop
    far first target   first action > --start-tol deg from measured    refuse to start (nothing moved)
    diverged           a target > --max-gap deg from measured          hold, stop
    error              the policy or the loop raises                   hold, then re-raise
    otherwise          -                                               publish the action

"Hold" publishes the measured pose (gripper left where it is) until Ctrl-C. Going silent instead
would leave joint_command ramping to the last target for command_timeout_sec (10 s). So every way
out of the loop, Ctrl-C and exceptions included, goes through hold().
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# The arm's calibration and feedback readers live with the CAN tools, which are scripts rather than a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "interfacing" / "can" / "scripts"))

from arm_feedback import ArmFeedbackState  # noqa: E402
from humanoid_robot_learning.recording.arm_pose_io import vector_to_arm_pose  # noqa: E402
from humanoid_robot_learning.recording.observation import fit_image, image_msg_to_hwc  # noqa: E402

ACTION_TOPIC = "/arm/joint_targets"
FEEDBACK_TOPIC = "/interfacing/motorFeedback"
START_TOL_DEG = 15.0  # the leader teleop's AGREE_TOL_DEG: the first target must start from the arm
MAX_GAP_DEG = 30.0    # 2x START_TOL: past the rate-limit lag, the policy has left the arm behind
PREFLIGHT_WAIT_S = 5.0
IMAGE_STALE_S = 0.5   # s, as FEEDBACK_STALE_S: an older frame is a camera that stopped publishing
STATUS_PERIOD_S = 1.0

PolicyFn = Callable[[dict[str, np.ndarray]], np.ndarray]


def action_failures(action: np.ndarray, measured_deg: np.ndarray, names: list[str], tol_deg: float) -> list[str]:
    """Why this action must not be published; empty when it may. measured_deg: (6,) command frame."""
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        return [f"action must be 7 finite values (6 joints rad + gripper), got {action.tolist()}"]
    gap = np.degrees(action[:6]) - measured_deg
    return [f"{n}: target {g:+.1f} deg from measured" for n, g in zip(names, gap) if abs(g) > tol_deg]


class RealArmPolicyRunner:
    """Run policy_fn on the real arm: read -> infer -> check -> command, once per period (see _policy_loop).

    Adapter in:  /interfacing/motorFeedback + camera topics -> observation.state (7,), observation.images.<cam>
    Adapter out: action (7,) rad + gripper 0..1 -> ArmPose (deg) on /arm/joint_targets
    Safety:      preflight before the start, action_failures on every action, hold() on every way out.
    """

    def __init__(self, node: Any, policy_fn: PolicyFn, feedback: ArmFeedbackState, *, fps: float,
                 cameras: dict[str, tuple[str, int, int]], live: bool = False,
                 start_tol_deg: float = START_TOL_DEG, max_gap_deg: float = MAX_GAP_DEG) -> None:
        from common_msgs.msg import ArmPose, MotorFeedback
        from sensor_msgs.msg import Image

        self.node, self.policy_fn, self.feedback = node, policy_fn, feedback
        self.period, self.live = 1.0 / fps, live
        self.start_tol_deg, self.max_gap_deg = start_tol_deg, max_gap_deg
        self.names = [j["name"] for j in feedback.joints]
        self._ArmPose = ArmPose
        self._sizes = {name: (h, w) for name, (_, h, w) in cameras.items()}
        self._images: dict[str, tuple[np.ndarray, float]] = {}  # camera -> (image, monotonic time)
        self._lock = threading.Lock()
        self._q_deg: np.ndarray | None = None  # last measured joints: what hold() publishes
        node.create_subscription(MotorFeedback, FEEDBACK_TOPIC, feedback.on_feedback, 20)
        for name, (topic, _, _) in cameras.items():
            node.create_subscription(Image, topic, lambda msg, k=name: self._on_image(k, msg), 2)
        # Dry run: no publisher exists, so nothing here can reach joint_command.
        self._pub = node.create_publisher(ArmPose, ACTION_TOPIC, 10) if live else None

    # ---- loop -------------------------------------------------------------------------------------------

    def run(self) -> None:
        """Policy loop at fps until Ctrl-C. Every way out after the start, errors included, holds the arm."""
        start = self.observation()
        if start is None:  # the preflight passed, but data went stale while waiting at the `live` prompt
            print("[REFUSED] no fresh data: " + ", ".join(self.missing()), flush=True)
            return
        self._q_deg = start[1]
        try:
            reason = self._policy_loop()
        except KeyboardInterrupt:
            reason = "Ctrl-C"
        except Exception as e:
            self.hold(f"{type(e).__name__}: {e}")
            raise
        if reason is not None:
            self.hold(reason)

    def _policy_loop(self) -> str | None:
        """-> why to hold, or None if the first action was refused (nothing published)."""
        first, next_status = True, 0.0
        while True:
            t0 = time.monotonic()

            # 1. Read (adapter in): motor feedback + cameras -> policy observation, and q_deg of the same sample.
            snapshot = self.observation()
            if snapshot is None:
                return "lost data: " + ", ".join(self.missing())
            obs, self._q_deg = snapshot

            # 2. Infer: observation -> action (7,) = 6 joint targets in rad + gripper 0 open .. 1 closed.
            action = np.asarray(self.policy_fn(obs), dtype=np.float64).reshape(-1)

            # 3. Check: is the action safe to send? The first must start within start_tol of the arm.
            failures = action_failures(action, self._q_deg, self.names, self.start_tol_deg if first else self.max_gap_deg)
            if failures and first:
                print("[REFUSED] first target is not where the arm is:\n  " + "\n  ".join(failures), flush=True)
                return None
            if failures:
                return "; ".join(failures)
            first = False

            # 4. Command (adapter out): action rad -> ArmPose deg -> /arm/joint_targets (no-op in a dry run).
            self.publish(action)

            # 5. Pace: status line once a second, then sleep out the rest of the period.
            if t0 >= next_status:
                next_status = t0 + STATUS_PERIOD_S
                self._print_status(action, time.monotonic() - t0)
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)))

    def _print_status(self, action: np.ndarray, step_s: float) -> None:
        cols = "  ".join(f"{n} {np.degrees(a):+6.1f}/{m:+6.1f}" for n, a, m in zip(self.names, action, self._q_deg))
        late = "  LATE: inference slower than --fps" if step_s > self.period else ""
        tag = "LIVE" if self.live else "DRY"
        print(f"[{tag}] target/measured deg  {cols}  grip {action[6]:.2f}  step {step_s * 1e3:.0f} ms{late}", flush=True)

    # ---- adapter in: ROS -> policy observation ----------------------------------------------------------

    def observation(self) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        """(policy observation, measured q_deg of the same feedback sample), or None if any data is stale."""
        snapshot, images = self.feedback.snapshot(), self._fresh_images()
        if snapshot is None or len(images) < len(self._sizes):
            return None
        state, q_deg = snapshot
        return {"observation.state": state, **{f"observation.images.{k}": v for k, v in images.items()}}, q_deg

    def missing(self) -> list[str]:
        """Motors and cameras with no fresh data; empty when observation() can be built."""
        images = self._fresh_images()
        return self.feedback.missing() + [f"camera {k}" for k in self._sizes if k not in images]

    def _on_image(self, name: str, msg: Any) -> None:
        image = fit_image(image_msg_to_hwc(msg), *self._sizes[name])
        with self._lock:
            self._images[name] = (image, time.monotonic())

    def _fresh_images(self) -> dict[str, np.ndarray]:
        now = time.monotonic()
        with self._lock:
            return {k: image for k, (image, t) in self._images.items() if now - t < IMAGE_STALE_S}

    # ---- adapter out: policy action -> ArmPose ----------------------------------------------------------

    def publish(self, action: np.ndarray, unit: str = "rad") -> None:
        if self._pub is None:
            return
        msg = vector_to_arm_pose(self._ArmPose, action, unit=unit, name="real_arm_policy_runner")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self._pub.publish(msg)

    # ---- safety -----------------------------------------------------------------------------------------

    def preflight(self) -> list[str]:
        """Why the policy may not start; empty when it may. Waits up to PREFLIGHT_WAIT_S for data."""
        deadline = time.monotonic() + PREFLIGHT_WAIT_S
        while time.monotonic() < deadline and self.observation() is None:
            time.sleep(0.1)
        out = [f"{k}: no data in the last 0.5 s" for k in self.missing()]
        out += self.feedback.out_of_limits()
        others = [i.node_name for i in self.node.get_publishers_info_by_topic(ACTION_TOPIC) if i.node_name != self.node.get_name()]
        if others:
            out.append(f"{ACTION_TOPIC} already has a publisher: {others} (teleop? stop it)")
        if self.live and not self.node.get_subscriptions_info_by_topic(ACTION_TOPIC):
            out.append(f"nobody subscribes to {ACTION_TOPIC}: start joint_command (interfacing container)")
        return out

    def hold(self, reason: str) -> None:
        """Publish the last measured pose (gripper left where it is) until Ctrl-C."""
        print(f"\n[STOP] {reason}", flush=True)
        if self._pub is None:
            return
        print("[HOLD] holding the measured pose. Support the arm, then Ctrl-C: joint_command goes to kp = 0 "
              "command_timeout_sec (10 s) after the last ArmPose.", flush=True)
        try:
            while True:
                self.publish(self._q_deg, unit="deg")
                time.sleep(self.period)
        except KeyboardInterrupt:
            pass


def make_policy_fn(path: str, device: str, task: str | None) -> tuple[PolicyFn, dict[str, tuple[int, int]], Any]:
    """-> (policy_fn, {camera: (h, w)} the policy was trained on, policy) for a LeRobot checkpoint."""
    import torch
    from lerobot.utils.control_utils import predict_action

    from humanoid_robot_learning.utils.checkpoint import load_policy

    policy, pre, post = load_policy(path, device)
    cameras = {k.removeprefix("observation.images."): tuple(ft.shape[1:]) for k, ft in policy.config.image_features.items()}

    def policy_fn(obs: dict[str, np.ndarray]) -> np.ndarray:
        action = predict_action(obs, policy, torch.device(device), pre, post, use_amp=False, task=task)
        return action.reshape(-1).cpu().numpy()

    return policy_fn, cameras, policy


def camera_topics(schema: dict[str, Any], sizes: dict[str, tuple[int, int]]) -> dict[str, tuple[str, int, int]]:
    """{camera: (ROS topic, h, w)} for every camera the policy reads; exits if one has no real topic."""
    images = schema.get("images") or {}
    missing = [k for k in sizes if not (images.get(k) or {}).get("topic")]
    if missing:
        raise SystemExit(f"the policy reads cameras {missing}, which have no `topic` in the schema (no real camera yet)")
    return {k: (images[k]["topic"], h, w) for k, (h, w) in sizes.items()}


def parse_args() -> argparse.Namespace:
    pkg = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Run a LeRobot policy on the real left arm through joint_command: "
                                            "a dry run (publishes nothing) unless --live.")
    p.add_argument("--policy", required=True, help="pretrained_model directory of a LeRobot checkpoint")
    p.add_argument("--live", action="store_true", help=f"publish {ACTION_TOPIC} once the preflight passes and you type `live`")
    p.add_argument("--task", default=None, help="task text (default: the training dataset's only task)")
    p.add_argument("--device", default="cpu", help="torch device for inference (simulation_mj has CPU torch)")
    p.add_argument("--schema", default=str(pkg / "config" / "pioneer_humanoid_dataset_schema.yaml"),
                   help="dataset schema: fps and camera topics (default: the one the teleops record)")
    p.add_argument("--fps", type=float, default=None, help="policy rate (default: the schema's fps, which training used)")
    p.add_argument("--arm-calibration", default=None, help="arm_calibration.yaml (default: auto, as joint_command)")
    p.add_argument("--start-tol", type=float, default=START_TOL_DEG, help="deg: the first target must be this close to the arm")
    p.add_argument("--max-gap", type=float, default=MAX_GAP_DEG, help="deg: stop if a target gets this far from the arm")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from joint_config import find_calibration

    from humanoid_robot_learning.utils.checkpoint import training_task
    from humanoid_robot_learning.recording.schema import load_yaml

    schema = load_yaml(Path(args.schema))
    task = args.task or training_task(args.policy)
    if task is None:
        raise SystemExit("pass --task: the training dataset's task was not found")
    policy_fn, sizes, policy = make_policy_fn(args.policy, args.device, task)
    cameras = camera_topics(schema, sizes)
    calibration = find_calibration(args.arm_calibration)
    feedback = ArmFeedbackState(calibration)
    print(f"left arm, {'LIVE once the preflight passes' if args.live else 'DRY RUN (publishes nothing)'}. "
          f"calibration: {calibration}, task: {task!r}, cameras: {cameras}")

    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    # rclpy's own SIGINT handler shuts the context down, after which hold() could not publish. Without it,
    # Ctrl-C is a plain KeyboardInterrupt in this thread and the publisher stays valid for the hold.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("real_arm_policy_runner")
    runner = RealArmPolicyRunner(node, policy_fn, feedback, fps=args.fps or float(schema["fps"]), cameras=cameras,
                                 live=args.live, start_tol_deg=args.start_tol, max_gap_deg=args.max_gap)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        failures = runner.preflight()
        if failures:
            raise SystemExit("[PREFLIGHT] not starting:\n  " + "\n  ".join(failures))
        if args.live and input("\nPreflight passed. Arm supported, hand on the E-stop, then type `live` "
                               "(anything else quits): ").strip() != "live":
            return
        policy.reset()
        runner.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
