"""Drive the Pioneer left arm in Isaac Sim from the 5-servo leader arm.

The leader is zeroed wherever it is when this program starts.  Its relative
angles are added to the simulated arm's default pose:

    A (servo ID 2) -> joint1L (shoulder flexion)
    B (servo ID 3) -> joint2l (shoulder abduction)
    C (servo ID 1) -> joint3l (shoulder rotation)
    D (servo ID 5) -> joint4l (elbow flexion)
    E (servo ID 4) -> joint5l (forearm rotation)

The remaining robot joints are held at the canonical default pose.  Leader
torque is always disabled; the physical arm is an input device only.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from arm_limits import joint_limits_deg, limited_target

from isaaclab.app import AppLauncher


_REPO_ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description="5-servo leader control of the Pioneer arm in Isaac Sim.")
parser.add_argument("--port", default="/dev/ttyACM0", help="Leader serial port")
parser.add_argument("--baud", type=int, default=1_000_000, help="Leader serial baud rate")
parser.add_argument("--scene", default="bare", help="Registered humanoid_scenes scene")
parser.add_argument(
    "--arm",
    choices=("left", "right"),
    default="left",
    help="Sim arm to command (left is the L-suffixed physical-left chain)",
)
parser.add_argument(
    "--signs",
    default="1,-1,1,-1,1",
    help="Per-axis direction for A,B,C,D,E (default: 1,-1,1,-1,1)",
)
parser.add_argument("--scale", type=float, default=1.0, help="Leader-to-sim angular scale")
parser.add_argument("--verify-travel", action="store_true", help="Test sim-only travel past old limits and return to zero before connecting leader")
parser.add_argument(
    "--filter-alpha",
    type=float,
    default=0.35,
    help="Target low-pass coefficient in (0,1]; 1 disables filtering",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.ui as ui

import isaaclab.sim as sim_utils
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.scene import InteractiveScene

sys.path.insert(0, str(_REPO_ROOT / "src" / "pioneer_humanoid"))

from humanoid_scenes import list_scenes, make_scene_cfg, scene_camera
from pioneer_humanoid.bimanual_arm import (
    BIMANUAL_ARM_CFG,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    resolve_joint_name,
)


SERVO_IDS = {"A": 2, "B": 3, "C": 1, "D": 5, "E": 4}
COUNTS_PER_REV = 4096
COUNTS_PER_RAD = COUNTS_PER_REV / (2.0 * math.pi)
TORQUE_ENABLE_ADDRESS = 40
PRESENT_POSITION_ADDRESS = 56


def _parse_signs(value: str) -> tuple[float, ...]:
    try:
        signs = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--signs must contain five numbers") from exc
    if len(signs) != len(SERVO_IDS) or any(not math.isfinite(sign) or sign == 0.0 for sign in signs):
        raise argparse.ArgumentTypeError("--signs must contain five finite non-zero numbers")
    return signs


def _wrapped_count_delta(position: int, zero: int) -> int:
    """Shortest signed displacement between two consecutive samples."""
    return (position - zero + COUNTS_PER_REV // 2) % COUNTS_PER_REV - COUNTS_PER_REV // 2


class ServoLeader:
    """Read A/B/C/D/E as startup-relative radians while keeping all torque off."""

    def __init__(self, port_name: str, baud: int):
        from scservo_sdk import PacketHandler, PortHandler

        self._port = PortHandler(port_name)
        # SMS/STS servos use protocol-end value 0.  The SDK is installed by the
        # Isaac container, so no copied virtual environment is required.
        self._servo = PacketHandler(0)
        self._zeros: dict[int, int] = {}
        self._last_positions: dict[int, int] = {}
        self._relative_counts: dict[int, int] = {}

        if not self._port.openPort():
            raise RuntimeError(f"could not open leader port {port_name}")
        if not self._port.setBaudRate(baud):
            self._port.closePort()
            raise RuntimeError(f"could not set leader baud rate to {baud}")

        try:
            self.rezero()
        except Exception:
            self.close()
            raise

    def _read_position(self, label: str, servo_id: int) -> int:
        position, result, error = self._servo.read2ByteTxRx(
            self._port, servo_id, PRESENT_POSITION_ADDRESS
        )
        if result != 0 or error != 0:
            raise RuntimeError(
                f"leader {label} (ID {servo_id}) read failed: result={result}, error={error}"
            )
        return int(position)

    def rezero(self) -> None:
        zeros: dict[int, int] = {}
        for label, servo_id in SERVO_IDS.items():
            result, error = self._servo.write1ByteTxRx(
                self._port, servo_id, TORQUE_ENABLE_ADDRESS, 0
            )
            if result != 0 or error != 0:
                raise RuntimeError(
                    f"leader {label} (ID {servo_id}) torque-off failed: result={result}, error={error}"
                )
            time.sleep(0.02)
            zeros[servo_id] = self._read_position(label, servo_id)
        self._zeros = zeros
        self._last_positions = zeros.copy()
        self._relative_counts = {servo_id: 0 for servo_id in zeros}
        print(
            "[INFO] Leader zeroed: "
            + ", ".join(f"{label}=ID{servo_id}@{zeros[servo_id]}" for label, servo_id in SERVO_IDS.items()),
            flush=True,
        )

    def read_radians(self) -> tuple[float, ...]:
        angles = []
        for label, servo_id in SERVO_IDS.items():
            position = self._read_position(label, servo_id)
            # Unwrap incrementally instead of choosing the shortest path back
            # to the startup reading.  This preserves direction through the
            # encoder's 4095 -> 0 boundary and across multiple revolutions.
            step = _wrapped_count_delta(position, self._last_positions[servo_id])
            self._relative_counts[servo_id] += step
            self._last_positions[servo_id] = position
            angles.append(self._relative_counts[servo_id] / COUNTS_PER_RAD)
        return tuple(angles)

    def close(self) -> None:
        for servo_id in SERVO_IDS.values():
            try:
                self._servo.write1ByteTxRx(
                    self._port, servo_id, TORQUE_ENABLE_ADDRESS, 0
                )
            except Exception:
                pass
        try:
            self._port.closePort()
        except Exception:
            pass


def _joint_ids(robot, names: list[str]) -> list[int]:
    name_to_id = {name: idx for idx, name in enumerate(robot.data.joint_names)}
    resolved = [resolve_joint_name(robot, name) for name in names]
    return [name_to_id[name] for name in resolved]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    scene.update(sim_dt)

    arm_joints = LEFT_ARM_JOINTS if args_cli.arm == "left" else RIGHT_ARM_JOINTS
    controlled_names = arm_joints[:len(SERVO_IDS)]
    controlled_ids = _joint_ids(robot, controlled_names)
    actual_limits = robot.root_physx_view.get_dof_limits()[:, controlled_ids, :]
    bounds = joint_limits_deg(args_cli.arm)
    print(f"[INFO] Human-style limits (degrees): {dict(zip(SERVO_IDS, bounds))}; physics bounds={actual_limits.tolist()}", flush=True)
    signs = _parse_signs(args_cli.signs)
    print(f"[INFO] Active directions: {dict(zip(SERVO_IDS, signs))}", flush=True)
    if not 0.0 < args_cli.filter_alpha <= 1.0:
        raise ValueError("--filter-alpha must be in (0, 1]")

    default_pos = robot.data.default_joint_pos.clone()
    default_vel = robot.data.default_joint_vel.clone()
    # The canonical articulation pose bends joint4l by -75 degrees to avoid an
    # IK singularity.  This direct joint-space leader instead uses the physical
    # arm's fully straight resting pose as zero, so A-E all start at 0 degrees.
    neutral_pos = default_pos.clone()
    neutral_pos[:, controlled_ids] = 0.0
    robot.write_joint_state_to_sim(neutral_pos, default_vel)
    filtered_target = neutral_pos[:, controlled_ids].clone()

    if args_cli.verify_travel:
        for label, degrees in (("upper overshoot", [hi+5 for lo, hi in bounds]), ("return to zero", [0]*5), ("lower overshoot", [lo-5 for lo, hi in bounds]), ("return to zero", [0]*5)):
            target = neutral_pos.clone()
            target[:, controlled_ids] = torch.tensor([limited_target(math.radians(v), 1, 1, b) for v, b in zip(degrees, bounds)], device=sim.device)
            for _ in range(500):
                robot.set_joint_position_target(target)
                robot.set_joint_velocity_target(torch.zeros_like(default_vel))
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
            measured = robot.data.joint_pos[:, controlled_ids]
            error = torch.rad2deg((measured - target[:, controlled_ids]).abs()).max().item()
            print(f"[VERIFY] {label}: measured_deg={torch.rad2deg(measured).tolist()} max_error_deg={error:.2f}", flush=True)
            if not math.isfinite(error) or error > (1 if label == "return to zero" else 5):
                raise RuntimeError(f"Sim travel verification failed: {label}")
        robot.write_joint_state_to_sim(neutral_pos, default_vel)

    leader = ServoLeader(args_cli.port, args_cli.baud)
    should_rezero = False
    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0))

    def request_rezero() -> None:
        nonlocal should_rezero
        should_rezero = True

    keyboard.add_callback("R", request_rezero)
    controls_window = ui.Window("Leader Arm Controls", width=420, height=240)
    with controls_window.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("Hold the leader in the desired straight zero pose")
            ui.Button("ZERO LEADER", height=44, clicked_fn=request_rezero)
            for label, (lo, hi) in zip(SERVO_IDS, bounds):
                ui.Label(f"{label}: {lo} to {hi} degrees")
    print("[INFO] Five-joint leader teleop ready.", flush=True)
    print(
        "[INFO] Mapping: "
        + " | ".join(
            f"{label}(ID{SERVO_IDS[label]}) -> {joint}"
            for label, joint in zip(SERVO_IDS, controlled_names)
        ),
        flush=True,
    )
    print("[INFO] Move the leader by hand. Use ZERO LEADER or press R to re-zero.", flush=True)

    last_angles = (0.0,) * len(SERVO_IDS)
    last_report = time.monotonic()
    last_warning = 0.0
    window_samples = 0
    total_samples = 0
    serial_failures = 0
    try:
        while simulation_app.is_running():
            loop_started = time.monotonic()
            window_samples += 1
            total_samples += 1
            keyboard.advance()  # pumps callbacks; motion output is intentionally unused
            if should_rezero:
                leader.rezero()
                robot.write_joint_state_to_sim(neutral_pos, default_vel)
                robot.reset()
                filtered_target.copy_(neutral_pos[:, controlled_ids])
                last_angles = (0.0,) * len(SERVO_IDS)
                should_rezero = False
                print("[INFO] Leader and simulated arm re-zeroed.", flush=True)

            try:
                last_angles = leader.read_radians()
            except RuntimeError as exc:
                serial_failures += 1
                # A transient serial miss should hold the last safe target, not stop physics.
                now = time.monotonic()
                if now - last_warning >= 1.0:
                    print(f"[WARN] {exc}; holding last target", flush=True)
                    last_warning = now

            desired = neutral_pos[:, controlled_ids].clone()
            for axis, (joint_name, angle, sign) in enumerate(zip(controlled_names, last_angles, signs)):
                value = limited_target(angle, sign, args_cli.scale, bounds[axis])
                desired[0, axis] = value

            filtered_target.lerp_(desired, args_cli.filter_alpha)

            # Hold every non-driven joint at its canonical pose and command only
            # the first five arm joints from the leader.
            robot.set_joint_position_target(neutral_pos)
            robot.set_joint_velocity_target(torch.zeros_like(default_vel))
            robot.set_joint_position_target(filtered_target, joint_ids=controlled_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            now = time.monotonic()
            if now - last_report >= 0.5:
                report_elapsed = now - last_report
                loop_hz = window_samples / report_elapsed
                failure_rate = 100.0 * serial_failures / max(1, total_samples)
                degrees = [math.degrees(value) for value in last_angles]
                targets = [math.degrees(float(value)) for value in filtered_target[0]]
                actuals = [
                    math.degrees(float(value))
                    for value in robot.data.joint_pos[0, controlled_ids]
                ]
                errors = [target - actual for target, actual in zip(targets, actuals)]
                print(
                    f"\r[LEADER {loop_hz:5.1f}Hz serial_fail={failure_rate:.3f}%] "
                    + " ".join(
                        f"{label}={value:+6.1f}deg"
                        for label, value in zip(SERVO_IDS, degrees)
                    )
                    + "  ->  "
                    + " ".join(
                        f"{name}={value:+6.1f}deg"
                        for name, value in zip(controlled_names, targets)
                    )
                    + "  error: "
                    + " ".join(
                        f"{name}={error:+5.1f}deg"
                        for name, error in zip(controlled_names, errors)
                    ),
                    end="",
                    flush=True,
                )
                last_report = now
                window_samples = 0

            # Cap polling at 50 Hz, matching the standalone reader's cadence.
            # Only one process may own this serial port at a time.
            elapsed = time.monotonic() - loop_started
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)
    finally:
        leader.close()
        print("\n[INFO] Stopped. Leader torque is OFF.", flush=True)


def main() -> None:
    if args_cli.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args_cli.scene!r}; available: {list_scenes()}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view(*(scene_camera(args_cli.scene) or ([2.5, 2.5, 2.0], [0.0, 0.0, 0.8])))
    scene = InteractiveScene(
        make_scene_cfg(args_cli.scene, BIMANUAL_ARM_CFG, num_envs=1, env_spacing=2.0)
    )
    # Set the same bounds in USD before PhysX creates the articulation.
    from pxr import Usd, UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    names = (LEFT_ARM_JOINTS if args_cli.arm == "left" else RIGHT_ARM_JOINTS)[:len(SERVO_IDS)]
    limits_by_name = dict(zip(names, joint_limits_deg(args_cli.arm)))
    changed = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
        if prim.GetName() in names and prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            lo, hi = limits_by_name[prim.GetName()]
            joint.GetLowerLimitAttr().Set(float(lo))
            joint.GetUpperLimitAttr().Set(float(hi))
            changed.add(prim.GetName())
    if changed != set(names):
        raise RuntimeError(f"Missing joints while setting limits: {set(names) - changed}")
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
