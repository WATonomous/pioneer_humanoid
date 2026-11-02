"""Run the emulator: 1 kHz physics in real time, the bridge to ROS, the cameras and, optionally, a viewer.

    python3 -m arm_emulator.run_emulator --scene rubiks_cube [--viewer] [--cameras wrist_left,ego|none]

Three processes, so none can hold another back: physics and drives (this file's physics_main), the ROS bridge
(bridge_node.py) and the cameras (camera.py).

Keys in the viewer window: 6 = a new cube position. The arm isn't reset: bring it home with the leader.
Falling behind real time is logged and counted (status topic, `late_ticks`): a slow machine changes the
control timing, so results from such a run don't count.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import sys
import threading
import time

from arm_emulator import config, faults, safety_guard

_KEY_NEW_LAYOUT = ord("6")  # GLFW key code; the same key resets the scene in the MuJoCo teleop
LATE_S = 0.002              # behind the wall clock by more than this: the tick counts as late
MAX_CATCH_UP_S = 0.05       # further behind than this: skip ahead rather than run fast to catch up
REPORT_PERIOD_S = 5.0
STATUS_PERIOD_S = 0.1
QPOS_PERIOD_S = 0.02
DISCOVERY_WAIT_S = 2.0      # before the first command: long enough to see a real can_node on the graph
PHYSICS_START_TIMEOUT_S = 60.0
START_POSES = {"home": [0.0, 0.0, 0.0, -90.0, 0.0, 0.0], "hanging": [0.0] * 6}  # deg, command frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emulate the arm below joint_command in MuJoCo, in real time.")
    p.add_argument("--scene", default="rubiks_cube")
    p.add_argument("--cameras", default="wrist_left,ego", help="comma-separated, or none")
    p.add_argument("--image-size", default="480x640", help="HxW")
    p.add_argument("--camera-hz", type=float, default=30.0)
    p.add_argument("--viewer", action="store_true", help="open the MuJoCo window (needs a display)")
    p.add_argument("--seed", type=int, default=0)
    # Hanging, the forearm is under the table's edge: raising the elbow from there hits the table.
    p.add_argument("--start-pose", choices=sorted(START_POSES), default="home",
                   help="where the arm is, held by hand until the first stiff command")
    p.add_argument("--gripper-force-scale", type=float, default=None,
                   help="the linkage's force advantage over the linear mapping (motor_models.yaml; measure it on the bench)")
    p.add_argument("--fault", action="append", default=[], help=f"inject at start: kind:motor_id[:value], {sorted(faults.KINDS)}")
    p.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    return p.parse_args(argv)


class PhysicsLoop:
    """Advances physics with the wall clock, one tick at a time, and answers the bridge between ticks."""

    def __init__(self, arm, pipe, viewer=None) -> None:
        self.arm, self.pipe, self.viewer = arm, pipe, viewer
        self.late, self.skipped, self.worst = 0, 0, 0.0
        self.stopping = False

    def _serve(self) -> None:
        """Everything waiting on the pipe. A command's reply goes straight back, as a drive answers on the bus."""
        while self.pipe.poll():
            kind, *rest = self.pipe.recv()
            if kind == "command":
                reply = self.arm.command(*rest)
                if reply is not None:
                    self.pipe.send(("feedback", reply))
            elif kind == "reset_objects":
                self.arm.reset_objects(rest[0])
            elif kind == "fault":
                self.arm.inject(rest[0])
            elif kind == "stop":
                self.stopping = True

    def _status(self) -> dict:
        arm = self.arm
        return {"sim_time": float(arm.data.time), "wall_time": time.monotonic(), "success": arm.success(),
                "closure": arm.closure(), "breaches": list(arm.breaches), "peak_contact_n": float(arm.peak_contact),
                "peak_torque": {s.name: float(arm.peak_torque[s.motor_id]) for s in arm.specs},
                "late_ticks": self.late, "skipped_ticks": self.skipped, "worst_behind_s": self.worst, "steps": arm.steps}

    def run(self, stop_after: float | None) -> None:
        arm, viewer = self.arm, self.viewer
        start = time.monotonic()
        next_report, next_status, next_qpos = start + REPORT_PERIOD_S, start, start
        while not self.stopping and (viewer is None or viewer.is_running()):
            now = time.monotonic()
            if stop_after is not None and now - start > stop_after:
                return
            self._serve()
            behind = now - start - arm.steps * arm.dt
            if behind < 0.0:
                time.sleep(min(-behind, 0.0005))
                continue
            if behind > LATE_S:
                self.late += 1
                self.worst = max(self.worst, behind)
            if behind > MAX_CATCH_UP_S:
                lost = int(behind / arm.dt)
                self.skipped += lost
                start += lost * arm.dt
            if viewer is not None:
                with viewer.lock():
                    arm.step()
                if arm.steps % 16 == 0:
                    viewer.sync()
            else:
                arm.step()
            if now >= next_qpos:
                next_qpos = now + QPOS_PERIOD_S
                self.pipe.send(("qpos", arm.qpos_copy().tolist()))
            if now >= next_status:
                next_status = now + STATUS_PERIOD_S
                self.pipe.send(("status", self._status()))
            if now >= next_report:
                next_report = now + REPORT_PERIOD_S
                if self.late:
                    print(f"[EMULATOR] behind real time: {self.late} late ticks, worst {self.worst * 1e3:.1f} ms, "
                          f"{self.skipped} skipped, of {arm.steps}", flush=True)


def physics_main(pipe, args: argparse.Namespace) -> None:
    """The physics process."""
    if not args.viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco.viewer

    from arm_emulator.physics import EmulatedArm
    from humanoid_mujoco_scenes import list_scenes, scene_camera

    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")
    arm = EmulatedArm(args.scene, seed=args.seed, start_pose_deg=START_POSES[args.start_pose], supported=True)
    if args.gripper_force_scale is not None:
        arm.grip_force_scale = args.gripper_force_scale
    print(f"[EMULATOR] gripper force scale {arm.grip_force_scale:g} (ASSUMED until measured)", flush=True)
    for spec in args.fault:
        arm.inject(faults.parse(spec))

    def on_key(key: int) -> None:
        if key == _KEY_NEW_LAYOUT:
            arm.reset_objects()
            print("[EMULATOR] new object layout", flush=True)

    pipe.send(("ready",))
    try:
        if args.viewer:
            with mujoco.viewer.launch_passive(arm.model, arm.data, key_callback=on_key) as viewer:
                for key, value in (scene_camera(args.scene) or {}).items():
                    setattr(viewer.cam, key, value)
                loop = PhysicsLoop(arm, pipe, viewer)
                loop.run(args.duration)
        else:
            loop = PhysicsLoop(arm, pipe)
            loop.run(args.duration)
    except KeyboardInterrupt:
        return  # the parent reports; this is the same Ctrl-C it got
    except (BrokenPipeError, EOFError):
        return
    print(f"[EMULATOR] stopped. {loop.late} late ticks, worst {loop.worst * 1e3:.1f} ms, {loop.skipped} skipped, of "
          f"{arm.steps}. Peak contact {arm.peak_contact:.0f} N. Breaches: {arm.breaches or 'none'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    failures = safety_guard.environment_failures()
    if failures:
        raise SystemExit("[REFUSED] not starting the emulator:\n  " + "\n  ".join(failures))

    print("[EMULATOR] Results depend on these placeholders (config/motor_models.yaml):")
    for line in config.assumptions():
        print(f"  {line}")

    context = multiprocessing.get_context("spawn")  # ROS and GL state must not be inherited
    pipe, child_pipe = context.Pipe()
    physics = context.Process(target=physics_main, args=(child_pipe, args), name="emulator-physics")
    physics.start()
    child_pipe.close()

    import rclpy

    from arm_emulator.bridge_node import BridgeNode

    rclpy.init()
    node = BridgeNode(pipe)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True, name="emulator-ros").start()
    camera = None
    try:
        if not node.ready.wait(PHYSICS_START_TIMEOUT_S):
            raise SystemExit("[REFUSED] the physics process did not start")
        time.sleep(DISCOVERY_WAIT_S)
        node.check_graph()
        if node.refusal is not None:
            raise SystemExit(f"[REFUSED] {node.refusal}")
        names = [] if args.cameras == "none" else args.cameras.split(",")
        # Its own process: see camera.py.
        camera = None if not names else subprocess.Popen(
            [sys.executable, "-m", "arm_emulator.camera", "--scene", args.scene, "--cameras", ",".join(names),
             "--image-size", args.image_size, "--hz", str(args.camera_hz)])
        print(f"[EMULATOR] {args.scene}: running. Cameras: {names or 'none'}. Waiting for joint_command.", flush=True)
        while physics.is_alive():
            if node.refusal is not None:
                raise SystemExit(f"[REFUSED] {node.refusal}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pipe.send(("stop",))
        except OSError:
            pass
        physics.join(timeout=5.0)
        if physics.is_alive():
            physics.terminate()
        if camera is not None:
            camera.terminate()
            camera.wait(timeout=5.0)
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
