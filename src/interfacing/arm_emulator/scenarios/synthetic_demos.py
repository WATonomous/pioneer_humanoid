"""Scripted demos through the real stack: the synthetic operator -> ArmPose -> real joint_command -> emulator.

Same recorder, topics and features as a human take with humanoid-record. Runs in real time, so scale up by
running one per job on WATcloud (each container has its own ROS graph) and pointing --dataset_root apart.

    python3 scenarios/synthetic_demos.py --episodes 20 --gripper-force-scale 2 --dataset_root datasets/.../sim_low_level/synthetic_0
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from arm_emulator import config  # noqa: F401 -- sys.path for the arm scripts and scenes
from harness import HOME_DEG, Client, Stack

_SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_SRC / "teleop" / "pioneer_leader_arm_teleop" / "synthetic"))
sys.path.insert(0, str(_SRC / "robot_learning"))

CONTROL_DT = 0.02
SCHEMA = _SRC / "robot_learning" / "config" / "pioneer_emulator_dataset_schema.yaml"
HOME_SECONDS = 4.0
TASK_TEXT = "pick up the cube"
log = logging.getLogger("synthetic_demos")


class Mirror:
    """The operator's view of the scene: a local copy of the model, posed from /arm_emulator/qpos."""

    def __init__(self, client: Client, scene: str) -> None:
        from humanoid_mujoco_scenes import make_model
        from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, LEFT_GRIPPER_JOINTS
        from std_msgs.msg import Float64MultiArray

        self.model = make_model(scene, torque_joints=[*LEFT_ARM_JOINTS, *LEFT_GRIPPER_JOINTS])
        self.data = mujoco.MjData(self.model)
        self._lock = threading.Lock()
        self._qpos: list[float] | None = None
        client.node.create_subscription(Float64MultiArray, "/arm_emulator/qpos", self._on_qpos, 10)

    def _on_qpos(self, msg) -> None:
        with self._lock:
            self._qpos = list(msg.data)

    def look(self) -> mujoco.MjData:
        with self._lock:
            if self._qpos is not None:
                self.data.qpos[:] = self._qpos
        mujoco.mj_kinematics(self.model, self.data)
        return self.data


class Halted(RuntimeError):
    """joint_command reported a fault: nothing more can be commanded until it restarts."""


def play_take(client: Client, stack: Stack, mirror: Mirror, seed: int) -> tuple[bool, list[str]]:
    from human_operator import HumanOperator
    from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS
    from synthetic_tasks import TASKS, TaskFailed

    def tick(q_rad, grip: float) -> None:
        if stack.joint_command_says("MIT FAULT"):
            raise Halted(next(line for line in stack.joint_command_log.read_text().splitlines() if "MIT FAULT" in line))
        client.send(np.degrees(q_rad), grip)
        time.sleep(CONTROL_DT)  # the operator's own pace: joint_command runs on its own clock
        mirror.look()

    rng = np.random.default_rng([seed, 1])
    op = HumanOperator(mirror.model, mirror.look, tick, LEFT_ARM_JOINTS, np.radians(HOME_DEG), CONTROL_DT, rng)
    notes: list[str] = []
    try:
        TASKS["rubiks_cube"].plan(op, mirror.model, mirror.look(), notes)
    except TaskFailed as e:
        notes.append(str(e))
        return False, notes
    time.sleep(0.3)  # let the status topic catch up with the lift
    return bool(client.status.get("success")), notes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-tries", type=int, default=None, help="default 3 x --episodes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schema", default=str(SCHEMA))
    p.add_argument("--cameras", default=None, help="comma-separated (default: the schema's enabled images)")
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--gripper-force-scale", type=float, default=None)
    args = p.parse_args()

    from humanoid_robot_learning.recording.record_gripper import GripperRecordBuffer
    from humanoid_robot_learning.recording.record_utils import resolve_dataset_root
    from humanoid_robot_learning.recording.recorder import RecordSession, RecordSettings
    from humanoid_robot_learning.recording.schema import enabled_images, load_yaml, select_arms, select_cameras
    from humanoid_robot_learning.recording.sinks import create_sinks

    cfg = select_arms(select_cameras(load_yaml(Path(args.schema)), args.cameras), ["left"])
    extra = ["--gripper-force-scale", str(args.gripper_force_scale)] if args.gripper_force_scale is not None else []
    stack = Stack(cameras=",".join(enabled_images(cfg)) or "none", seed=args.seed, extra_args=extra)
    client = Client(stack)
    mirror = Mirror(client, "rubiks_cube")
    buffer = GripperRecordBuffer(client.node, cfg)
    root = resolve_dataset_root(cfg, args.dataset_root, subdir=cfg["record"].get("subdir", "real"))
    output_dir, sinks = create_sinks(cfg, record_root=root, sink_names=["lerobot"])
    settings = RecordSettings(num_episodes=args.episodes, task_description=TASK_TEXT, use_keyboard=False)
    session = RecordSession(cfg, sinks, settings, output_dir=output_dir)

    recording = threading.Event()
    stop = threading.Event()
    frames = [0, 0]  # written, skipped

    def frame_loop() -> None:
        period, next_t = 1.0 / float(cfg["fps"]), time.monotonic()
        while not stop.is_set():
            next_t += period
            if recording.is_set():
                try:
                    session.ingest_snapshot(buffer.snapshot())
                    frames[0] += 1
                except ValueError:
                    frames[1] += 1
            time.sleep(max(0.0, next_t - time.monotonic()))

    recorder_thread = threading.Thread(target=frame_loop, daemon=True)
    recorder_thread.start()

    saved, tried, code = 0, 0, 0
    try:
        client.measured()
        while saved < args.episodes and tried < (args.max_tries or 3 * args.episodes):
            seed = args.seed + tried
            tried += 1
            client.control(f"reset_objects {seed}")
            client.ramp(client.measured(), HOME_DEG, HOME_SECONDS, grip_from=client.closure(), grip_to=0.0)
            client.hold(0.5, HOME_DEG)
            time.sleep(0.3)
            mirror.look()
            session.begin_episode()
            recording.set()
            ok, notes = play_take(client, stack, mirror, seed)
            recording.clear()
            time.sleep(1.5 / float(cfg["fps"]))
            if ok:
                session.flags.success = True
                session.save_episode_if_ready()
                saved += 1
            else:
                session.clear_episode_buffers()
            log.info("take %d (seed %d): %s %s; saved %d/%d; frames written %d, skipped %d", tried, seed,
                     "kept" if ok else "discarded", notes or "", saved, args.episodes, *frames)
        if saved < args.episodes:
            log.error("only %d/%d takes succeeded in %d tries", saved, args.episodes, tried)
            code = 1
    except Halted as e:
        log.error("joint_command halted the arm, stopping: %s", e)
        code = 2
    finally:
        stop.set()
        recorder_thread.join(timeout=2.0)
        session.finalize()
        status = client.status
        log.info("dataset: %s. timing: late %s of %s ticks; breaches: %s", output_dir, status.get("late_ticks"),
                 status.get("steps"), status.get("breaches"))
        client.close()
        stack.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
