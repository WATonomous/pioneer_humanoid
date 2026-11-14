"""Controller input adapter for the scene backends (mujoco/mujoco_sim.py, isaac/isaac_sim.py).

    No simulator launch or robot writes at import. The scene script owns all
    simulation, rendering, recording and reset operations; this turns headset
    packets into joint commands and serves camera frames back.
"""
import asyncio
import math
from dataclasses import replace
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

import input_filter
from control import fit_pivot, rotation_exp, rotation_vector
from article_control import AnchoredTarget
from mink_control import PioneerMinkIK
from transport import Shared, load_pivots, serve

DEFAULT_CALIBRATION = Path(__file__).with_name("wrist_calibration.json")


def add_controller_args(parser):
    """Controller flags common to every scene backend."""
    parser.add_argument("--gain", type=float, default=1.5,
                        help="Motion gain: metres of EE motion per metre of real wrist motion")
    parser.add_argument("--controller-port", type=int, default=8444)
    parser.add_argument("--joint-speed", type=float, default=.9,
                        help="Shoulder joints 1-3 command speed limit (rad/s); also the elbow's unless --elbow-speed")
    # Above the shoulder cap, the elbow takes more of each motion (scripted reaches: elbow share of the first 0.3 s
    # 42% -> 47% at 1.8 vs 0.9). A shoulder damping weight was tried for this and changed nothing up to 50x: the
    # position group is square (3 joints, 3-D wrist), so each step is near-exact.
    parser.add_argument("--elbow-speed", type=float, default=None,
                        help="Elbow joint 4 command speed limit (rad/s); above --joint-speed, the elbow leads the shoulder")
    parser.add_argument("--wrist-speed", type=float, default=3.6, help="Wrist joints 5-7 command speed limit (rad/s)")
    parser.add_argument("--rotation-gain", type=float, default=1., help="Controller rotation gain (1 = one-to-one)")
    parser.add_argument("--angular-lead", type=float, default=.6, help="Controller angular target lead limit (radians)")
    parser.add_argument("--position-lead", type=float, default=.08, help="Controller position target lead limit (m)")
    parser.add_argument("--ik-orientation-weight", type=float, default=0.2,
                        help="Orientation cost relative to position (1 rad ~ 1/w m); lower = position first")
    parser.add_argument("--body-keepout", action="store_true",
                        help="Keep the upper arm/forearm 2 cm off an approximate torso box (off: the stand has no torso)")
    parser.add_argument("--controller-frame", choices=("camera", "base"), default="camera",
                        help="Align controller motion/rotation with the headset view camera or upright robot base")
    parser.add_argument("--controller-calibration", type=Path, default=DEFAULT_CALIBRATION)
    input_filter.add_args(parser)


def check_controller_args(parser, args):
    """Reject non-positive or non-finite gains, speeds and leads (exits through parser.error)."""
    values = [args.gain, args.joint_speed, args.wrist_speed, args.rotation_gain, args.angular_lead, args.position_lead,
              args.ik_orientation_weight]
    if args.elbow_speed is not None:
        values.append(args.elbow_speed)
    if not all(math.isfinite(v) and v > 0 for v in values):
        parser.error("Controller gains, speeds and leads must be positive and finite")
    try:
        input_filter.from_args(args)
    except ValueError as error:
        parser.error(str(error))


class ControllerSceneDriver:
    def __init__(self, joint_names, gain, velocity, calibration, port=8444,
                 wrist_velocity=3.6, rotation_gain=1., angular_lead=.6, frame_calibration=None,
                 position_lead=.08, elbow_velocity=None, input_filter_factory=None, ik_orientation_weight=0.2,
                 body_keepout=False):
        """joint_names: {side: the 7 arm joints} of pioneer_bimanual_arm_8dof (arm_8dof_params)."""
        self.shared = Shared()
        # Optional controller-pose filter per arm (input_filter.OneEuroPose); None passes the raw pose through.
        self.filters = {s: input_filter_factory() if input_filter_factory else None for s in joint_names}
        self.solvers = {s: PioneerMinkIK(names, velocity, wrist_velocity, orientation_weight=ik_orientation_weight,
                                         body_keepout=body_keepout) for s, names in joint_names.items()}
        for solver in self.solvers.values():
            if elbow_velocity is not None:
                solver.velocity[3] = elbow_velocity
            # Joint limits: the URDF's (the owner's joint chart), plus one rule: the elbow stays >= 10 deg bent, off the
            # straight-arm singularity. No hand-picked teleop ranges: with 7 joints the home pull keeps poses natural
            # (the elbow came near straight 1 % of the time without the rule, 0 % with it).
            limits = solver.physical_limits.copy()
            limits[3, 0] = max(limits[3, 0], math.radians(10.0))   # joint 4 = elbow; 0 = straight
            solver.set_teleop_limits(limits)
        self.targets = {s: AnchoredTarget(gain=gain, rotation_gain=rotation_gain, max_angle_error=angular_lead,
                                          max_position_error=position_lead)
                        for s in joint_names}
        if frame_calibration is not None:
            for target in self.targets.values():
                target.set_calibration(frame_calibration)
        self.calibration = Path(calibration)
        self.pivots = load_pivots(self.calibration)
        for side, pivot in self.pivots.items():
            self.targets[side].pivot = pivot
        self.samples = {s: [] for s in joint_names}
        self.history = {s: [] for s in joint_names}   # the two newest raw poses per arm: [(Quest time s, Input)]
        self.calibrating = {s: False for s in joint_names}
        self.grippers = {s: 0. for s in joint_names}
        self.sequence = -1
        self.reset_sequence = 0
        self.record_presses = 0   # right-B presses so far (the scene backend toggles recording on each)
        self.status = "Tracked controllers move the arms. Left Y resets; stick-click pauses to calibrate."
        self.error = None
        self.shared.feedback = {"status": self.status}
        self.port = port
        self.server_task = None
        self.owner_thread = None
        self.io_pump = None

    @classmethod
    def from_args(cls, args, joint_names, frame_calibration=None):
        """Build the driver from the add_controller_args flags."""
        return cls(
            joint_names, args.gain, args.joint_speed, args.controller_calibration, args.controller_port,
            wrist_velocity=args.wrist_speed, rotation_gain=args.rotation_gain, angular_lead=args.angular_lead,
            frame_calibration=frame_calibration, position_lead=args.position_lead,
            elbow_velocity=args.elbow_speed, input_filter_factory=input_filter.from_args(args),
            ik_orientation_weight=args.ik_orientation_weight, body_keepout=args.body_keepout,
        )

    def start(self, schedule=None, pump=None):
        """Run the web server on Kit's main loop, never asyncio.run in a worker.

        Kit owns an asyncio engine tied to its application updates. Starting a
        second event-loop runner on a Python worker in the same process is an
        avoidable cross-thread integration hazard, even with no Quest connected.
        `schedule` is injectable for tests without importing/launching Isaac.
        """
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Isaac controller service must start on the main thread")
        if self.server_task is not None:
            raise RuntimeError("Controller service is already started")
        if schedule is None:
            from omni.kit.async_engine import run_coroutine
            schedule = run_coroutine
            loop = asyncio.get_event_loop()
            run_once = getattr(loop, "run_once", None)
            if callable(run_once):
                def pump():
                    # Kit's installed run_once is nonblocking and normally
                    # called only on app updates. Physics-only ticks omit those.
                    # Never nest it inside an already executing asyncio callback.
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        run_once()
            else:
                print("[Quest] Async run_once unavailable; input remains tied to Kit app-update cadence.", flush=True)
        self.io_pump = pump
        self.owner_thread = threading.get_ident()
        coroutine = serve(self.shared, SimpleNamespace(host="127.0.0.1", port=self.port, cert=None))
        try:
            self.server_task = schedule(coroutine)
        except Exception:
            coroutine.close()
            raise

    def start_thread(self):
        """Run the web server on its own event loop in a daemon thread (simulators without an asyncio engine).

        The simulation stays on the calling thread; only Shared crosses threads, under its lock.
        """
        if self.server_task is not None:
            raise RuntimeError("Controller service is already started")
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, name="quest-web", daemon=True).start()
        self.owner_thread = threading.get_ident()
        # A concurrent.futures.Future: done()/cancelled()/exception() are what update() polls.
        self.server_task = asyncio.run_coroutine_threadsafe(
            serve(self.shared, SimpleNamespace(host="127.0.0.1", port=self.port, cert=None)), loop)

    def _check_thread(self):
        if self.owner_thread is not None and threading.get_ident() != self.owner_thread:
            raise RuntimeError("Isaac controller updates must remain on the main thread")

    def _reset_filter(self, side):
        if self.filters[side] is not None:
            self.filters[side].reset()

    def reset(self, joint_positions):
        """Reset references/commands after the owner restores the whole scene."""
        self._check_thread()
        for side, solver in self.solvers.items():
            self._reset_filter(side)
            solver.command = np.array(joint_positions[side], dtype=float).copy()
            self.targets[side].lose_tracking()
            self.targets[side].position = self.targets[side].rotation = None
            self.targets[side].requested_rotation = None
            self.samples[side].clear()
            self.history[side].clear()
            self.calibrating[side] = False
            self.grippers[side] = 0.
        self.status = "Scene reset. Control resumes from the current arm pose when tracking returns."

    def update(self, joint_positions):
        """Consume latest input once. Return a latched, single-shot scene reset."""
        self._check_thread()
        if self.io_pump is not None:
            self.io_pump()
        if self.server_task is not None and self.server_task.done():
            if self.server_task.cancelled():
                raise RuntimeError("Controller service was cancelled")
            self.error = self.server_task.exception() or RuntimeError("Controller service stopped unexpectedly")
        if self.error is not None:
            raise RuntimeError(f"Controller server failed: {self.error}") from self.error
        with self.shared.lock:
            packet, received, sequence = self.shared.packet, self.shared.received, self.shared.sequence
            lost = self.shared.interrupted.copy()
            self.shared.interrupted = {s: False for s in self.targets}
            reset_sequence = self.shared.reset_sequence
            self.record_presses = self.shared.record_presses
        if reset_sequence != self.reset_sequence:
            self.reset_sequence = reset_sequence
            # Consume the frame containing Y; it must not move an arm after reset.
            self.sequence = sequence
            return True
        fresh = packet is not None and time.monotonic()-received <= .2
        for side, target in self.targets.items():
            if lost[side] or not fresh:
                target.lose_tracking()
                self._reset_filter(side)
                self.samples[side].clear()
                self.history[side].clear()
                self.calibrating[side] = False
        now = time.monotonic()
        if fresh and sequence != self.sequence:
            for side, solver in self.solvers.items():
                control = packet[side]
                sample = packet.get("sample")
                self.history[side] = (self.history[side][-1:] + [((sample["time_ms"] / 1e3) if sample else received, control)]
                                      if control is not None and not control.calibrate else [])
                target = self.targets[side]
                p, r, _, _ = solver.kinematics(joint_positions[side])
                if control is not None and control.calibrate:
                    if not self.calibrating[side]:
                        self.samples[side] = []
                    self.calibrating[side] = True
                    if len(self.samples[side]) < 1500:
                        self.samples[side].append((control.position.copy(), control.rotation.copy()))
                    target.lose_tracking()
                    self._reset_filter(side)
                    self.status = f"{side}: twist with wrist stationary; release stick-click to save"
                else:
                    if self.calibrating[side]:
                        self.calibrating[side] = False
                        if control is not None:
                            try:
                                offset, residual = fit_pivot(self.samples[side])
                                pivots = {**self.pivots, side: offset}
                                import json
                                self.calibration.parent.mkdir(parents=True, exist_ok=True)
                                temporary = self.calibration.with_suffix(".tmp")
                                temporary.write_text(json.dumps({s: v.tolist() for s, v in pivots.items()}))
                                temporary.replace(self.calibration)
                                self.pivots = pivots
                                target.pivot = offset
                                self.status = f"{side} calibrated (RMS {residual*1000:.1f} mm); tracking resumes automatically"
                            except (ValueError, OSError) as error:
                                self.status = f"{side}: {error}"
                        self.samples[side].clear()
                    if control is None:
                        self._reset_filter(side)
                    self._drive(side, control, packet["head"], p, r, now)
            self.sequence = sequence
        elif fresh and packet["head"] is not None:
            # Between packets (90 Hz headset, 100 Hz ticks) the arm targets move on along the hand's last velocity.
            # Holding them instead left a tenth to a third of ticks with no motion: stop-go, felt as clicking (profiled).
            for side, solver in self.solvers.items():
                control = self._predict(side, now - received)
                if control is not None and self.targets[side].active:
                    p, r, _, _ = solver.kinematics(joint_positions[side])
                    self._drive(side, control, packet["head"], p, r, now)
        with self.shared.lock:
            self.shared.feedback = {
                "status": self.status if fresh else "Tracking paused. Control resumes from the current arm pose.",
                "active": {s: t.active for s, t in self.targets.items()},
            }
        return False

    def _predict(self, side, elapsed):
        """The raw pose `elapsed` s after the newest packet, from the last two: at most one packet interval ahead."""
        if len(self.history[side]) < 2:
            return None
        (t0, a), (t1, b) = self.history[side]
        if not 0 < t1 - t0 < .05:
            return None
        k = min(max(elapsed, 0.) / (t1 - t0), 1.)
        return replace(b, position=b.position + k * (b.position - a.position),
                       rotation=rotation_exp(k * rotation_vector(b.rotation @ a.rotation.T)) @ b.rotation)

    def _drive(self, side, control, head, p, r, now):
        """Filter the controller pose and move this arm's target (a None control loses tracking)."""
        target = self.targets[side]
        if control is not None and self.filters[side] is not None:
            # Smooth Quest jitter and cap one-frame tracking jumps before the mapping (input_filter.py). Timed by
            # control tick, not packet arrival: every tick now carries a pose, so the filter sees an even rate.
            position, rotation = self.filters[side](control.position, control.rotation, now)
            control = replace(control, position=position, rotation=rotation)
        if control is not None and head is not None:
            # Reuse the same incremental target mapping. The first tracked frame anchors to measured FK, including
            # after loss/reset/calibration. Never mutate the shared packet.
            target.armed = True
            control = replace(control, clutch=True)
        target.update(control, head, p, r)
        if target.active:
            self.grippers[side] = control.trigger

    def solve(self, joint_positions, dt):
        self._check_thread()
        return {s: solver.solve(joint_positions[s], self.targets[s], dt) for s, solver in self.solvers.items()}

    def close(self):
        self._check_thread()
        self.shared.stop.set()
        if self.server_task is not None and not self.server_task.done():
            # Kit owns pumping and cleanup. Do not run/nest/close its loop here.
            self.server_task.cancel()
