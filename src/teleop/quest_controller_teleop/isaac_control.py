"""Controller input adapter for the existing Isaac lightbox/push-box scene.

    No simulator launch or robot writes at import. The original scene owns all
    simulation, rendering, recording and reset operations.
"""
import io
import asyncio
from dataclasses import replace
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from control import fit_pivot, rotation_exp, rotation_vector
from article_control import ArticleTarget, AnchoredTarget, PioneerArticleIK
from transport import Shared, load_pivots, serve


class ControllerSceneDriver:
    def __init__(self, joint_names, gain, velocity, calibration, port=8444,
                 wrist_velocity=3.6, rotation_gain=1., angular_lead=.6, frame_calibration=None,
                 input_filter_factory=None, anchored=False, predict_input=False):
        self.shared = Shared()
        self.solvers = {s: PioneerArticleIK(names, velocity, wrist_velocity)
                        for s, names in joint_names.items()}
        target_class = AnchoredTarget if anchored else ArticleTarget
        self.filters = {s: input_filter_factory() if input_filter_factory else None for s in joint_names}
        self.history = {s: [] for s in joint_names}
        self.predict_input = predict_input
        self.record_presses = 0
        self.targets = {s: target_class(gain=gain, rotation_gain=rotation_gain, max_angle_error=angular_lead)
                        for s in joint_names}
        if frame_calibration is not None:
            for target in self.targets.values():
                target.set_calibration(frame_calibration)
        self.calibration = Path(calibration)
        self.pivots = load_pivots(self.calibration)
        for side, pivot in self.pivots.items():
            self.targets[side].pivot = pivot
        self.samples = {s: [] for s in joint_names}
        self.calibrating = {s: False for s in joint_names}
        self.grippers = {s: 0. for s in joint_names}
        self.sequence = -1
        self.reset_sequence = 0
        self.status = "Tracked controllers move the arms. Left Y resets; stick-click pauses to calibrate."
        self.error = None
        self.shared.feedback = {"scene": "isaac", "status": self.status}
        self.port = port
        self.server_task = None
        self.owner_thread = None
        self.io_pump = None

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
        elif self.predict_input and fresh and packet["head"] is not None:
            # Between packets (90 Hz headset, 100 Hz ticks) the arm targets move on along the hand's last velocity.
            # Holding them instead left a tenth to a third of ticks with no motion: stop-go, felt as clicking (profiled).
            for side, solver in self.solvers.items():
                control = self._predict(side, now - received)
                if control is not None and self.targets[side].active:
                    p, r, _, _ = solver.kinematics(joint_positions[side])
                    self._drive(side, control, packet["head"], p, r, now)
        with self.shared.lock:
            self.shared.feedback = {
                "scene": "isaac",
                "status": self.status if fresh else "Tracking paused. Control resumes from the current arm pose.",
                "active": {s: t.active for s, t in self.targets.items()},
                "haptics": {s: self.solvers[s].warning if t.active else 0. for s, t in self.targets.items()},
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

    def publish_cameras(self, left_eye, right_eye, left_wrist, right_wrist):
        self._check_thread()
        from PIL import Image
        images = {}
        for name, frame in (("pov_left", left_eye), ("pov_right", right_eye),
                            ("wrist_left", left_wrist), ("wrist_right", right_wrist)):
            output = io.BytesIO()
            Image.fromarray(frame).save(output, format="JPEG", quality=80)
            images[name] = output.getvalue()
        images["ego"] = images["pov_left"]
        with self.shared.lock:
            self.shared.images = images

    def close(self):
        self._check_thread()
        self.shared.stop.set()
        if self.server_task is not None and not self.server_task.done():
            # Kit owns pumping and cleanup. Do not run/nest/close its loop here.
            self.server_task.cancel()
