"""Quest v2 human expert for cargo HG-DAgger; no simulator launch or actuator writes.

The imported driver owns WebXR transport and IK. CargoEnv remains the sole robot
writer, and HumanGate remains the sole label admission point.
"""
from pathlib import Path
import sys
import time

import numpy as np

VR_ROOT = Path(__file__).resolve().parents[4] / "src/teleop/quest_controller_teleop"
sys.path.insert(0, str(VR_ROOT))
from isaac_control import ControllerSceneDriver
from control import quaternion_matrix
from input_filter import OneEuroPose
from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS


class VrInputUnavailable(RuntimeError):
    """Expected input timeout; hold physics and omit this frame's label."""


class VrExpert:
    def __init__(self, env, port=8444, gain=1.0, calibration=None, driver=None):
        self.right_ids = [env.robot.joint_names.index(j) for j in RIGHT_ARM_JOINTS]
        contract = env.contract()
        forward = np.array(contract["ego_target"]) - contract["ego_eye"]
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        camera_rotation = np.column_stack((right, np.cross(right, forward), -forward))
        root_quat = env.robot.data.root_quat_w[0].cpu().numpy()
        camera_in_base = quaternion_matrix(root_quat[[1, 2, 3, 0]]).T @ camera_rotation
        self.driver = driver or ControllerSceneDriver(
            {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS}, gain, .9,
            calibration or VR_ROOT / "wrist_calibration.json", port,
            wrist_velocity=1.5, frame_calibration=camera_in_base,
            input_filter_factory=OneEuroPose, anchored=True, predict_input=True)
        self.grip = 0.
        self.grip_matched = False
        self.button_synced = False
        self.right_grip = False
        self.ready = False
        self.waiting_for_start = True
        self.reset_requested = False
        self.save_requested = False
        self.record_presses = 0
        self.waiting_for_release = False
        self.fine = False  # keyboard shift is unused; X supplies VR precision mode

    def positions(self, env):
        return {"left": env.robot.data.joint_pos[0, env.arm_ids].cpu().numpy().copy(),
                "right": env.robot.data.joint_pos[0, self.right_ids].cpu().numpy().copy()}

    def start(self):
        self.driver.start()

    def seed(self, env):
        self.driver.reset(self.positions(env))
        self.grip = float(env.last[6])
        self.driver.grippers["left"] = self.grip
        self.grip_matched = False
        self.ready = False

    def poll(self, env):
        """Pump input even in policy/pause mode. Return explicit authority edges."""
        self.reset_requested = self.driver.update(self.positions(env))
        self.save_requested = False
        with self.driver.shared.lock:
            packet = self.driver.shared.packet
            fresh = packet is not None and time.monotonic() - self.driver.shared.received <= .2
            presses = self.driver.shared.record_presses
        pressed = presses > self.record_presses
        self.record_presses = presses
        # B is a latched UI event, not a pose command. A valid button press can
        # arrive just before the latest mailbox pose loses tracking or expires.
        self.save_requested = bool(pressed and not self.waiting_for_start and not self.waiting_for_release)
        self.ready = bool(fresh and self.driver.targets["left"].active and packet["right"] is not None)
        right = packet["right"] if fresh and packet["head"] is not None else None
        if right is None:
            self.button_synced = False
            return None
        held = bool(right.clutch)
        if self.waiting_for_release:
            self.button_synced, self.right_grip = True, held
            if self.is_ready() and not held:
                self.waiting_for_release = False
                return "POLICY"
            return None
        authority = None
        if not self.button_synced:
            self.button_synced, self.right_grip = True, held
        else:
            if held != self.right_grip:
                self.right_grip = held
                authority = "HUMAN" if held else "POLICY"
        if self.waiting_for_start and self.is_ready() and not held and not packet["left"].clutch:
            self.waiting_for_start = False
        return authority

    def action(self, env):
        if self.waiting_for_release or not self.is_ready():
            raise VrInputUnavailable("VR input unavailable; do not execute or label stale input")
        commands = self.driver.solve(self.positions(env), .04)
        trigger = float(self.driver.grippers["left"])
        # A relaxed trigger at takeover must not drop a bag already being carried.
        # Match the current command before enabling proportional trigger control.
        if abs(trigger - self.grip) <= .15:
            self.grip_matched = True
        if self.grip_matched:
            self.grip = trigger
        return np.r_[commands["left"], self.grip].astype(np.float32)

    def is_ready(self):
        with self.driver.shared.lock:
            packet = self.driver.shared.packet
            return bool(self.ready and packet is not None
                        and time.monotonic() - self.driver.shared.received <= .2
                        and packet["head"] is not None and packet["left"] is not None
                        and packet["right"] is not None and not packet["left"].calibrate
                        and not any(self.driver.shared.interrupted.values()))

    def publish(self, obs, gate):
        ego, wrist = obs["observation.images.ego"], obs["observation.images.wrist_left"]
        # Reuse the policy cameras. The headset overview is monoscopic, and the
        # two wrist panels both show the controlled left wrist.
        self.driver.publish_cameras(ego, ego, wrist, wrist)
        self.feedback(gate)

    def feedback(self, gate):
        status = f"Cargo HG-DAgger: {gate.mode}"
        if gate.notice:
            status += f" | {gate.notice}"
        if self.waiting_for_start:
            status += " | WAITING FOR VR: Enter VR, track both controllers and release both grips"
        elif self.driver.shared.record_button_supported is False:
            status += " | REFRESH QUEST BROWSER PAGE: old page is not sending the RIGHT B button"
        elif self.waiting_for_release:
            status += " | Release RIGHT grip to start the next attempt"
        elif gate.buffer_full:
            status += " | BUFFER FULL: RIGHT B saves; LEFT Y discards/resets; physics held"
        elif gate.paused:
            status += " | PAUSED"
        elif gate.mode == "HUMAN" and not self.ready:
            status += " | WAITING FOR TRACKING / LEFT CALIBRATION; physics and recording held"
        elif gate.mode == "HUMAN" and not self.grip_matched:
            status += " | Match left trigger to held grip before changing it"
        status += (f" | Saved: {gate.saved_segments} segments | Unsaved: {gate.pending_frames} frames; "
                   "RIGHT B save+reset / LEFT Y discard+reset")
        with self.driver.shared.lock:
            self.driver.shared.feedback.update(
                status=status, right_b_presses=self.record_presses, unsaved_frames=gate.pending_frames,
                saved_segments=gate.saved_segments,
                vr_client_has_save_button=self.driver.shared.record_button_supported, controls=(
                "Cargo left arm only. Hold RIGHT grip to intervene; release returns to policy WITHOUT saving. "
                "LEFT controller moves arm; LEFT index trigger grips; X = precision. "
                "RIGHT B saves unsaved corrections AND resets; LEFT Y discards them AND resets. "
                "Keyboard H toggles authority, N saves, P pauses, B resets. "
                "Headset overview is monoscopic; both wrist panels show the left wrist."))

    def close(self):
        self.driver.close()
