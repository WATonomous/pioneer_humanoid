"""Small Isaac UI windows for leader-arm tuning and the wrist-camera preview."""
from __future__ import annotations

import math
from collections import deque

import omni.ui as ui

from servo_leader import ARM_SERVOS, GRIPPER_SERVO, SERVO_IDS


class LeaderControlWindow:
    """Live controls whose actions are consumed safely by the simulation loop."""

    def __init__(self, joint_names: list[str], signs: list[float], arm_defaults_rad: list[float], grip: float):
        self._actions: deque[tuple] = deque()
        self._current_labels = []
        self._target_labels = []
        self._invert_models = [ui.SimpleBoolModel(sign < 0.0) for sign in signs]
        self._default_models = [ui.SimpleFloatModel(math.degrees(value)) for value in arm_defaults_rad]
        self._grip_default_model = ui.SimpleFloatModel(float(grip))

        self.window = ui.Window("Leader Arm Controls", width=650, height=425)
        with self.window.frame:
            with ui.VStack(spacing=6, style={"margin": 8}):
                ui.Label(
                    "Live tuning — changes apply without restarting Isaac",
                    height=24,
                    style={"font_size": 18},
                )
                ui.Label(
                    "Direction changes preserve the current pose. Default is the sim angle assigned to the pose you are holding.",
                    height=32,
                    word_wrap=True,
                )
                with ui.HStack(height=22):
                    ui.Label("Motor / joint", width=180)
                    ui.Label("Leader", width=82)
                    ui.Label("Target", width=82)
                    ui.Label("Inverted", width=75)
                    ui.Label("Default", width=100)

                for index, (label, joint) in enumerate(zip(ARM_SERVOS, joint_names)):
                    with ui.HStack(height=28, spacing=5):
                        ui.Label(f"{label}  ID{SERVO_IDS[label]}  {joint}", width=180)
                        current = ui.Label("+0.0°", width=82)
                        target = ui.Label("+0.0°", width=82)
                        ui.CheckBox(model=self._invert_models[index], width=75)
                        ui.FloatDrag(
                            model=self._default_models[index],
                            min=-180.0,
                            max=180.0,
                            step=1.0,
                            format="%.1f°",
                            width=100,
                        )
                        self._current_labels.append(current)
                        self._target_labels.append(target)

                with ui.HStack(height=28, spacing=5):
                    ui.Label(f"{GRIPPER_SERVO}  ID{SERVO_IDS[GRIPPER_SERVO]}  gripper", width=180)
                    current = ui.Label("+0.0°", width=82)
                    target = ui.Label("0%", width=82)
                    ui.CheckBox(model=self._invert_models[-1], width=75)
                    ui.FloatDrag(
                        model=self._grip_default_model,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        format="%.2f",
                        width=100,
                    )
                    self._current_labels.append(current)
                    self._target_labels.append(target)

                with ui.HStack(height=32, spacing=6):
                    ui.Button("Apply directions", clicked_fn=self._queue_directions)
                    ui.Button("Held pose → defaults", clicked_fn=self._queue_defaults)
                    ui.Button("Held pose → all zero", clicked_fn=self._queue_zero)
                    ui.Button("Use saved calibration", clicked_fn=self._queue_clear)

                self._status = ui.Label(
                    "Ready. S starts a demo · N saves it · D discards it · R resets the cube.",
                    height=34,
                    word_wrap=True,
                )

    def _signs(self) -> list[float]:
        return [-1.0 if model.get_value_as_bool() else 1.0 for model in self._invert_models]

    def _defaults(self) -> tuple[list[float], float]:
        arm = [model.get_value_as_float() for model in self._default_models]
        return arm, self._grip_default_model.get_value_as_float()

    def _queue_directions(self) -> None:
        self._actions.append(("directions", self._signs()))

    def _queue_defaults(self) -> None:
        arm, grip = self._defaults()
        self._actions.append(("defaults", self._signs(), arm, grip))

    def _queue_zero(self) -> None:
        for model in self._default_models:
            model.set_value(0.0)
        self._grip_default_model.set_value(0.0)
        self._actions.append(("defaults", self._signs(), [0.0] * len(ARM_SERVOS), 0.0))

    def _queue_clear(self) -> None:
        self._actions.append(("clear", self._signs()))

    def pop_action(self) -> tuple | None:
        return self._actions.popleft() if self._actions else None

    def set_status(self, message: str) -> None:
        self._status.text = message

    def update(self, angles: tuple[float, ...], arm_targets: list[float], grip: float) -> None:
        for label, angle in zip(self._current_labels, angles):
            label.text = f"{math.degrees(angle):+6.1f}°"
        for label, target in zip(self._target_labels[:-1], arm_targets):
            label.text = f"{math.degrees(target):+6.1f}°"
        self._target_labels[-1].text = f"{grip * 100:5.0f}%"

    def close(self) -> None:
        self.window.visible = False


class WristCameraWindow:
    """Display the existing Isaac camera tensor without creating another RTX viewport."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.provider = ui.ByteImageProvider()
        self.provider.set_bytes_data(
            bytearray((0, 0, 0, 255)) * (self.width * self.height),
            [self.width, self.height],
        )
        # Keep this deliberately small: it is a framing aid, not another main viewport.
        display_width = min(400, self.width)
        display_height = round(display_width * self.height / self.width)
        self.window = ui.Window("Left Wrist Camera", width=display_width + 16, height=display_height + 42)
        with self.window.frame:
            ui.ImageWithProvider(
                self.provider,
                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                pixel_aligned=True,
            )

    def update(self, rgba) -> None:
        """Accept an HxWx3/4 uint8 torch tensor from Isaac Lab's Camera sensor."""
        frame = rgba.detach()
        if frame.shape[-1] == 3:
            import torch

            alpha = torch.full((*frame.shape[:-1], 1), 255, dtype=frame.dtype, device=frame.device)
            frame = torch.cat((frame, alpha), dim=-1)
        frame = frame.contiguous().cpu()
        self.provider.set_bytes_data(bytearray(frame.numpy().tobytes()), [self.width, self.height])

    def close(self) -> None:
        self.window.visible = False


class RecordingStatusWindow:
    """Large unambiguous recorder state that never stops at an episode count."""

    _READY = {"background_color": 0xFF505050, "border_radius": 8}
    _RECORDING = {"background_color": 0xFF3030D0, "border_radius": 8}
    _SAVING = {"background_color": 0xFF2070C0, "border_radius": 8}
    def __init__(self):
        self.window = ui.Window("Recording Status", width=430, height=125)
        with self.window.frame:
            with ui.ZStack(style={"margin": 8}):
                self._background = ui.Rectangle(style=self._READY)
                with ui.VStack(spacing=2):
                    self._state = ui.Label(
                        "READY — PRESS S",
                        height=54,
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 26, "color": 0xFFFFFFFF},
                    )
                    self._detail = ui.Label(
                        "0 demos saved",
                        height=34,
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 17, "color": 0xFFFFFFFF},
                    )

    def update(self, *, recording: bool, saved: int, frames: int, pending: int) -> None:
        if recording:
            self._background.style = self._RECORDING
            self._state.text = "● RECORDING"
            self._detail.text = f"{saved} saved  ·  {frames} frames captured  ·  N to save"
        elif pending:
            self._background.style = self._SAVING
            self._state.text = "SAVING…"
            self._detail.text = f"{saved} saved  ·  {pending} writing  ·  wait for READY"
        else:
            self._background.style = self._READY
            self._state.text = "READY — PRESS S"
            self._detail.text = f"{saved} demos saved"

    def close(self) -> None:
        self.window.visible = False
