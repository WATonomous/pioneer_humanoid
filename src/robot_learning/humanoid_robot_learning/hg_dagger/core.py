# Manage data formats and switches between human and policy control.
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Protocol

import numpy as np


# Keep state, action, camera and timing formats consistent across the pipeline.
@dataclass(frozen=True)
class Contract:

    scene: str
    robot: str
    fps: int
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    state_units: tuple[str, ...]
    action_units: tuple[str, ...]
    images: dict[str, tuple[int, int, int]]
    action_semantics: str = "absolute_joint_position"

    # Check names, units, frame rate and HWC RGB image shapes.
    def __post_init__(self):
        if self.fps <= 0 or not self.state_names or not self.action_names:
            raise ValueError("Positive fps and nonempty state/action names are required")
        for names, units in ((self.state_names, self.state_units), (self.action_names, self.action_units)):
            if len(names) != len(units) or len(set(names)) != len(names):
                raise ValueError("Names must be unique and have one unit each")
        for name, shape in self.images.items():
            if not name.startswith("observation.images.") or len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
                raise ValueError("Images require observation.images.<name> keys and positive HWC RGB shapes")

    # Serialize the data contract for recordings and checkpoints.
    def json(self):
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    # Restore a saved data contract and its tuple fields.
    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        for key in ("state_names", "action_names", "state_units", "action_units"):
            data[key] = tuple(data[key])
        data["images"] = {k: tuple(v) for k, v in data["images"].items()}
        return cls(**data)

    # Check that state vectors and uint8 RGB images match the contract.
    def check_observation(self, obs):
        if set(obs) != {"observation.state", *self.images}:
            raise ValueError("Observation keys do not match the contract")
        self.vector(obs["observation.state"], len(self.state_names), "state")
        for key, shape in self.images.items():
            a = np.asarray(obs[key])
            if a.shape != shape or a.dtype != np.uint8:
                raise ValueError(f"{key}: expected RGB uint8 {shape}, got {a.dtype} {a.shape}")

    # Return a finite float32 vector with the expected size.
    @staticmethod
    def vector(value, size, name):
        a = np.asarray(value, dtype=np.float32)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(f"{name}: expected finite vector of size {size}, got {a.shape}")
        return a.copy()

    # Check and copy an action before it reaches the scene.
    def action(self, value):
        return self.vector(value, len(self.action_names), "action")


# Describe readiness, human takeover and explicit collection decisions.
@dataclass(frozen=True)
class Signals:
    ready: bool = False
    takeover: bool = False
    accept: bool = False
    discard: bool = False
    quit: bool = False


# Define what the collector needs from any scene or simulator.
class Scene(Protocol):
    contract: Contract

    # Reset the scene using a reproducible seed.
    def reset(self, seed: int) -> None: ...
    # Read the current state and camera images.
    def observe(self) -> dict: ...
    # Advance one tick and return the action actually sent after limits.
    def execute(self, action: np.ndarray) -> np.ndarray:
        ...
    # Check task success or raise if the scene has no success rule.
    def success(self) -> bool:
        ...
    # Release the scene resources.
    def close(self) -> None: ...


# Define how a human controller supplies commands and decisions.
class Teleop(Protocol):
    # Read readiness, takeover and save or discard decisions.
    def poll(self) -> Signals: ...
    # Reanchor human control to the current scene state.
    def seed(self) -> None: ...
    # Get the next command from the human controller.
    def action(self) -> np.ndarray: ...
    # Release the controller resources.
    def close(self) -> None: ...


# Define how a learned policy supplies and resets actions.
class Policy(Protocol):
    # Clear any queued actions or policy history.
    def reset(self) -> None: ...
    # Predict the next action from observations and the task instruction.
    def action(self, observation: dict, task: str) -> np.ndarray: ...


# Record human interventions and save them only when explicitly accepted.
class Session:

    # Prepare the scene, controllers and correction buffer for a session.
    def __init__(self, scene, teleop, policy, store, task, seed=0, max_frames=1500):
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        self.scene, self.teleop, self.policy, self.store = scene, teleop, policy, store
        self.task, self.seed, self.max_frames = task, seed, max_frames
        self.mode = "POLICY"
        self.started = False
        self.wait_release = True
        self.segments, self.frames = [], []
        self.accepted = 0
        self.scene.reset(seed)
        self.policy.reset()
        self.teleop.seed()

    # Count unsaved frames across all pending interventions.
    @property
    def pending_frames(self):
        return len(self.frames) + sum(map(len, self.segments))

    # Keep the current intervention separate when human control ends.
    def stage(self):
        if self.frames:
            self.segments.append(self.frames)
            self.frames = []

    # Clear pending corrections and prepare the next episode.
    def reset(self):
        self.segments, self.frames = [], []
        self.mode = "POLICY"
        self.seed += 1
        self.scene.reset(self.seed)
        self.policy.reset()
        self.teleop.seed()
        self.wait_release = True

    # Handle control decisions, execute one action and record human labels.
    def tick(self):
        signal = self.teleop.poll()
        if signal.quit:
            # Leave pending corrections unsaved when quitting.
            return False
        if not self.started:
            if signal.ready and not signal.takeover:
                self.started = True
                self.wait_release = False
                self.teleop.seed()
            # Hold physics and recording until the controller is ready.
            return True
        if signal.discard:
            # Discard wins if accept and discard arrive together.
            self.reset()
            return True
        if signal.accept:
            self.stage()
            if self.segments:
                self.store.save(self.segments, {"task": self.task, "seed": self.seed})
                self.accepted += len(self.segments)
                print(f"[HG-DAgger] Saved {len(self.segments)} segment(s), {self.pending_frames} frames", flush=True)
            else:
                print("[HG-DAgger] No human frames to save; resetting", flush=True)
            self.reset()
            return True
        if self.wait_release:
            if signal.ready and not signal.takeover:
                self.wait_release = False
                self.teleop.seed()
            return True
        if not signal.ready:
            # End the segment and clear queued actions when controller input is lost.
            self.stage()
            self.policy.reset()
            self.teleop.seed()
            self.wait_release = True
            return True
        mode = "HUMAN" if signal.takeover else "POLICY"
        if mode != self.mode:
            self.stage()
            self.policy.reset()
            self.teleop.seed()
            self.mode = mode
        if self.pending_frames >= self.max_frames:
            # Hold until the operator accepts or discards the full buffer.
            return True
        obs = self.scene.observe()
        self.scene.contract.check_observation(obs)
        # Copy observations before stepping because simulators may reuse these arrays.
        frame = {k: np.asarray(v).copy() for k, v in obs.items()} if self.mode == "HUMAN" else None
        action = (self.teleop.action() if self.mode == "HUMAN"
                  else self.policy.action(obs, self.task))
        executed = self.scene.contract.action(self.scene.execute(self.scene.contract.action(action)))
        if frame is not None:
            frame["action"] = executed
            self.frames.append(frame)
        return True
