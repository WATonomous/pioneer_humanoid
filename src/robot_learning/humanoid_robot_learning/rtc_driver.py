"""Generic Real-Time Chunking (RTC) driver for flow-matching lerobot policies
(pi0 / pi0.5 / SmolVLA).

lerobot's own ``PreTrainedPolicy.select_action()`` refuses to run when RTC is
enabled on the policy config (it asserts you must drive the policy via
``predict_action_chunk()`` instead) -- RTC needs to see the unconsumed tail of
the previous chunk and blend it into the new one via prefix-guided denoising,
which the simple "queue empties -> pull a whole new chunk" flow used by
``select_action`` can't express.

This module owns that driving logic only. It is embodiment-agnostic: it knows
nothing about SO101, pioneer, cameras, or joint layouts. Callers hand it a
LeRobot-schema ``obs_frame`` dict (however their own robot/sim interface
builds one) and get back a raw, postprocessed action tensor in the policy's
own action space -- converting that to real sim/robot commands is the
caller's job, same as it always was with ``predict_action``.

Two bugs worth knowing about if you extend this:

1. RTC's guidance term is computed via a real backward pass
   (``torch.enable_grad()`` + ``torch.autograd.grad()`` inside
   ``RTCProcessor.denoise_step``). Never wrap the call chain into
   ``predict_action_chunk`` in ``torch.inference_mode()`` -- unlike
   ``torch.no_grad()``, inference-mode tensors can never be used in autograd
   again, even under a nested ``enable_grad()``. Use ``torch.no_grad()`` in
   the caller's rollout loop instead.

2. ``max_guidance_weight`` (how strongly RTC pulls the new chunk toward
   consistency with the old one) is baked into the policy at *load* time via
   ``policy.config.rtc_config`` -- it is not a per-call parameter here. Tune
   it low (we found ~1.0 clearly better than lerobot's ACTConfig default of
   10.0 on our SO101 task); too-strong guidance over-constrains the new chunk
   toward a trajectory that may already be failing.
"""

from __future__ import annotations

import math
import time
import threading
from collections import deque
from typing import Any

import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.control_utils import prepare_observation_for_inference
from lerobot.utils.utils import get_safe_torch_device


class RTCDrivenPolicy:
    """Drives a flow-matching policy with proper RTC prefix-guided chunk replanning.

    Args:
        policy: a loaded flow-matching ``PreTrainedPolicy`` (pi0/pi0.5/SmolVLA)
            whose ``config.rtc_config`` already has ``enabled=True`` (set this
            at policy-load time, e.g. via ``PreTrainedConfig.from_pretrained``
            + overriding ``policy_config.rtc_config`` before ``make_policy``).
        preprocessor / postprocessor: the policy's own
            ``PolicyProcessorPipeline`` pair, as returned by
            ``make_pre_post_processors``.
        task_description: language instruction string for the task.
        robot_type: robot-type string lerobot expects in the obs frame
            (``prepare_observation_for_inference``'s ``robot_type`` arg).
        execution_horizon: how many actions to consume from a chunk before
            triggering a fresh, prefix-guided replan. Also controls how much
            of the previous chunk's tail RTC treats as "in flight" guidance
            context.
        control_hz: control-loop rate, used only to convert measured
            inference latency (wall-clock seconds) into a tick count for
            ``real_delay``. Must match the caller's actual tick rate, not
            necessarily the training dataset's fps.
        initial_action: safe fallback action tensor, returned by
            ``get_action`` on the rare tick where the queue is empty but a
            replan is already in flight (so a fresh action can't be served
            yet either). Should be whatever safe/home pose the caller
            already uses to warm-start an episode -- this class has no
            embodiment knowledge of its own to invent one. After the first
            real action is produced, this is replaced by that action (held,
            not reused) for any later gap; ``reset()`` restores it.
        device: inference device; defaults to ``policy.config.device``.
    """

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        task_description: str,
        robot_type: str,
        execution_horizon: int,
        control_hz: float,
        initial_action: torch.Tensor,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.task_description = task_description
        self.robot_type = robot_type
        self.execution_horizon = execution_horizon
        self.tick_length_s = 1.0 / control_hz
        self.device = device or get_safe_torch_device(policy.config.device)
        self.queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=execution_horizon))
        # Rolling window of the last 10 measured delays (in ticks); `self.d`
        # (the estimate fed to merge()) is the max of this window, biased
        # toward overestimating since under-triggering is the worse failure
        # mode (queue runs dry mid-inference) than over-triggering (mildly
        # stale guidance target). Seeded with 1 tick so the very first
        # _replan call -- before any real measurement exists -- isn't 0.
        self.d_arr = deque([1], maxlen=10)
        self.d = 1
        self._replanning = False
        self._episode_id = 0
        self._initial_action = initial_action
        self._last_action = initial_action

    def reset(self) -> None:
        self.policy.reset()
        self.queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=self.execution_horizon))
        self._episode_id += 1
        # Otherwise the new episode's first ticks (queue empty, replan in flight) would
        # replay the previous episode's final action.
        self._last_action = self._initial_action

    def _replan(self, obs_frame: dict, prev_left_over, action_index_before) -> None:
        try:
            obs = dict(obs_frame)
            obs = prepare_observation_for_inference(
                obs, self.device, task=self.task_description, robot_type=self.robot_type
            )
            obs = self.preprocessor(obs)

            start = time.perf_counter()
            raw_chunk = self.policy.predict_action_chunk(
                obs,
                # inference_delay is estimated delay based on highest delay from last 10 calls
                inference_delay=self.d,
                prev_chunk_left_over=prev_left_over,
                execution_horizon=self.execution_horizon,
            )  # (1, chunk_size, action_dim)

            with torch.no_grad():
                processed = [self.postprocessor(raw_chunk[:, t]) for t in range(raw_chunk.shape[1])]
            processed_chunk = torch.stack(processed, dim=0)  # (chunk_size, 1, action_dim) or (chunk_size, action_dim)
            if processed_chunk.dim() == 3:
                processed_chunk = processed_chunk.squeeze(1)
            raw_chunk_sq = raw_chunk.squeeze(0)  # (chunk_size, action_dim)

            end = time.perf_counter()
            elapsed_s = end - start
            # Seconds -> ticks, rounded UP: underestimating real_delay is the bad
            # direction (merge() would keep stale slots it should have dropped),
            # so always err toward a larger tick count, never a smaller one.
            elapsed_ticks = math.ceil(elapsed_s / self.tick_length_s)
            self.d_arr.append(elapsed_ticks)
            self.d = max(self.d_arr)

            if self._temp_episode_id != self._episode_id:
                return

            self.queue.merge(
                original_actions=raw_chunk_sq,
                processed_actions=processed_chunk,
                real_delay=self.d,
                action_index_before_inference=action_index_before,
            )
        finally:
            # always reset _replanning to prevent soft lock in case of exception
            self._replanning = False

    def get_action(self, obs_frame: dict) -> torch.Tensor:
        """Returns the next postprocessed action tensor for ``obs_frame``.

        ``obs_frame`` must already be in LeRobot-schema form (the same shape
        of dict you'd hand to lerobot's own ``predict_action``) -- building
        that from raw sim/robot state, and converting the returned action
        back into sim/robot commands, are both the caller's responsibility.
        """
        if (self.queue.queue is None or self.queue.get_action_index() >= self.execution_horizon - self.d) and not self._replanning:
            self._replanning = True
            self._temp_episode_id = self._episode_id
            threading.Thread(target=self._replan, args=(obs_frame, self.queue.get_left_over(), self.queue.get_action_index())).start()

        action_values = self.queue.get()

        if action_values is None:
            action_values = self._last_action
        elif action_values.dim() == 1:
            action_values = action_values.unsqueeze(0)
        self._last_action = action_values
        return action_values
