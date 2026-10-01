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
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.task_description = task_description
        self.robot_type = robot_type
        self.execution_horizon = execution_horizon
        self.device = device or get_safe_torch_device(policy.config.device)
        self.queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=execution_horizon))

    def reset(self) -> None:
        self.policy.reset()
        self.queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=self.execution_horizon))

    def _replan(self, obs_frame: dict) -> None:
        prev_left_over = self.queue.get_left_over()  # None on first call of an episode

        obs = dict(obs_frame)
        obs = prepare_observation_for_inference(
            obs, self.device, task=self.task_description, robot_type=self.robot_type
        )
        obs = self.preprocessor(obs)

        # See module docstring point 1: no torch.inference_mode() anywhere in
        # this call chain.
        action_index_before = self.queue.get_action_index()
        raw_chunk = self.policy.predict_action_chunk(
            obs,
            inference_delay=0,
            prev_chunk_left_over=prev_left_over,
            execution_horizon=self.execution_horizon,
        )  # (1, chunk_size, action_dim)

        with torch.no_grad():
            processed = [self.postprocessor(raw_chunk[:, t]) for t in range(raw_chunk.shape[1])]
        processed_chunk = torch.stack(processed, dim=0)  # (chunk_size, 1, action_dim) or (chunk_size, action_dim)
        if processed_chunk.dim() == 3:
            processed_chunk = processed_chunk.squeeze(1)
        raw_chunk_sq = raw_chunk.squeeze(0)  # (chunk_size, action_dim)

        self.queue.merge(
            original_actions=raw_chunk_sq,
            processed_actions=processed_chunk,
            real_delay=0,
            action_index_before_inference=action_index_before,
        )

    def get_action(self, obs_frame: dict) -> torch.Tensor:
        """Returns the next postprocessed action tensor for ``obs_frame``.

        ``obs_frame`` must already be in LeRobot-schema form (the same shape
        of dict you'd hand to lerobot's own ``predict_action``) -- building
        that from raw sim/robot state, and converting the returned action
        back into sim/robot commands, are both the caller's responsibility.
        """
        if self.queue.queue is None or self.queue.get_action_index() >= self.execution_horizon:
            self._replan(obs_frame)

        action_values = self.queue.get()
        if action_values.dim() == 1:
            action_values = action_values.unsqueeze(0)
        return action_values
