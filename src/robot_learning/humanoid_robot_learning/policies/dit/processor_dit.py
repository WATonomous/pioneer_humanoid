from __future__ import annotations

from typing import Any

import torch
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_dit import DiTConfig


def make_dit_pre_post_processors(config: DiTConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None):
    """Batch, tokenize the task with the CLIP tokenizer, move to the device and normalize; unnormalize actions on the way out."""
    features = {**config.input_features, **config.output_features}
    pre = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            TokenizerProcessorStep(tokenizer_name=config.clip_name, max_length=config.max_text_tokens, padding="max_length",
                                   padding_side="right", truncation=True),
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(features=features, norm_map=config.normalization_mapping, stats=dataset_stats),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    post = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[
            UnnormalizerProcessorStep(features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats),
            DeviceProcessorStep(device="cpu"),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return pre, post
