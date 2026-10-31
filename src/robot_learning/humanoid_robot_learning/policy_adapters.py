"""One closed-loop interface over trained policies, for simulator rollouts, corrections and audits.

An adapter takes raw numpy observations (``state`` [D] float32, ``images`` {camera: HxWx3 uint8})
and returns absolute actions in the dataset's units; normalization and action queues are internal.

    policy = load_policy(path, device="cuda", n_action_steps=25)
    policy.reset(seed)                               # new episode: clear queues/history, reseed noise
    action = policy.select_action(state, images)     # [D], replans when its queue is empty
    chunk = policy.predict_chunk(state, images)      # [chunk_size, D], no queue side effects

``load_policy`` recognises a LeRobot ``pretrained_model`` directory (ACT).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def to_tensor_images(images):
    return {name: torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255 for name, rgb in images.items()}


class ACTAdapter:
    """LeRobot ACT with its saved pre/post-processors; chunk execution or temporal ensembling."""

    kind = "act"

    def __init__(self, checkpoint, device="cpu", n_action_steps=None, temporal_ensemble_coeff=None, no_temporal_ensemble=False):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(str(checkpoint))
        if config.type != "act":
            raise ValueError(f"{checkpoint} is a {config.type} checkpoint, not ACT")
        config.device = device
        config.pretrained_backbone_weights = None  # trained backbone is in the checkpoint; skip the ImageNet download
        if no_temporal_ensemble:
            config.temporal_ensemble_coeff = None
        if temporal_ensemble_coeff is not None:
            config.temporal_ensemble_coeff, config.n_action_steps = temporal_ensemble_coeff, 1
        if n_action_steps is not None:
            if not 1 <= n_action_steps <= config.chunk_size or (config.temporal_ensemble_coeff is not None and n_action_steps != 1):
                raise ValueError("n_action_steps must be in [1, chunk_size], and 1 with temporal ensembling")
            config.n_action_steps = n_action_steps
        self.config, self.device = config, device
        self.policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True).to(device).eval()
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
                                                       preprocessor_overrides={"device_processor": {"device": device}},
                                                       postprocessor_overrides={"device_processor": {"device": "cpu"}})
        self.cameras = {key.split(".")[-1]: tuple(f.shape[1:]) for key, f in config.input_features.items() if key.startswith("observation.images.")}
        self.chunk_size = config.chunk_size
        self.checkpoint = str(Path(checkpoint).resolve())

    def describe(self):
        return {"policy": "act", "checkpoint": self.checkpoint, "chunk_size": self.chunk_size,
                "n_action_steps": self.config.n_action_steps, "temporal_ensemble_coeff": self.config.temporal_ensemble_coeff}

    def set_n_action_steps(self, steps):
        self.policy.config.n_action_steps = steps

    def reset(self, seed=None):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def _observation(self, state, images):
        obs = {"observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32))}
        obs.update({f"observation.images.{k}": v for k, v in to_tensor_images(images).items()})
        return self.pre(obs)

    @torch.inference_mode()
    def select_action(self, state, images):
        return self.post(self.policy.select_action(self._observation(state, images))).cpu().numpy().reshape(-1)

    @torch.inference_mode()
    def predict_chunk(self, state, images, noise_seed=None):
        chunk = self.post(self.policy.predict_action_chunk(self._observation(state, images)))
        return chunk.cpu().numpy().reshape(self.chunk_size, -1)


def load_policy(path, device="cpu", n_action_steps=None, **options):
    path = Path(path)
    if (path / "model.safetensors").is_file():
        return ACTAdapter(path, device, n_action_steps, **options)
    raise ValueError(f"unrecognised policy checkpoint: {path}")
