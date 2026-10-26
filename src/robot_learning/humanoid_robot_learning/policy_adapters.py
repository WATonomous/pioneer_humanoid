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


class FlowAdapter:
    """``humanoid_il.flow_transformer`` checkpoint (EMA weights); executes ``n_action_steps`` per chunk."""

    kind = "flow"
    NOISE_SEED_OFFSET = 50_000_000  # per-episode noise seed = layout seed + offset (matches the saved evaluations)

    def __init__(self, checkpoint, device="cpu", n_action_steps=None, refinement_steps=None):
        from humanoid_il.flow_transformer.policy import FlowConfig, FlowPolicy

        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = FlowConfig(**saved["config"])
        self.model = FlowPolicy(self.config).to(device).eval()
        self.model.load_state_dict(saved["ema"])
        self.norm = {k: (torch.tensor(v["mean"], device=device), torch.tensor(v["std"], device=device)) for k, v in saved["normalization"].items()}
        self.device, self.step, self.checkpoint = device, saved["step"], str(Path(checkpoint).resolve())
        self.chunk_size = self.config.chunk_size
        self.n_action_steps = n_action_steps or self.chunk_size
        self.refinement_steps = refinement_steps or self.config.refinement_steps
        self.cameras = {"ego": (self.config.height, self.config.width), "wrist_left": (self.config.height, self.config.width)}
        self.queue, self.generator = [], None

    def describe(self):
        return {"policy": "flow_transformer", "checkpoint": self.checkpoint, "step": self.step, "chunk_size": self.chunk_size,
                "n_action_steps": self.n_action_steps, "refinement_steps": self.refinement_steps, "ema": True}

    def set_n_action_steps(self, steps):
        self.n_action_steps = steps

    def reset(self, seed=None):
        self.queue = []
        self.generator = torch.Generator(device=self.device).manual_seed((seed or 0) + self.NOISE_SEED_OFFSET)

    def _inputs(self, state, images):
        frames = to_tensor_images(images)
        pixels = torch.stack([frames["ego"], frames["wrist_left"]])[None].to(self.device)
        mean, std = self.norm["observation.state"]
        return pixels, (torch.from_numpy(np.asarray(state, dtype=np.float32))[None].to(self.device) - mean) / std

    @torch.inference_mode()
    def predict_chunk(self, state, images, noise_seed=None, generator=None):
        if noise_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(noise_seed)
        actions = self.model.sample(*self._inputs(state, images), generator=generator or self.generator, steps=self.refinement_steps)[0]
        mean, std = self.norm["action"]
        return (actions * std + mean).cpu().numpy()

    def select_action(self, state, images):
        if not self.queue:
            self.queue = list(self.predict_chunk(state, images)[:self.n_action_steps])
        return self.queue.pop(0)


def load_policy(path, device="cpu", n_action_steps=None, **options):
    """ACT ``pretrained_model`` directory or flow ``<step>.pt`` checkpoint."""
    path = Path(path)
    if (path / "model.safetensors").is_file():
        return ACTAdapter(path, device, n_action_steps, **options)
    if path.suffix == ".pt":
        return FlowAdapter(path, device, n_action_steps, **options)
    raise ValueError(f"unrecognised policy checkpoint: {path}")
