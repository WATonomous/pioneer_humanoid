from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("humanoid_dit")
@dataclass
class DiTConfig(PreTrainedConfig):
    """Flow-matching diffusion transformer over action chunks, conditioned on CLIP image and text features and the state."""

    n_obs_steps: int = 2
    horizon: int = 33
    n_action_steps: int = 32

    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    clip_name: str = "openai/clip-vit-base-patch16"
    image_size: int = 224
    image_resize: int = 236
    keypoints: int = 32
    vision_lr_scale: float = 0.1
    max_text_tokens: int = 77

    flow_steps: int = 10
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0
    time_max: float = 0.999

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_weight_decay: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError(f"n_action_steps ({self.n_action_steps}) must be <= horizon - n_obs_steps + 1")
        if self.hidden_dim % self.num_heads or (self.hidden_dim // self.num_heads) % 2:
            raise ValueError("hidden_dim / num_heads must be an even integer (rotary embeddings)")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(lr=self.optimizer_lr, betas=self.optimizer_betas, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(name="cosine", num_warmup_steps=0)

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("humanoid_dit needs at least one camera")
        if self.robot_state_feature is None:
            raise ValueError("humanoid_dit needs observation.state")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
