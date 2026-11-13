from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("pioneer1")
@dataclass
class Pioneer1Config(PreTrainedConfig):
    """Flow-matching transformer over action chunks that cross-attends to language-modulated visual tokens."""

    n_obs_steps: int = 1
    horizon: int = 32
    n_action_steps: int = 32

    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 3072
    dropout: float = 0.1
    context_dropout: float = 0.1
    film_dropout: float = 0.2

    # cameras whose whole patch grid goes into the context; the others give only their class (and register) tokens
    full_token_cameras: tuple[str, ...] = ("observation.images.ego",)
    # DINOv3 is gated on Hugging Face: request access on its model page first
    vision_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    text_name: str = "openai/clip-vit-base-patch16"  # frozen; CLIP gives its sentence embedding, other encoders a mean
    # the instruction's word tokens also go into the context (projected, with a type embedding), and fusion_layers
    # self-attention layers mix the whole context first so image patches can attend to the words
    text_tokens: bool = True
    fusion_layers: int = 2
    # recompute the vision encoder and fusion layers in the backward pass: the same result, less memory
    grad_checkpoint: bool = False
    full_image_size: int = 224
    image_size: int = 224
    # the other cameras also give their patch grid average-pooled to pooled_patch_grid x pooled_patch_grid tokens (0: none)
    pooled_patch_grid: int = 7
    # exponential moving average of the trained weights, used in eval mode (0: off); decay warms up as (1 + n) / (10 + n)
    ema_decay: float = 0.999
    vision_lr_scale: float = 0.1
    max_text_tokens: int = 77
    use_state: bool = True

    # training-time real-time chunking: the first 0..rtc_max_prefix actions are given clean and left out of the loss
    rtc_max_prefix: int = 6

    augment: bool = True
    crop_scale: tuple[float, float] = (0.9, 1.0)
    rotate_deg: float = 5.0
    color_jitter: tuple[float, float, float, float] = (0.3, 0.4, 0.5, 0.08)

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

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_weight_decay: float = 1e-6
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 5000

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError(f"n_action_steps ({self.n_action_steps}) must be <= horizon - n_obs_steps + 1")
        if self.hidden_dim % self.num_heads or (self.hidden_dim // self.num_heads) % 2:
            raise ValueError("hidden_dim / num_heads must be an even integer (rotary embeddings)")
        if not 0 <= self.rtc_max_prefix < self.horizon:
            raise ValueError(f"rtc_max_prefix ({self.rtc_max_prefix}) must be in [0, horizon)")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, betas=self.optimizer_betas, weight_decay=self.optimizer_weight_decay,
                           grad_clip_norm=self.optimizer_grad_clip_norm)

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(name="cosine", num_warmup_steps=self.scheduler_warmup_steps)

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("pioneer1 needs at least one camera")
        missing = [c for c in self.full_token_cameras if c not in self.image_features]
        if missing:
            raise ValueError(f"full_token_cameras {missing} are not cameras of the dataset: {list(self.image_features)}")
        if self.use_state and self.robot_state_feature is None:
            raise ValueError("use_state needs observation.state")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
