"""Flow-matching diffusion transformer policy (DiT: Peebles & Xie 2023; flow matching: Lipman et al. 2023; the design of
TRI's Large Behavior Models, arXiv 2507.05331).

A chunk of ``horizon`` actions is generated from Gaussian noise by integrating a velocity field predicted by a transformer
over the chunk's time steps. Each block is conditioned through adaptive layer norm on one vector built from the flow time,
CLIP image features of the last ``n_obs_steps`` frames (class token plus spatial-softmax keypoints over the patch grid),
the joint state and the CLIP text embedding of the task.
"""
from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.v2.functional as TF  # noqa: N812
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPVisionModel

from .configuration_dit import DiTConfig

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    """Transformer block whose norms are shifted, scaled and gated by the conditioning (adaLN-Zero)."""

    def __init__(self, dim: int, cond_dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        self.modulation = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, c: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(F.silu(c)).unsqueeze(1).chunk(6, dim=-1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1, cos, sin)
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


class ImageEncoder(nn.Module):
    """CLIP ViT: the class token plus ``keypoints`` expected (x, y) positions, each a softmax map over the patch grid."""

    def __init__(self, name: str, keypoints: int):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(name)
        width = self.clip.config.hidden_size
        self.keypoints = nn.Linear(width, keypoints) if keypoints else None
        side = self.clip.config.image_size // self.clip.config.patch_size
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing="ij")
        self.register_buffer("grid", torch.stack([xs, ys], dim=-1).reshape(-1, 2), persistent=False)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)
        self.dim = width + 2 * keypoints

    def forward(self, images: Tensor) -> Tensor:
        out = self.clip(pixel_values=(images - self.mean) / self.std)
        features = out.last_hidden_state[:, 0]
        if self.keypoints is not None:
            weights = self.keypoints(out.last_hidden_state[:, 1:]).softmax(dim=1)
            points = torch.einsum("npk,pc->nkc", weights, self.grid.to(weights.dtype))
            features = torch.cat([features, points.flatten(1)], dim=-1)
        return features


class DiTPolicy(PreTrainedPolicy):
    config_class = DiTConfig
    name = "humanoid_dit"

    def __init__(self, config: DiTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        dim = config.hidden_dim
        self.cameras = list(config.image_features)
        self.action_dim = config.action_feature.shape[0]
        state_dim = config.robot_state_feature.shape[0]

        self.image_encoder = ImageEncoder(config.clip_name, config.keypoints)
        self.text_encoder = CLIPTextModel.from_pretrained(config.clip_name).requires_grad_(False).eval()
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, dim)
        self.time_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        cond_dim = config.n_obs_steps * (len(self.cameras) * self.image_encoder.dim + state_dim) + 2 * dim

        self.action_in = nn.Linear(self.action_dim, dim)
        self.blocks = nn.ModuleList(Block(dim, cond_dim, config.num_heads, config.dropout) for _ in range(config.num_layers))
        self.action_out = nn.Linear(dim, self.action_dim)

        head = dim // config.num_heads
        freqs = 1.0 / 10000 ** (torch.arange(0, head // 2) / (head // 2))
        angles = torch.arange(config.horizon)[:, None] * freqs[None]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        self.reset()

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def get_optim_params(self) -> list[dict]:
        clip = [p for p in self.image_encoder.clip.parameters() if p.requires_grad]
        clip_ids = {id(p) for p in clip}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in clip_ids]
        return [{"params": rest}, {"params": clip, "lr": self.config.optimizer_lr * self.config.vision_lr_scale}]

    def reset(self):
        self._queues = {OBS_STATE: deque(maxlen=self.config.n_obs_steps), OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
                        ACTION: deque(maxlen=self.config.n_action_steps)}

    def _images(self, images: Tensor) -> Tensor:
        """(N, 3, H, W) in [0, 1] -> (N, 3, S, S): resize, then a random crop when training and a centre crop otherwise."""
        size, resize = self.config.image_size, self.config.image_resize
        images = TF.resize(images, [resize, resize], antialias=True)
        if self.training:
            top, left = torch.randint(0, resize - size + 1, (2,)).tolist()
            return images[..., top:top + size, left:left + size]
        return TF.center_crop(images, [size, size])

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        """Observation and task -> conditioning vector. ``batch[OBS_IMAGES]``: (B, n_obs_steps, cameras, 3, H, W)."""
        images = batch[OBS_IMAGES]
        b, steps, cams = images.shape[:3]
        features = self.image_encoder(self._images(images.flatten(0, 2))).view(b, steps, cams * self.image_encoder.dim)
        obs = torch.cat([features, batch[OBS_STATE]], dim=-1).flatten(1)
        with torch.no_grad():
            text = self.text_encoder(input_ids=batch[OBS_LANGUAGE_TOKENS],
                                     attention_mask=batch[OBS_LANGUAGE_ATTENTION_MASK]).pooler_output
        return torch.cat([obs, self.text_proj(text)], dim=-1)

    def velocity(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        angles = t[:, None] * freqs[None]
        c = torch.cat([cond, self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))], dim=-1)
        h = self.action_in(x)
        for block in self.blocks:
            h = block(h, c, self.cos, self.sin)
        return self.action_out(h)

    @torch.no_grad()
    def sample(self, cond: Tensor, steps: int | None = None, generator: torch.Generator | None = None) -> Tensor:
        """Integrate the velocity field from noise (t = 0) to actions (t = 1) with Euler steps: (B, horizon, action_dim)."""
        steps = steps or self.config.flow_steps
        x = torch.randn(cond.shape[0], self.config.horizon, self.action_dim, device=cond.device, generator=generator)
        for i in range(steps):
            t = torch.full((cond.shape[0],), i / steps, device=cond.device)
            x = x + self.velocity(x, t, cond) / steps
        return x

    def _stack_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in self.cameras], dim=-4)
        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        batch = self._stack_images(batch)
        for key in (OBS_STATE, OBS_IMAGES):
            queue = self._queues[key]
            queue.append(batch[key])
            while len(queue) < queue.maxlen:
                queue.append(batch[key])
        history = {key: torch.stack(list(self._queues[key]), dim=1) for key in (OBS_STATE, OBS_IMAGES)}
        history[OBS_LANGUAGE_TOKENS], history[OBS_LANGUAGE_ATTENTION_MASK] = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        start = self.config.n_obs_steps - 1
        return self.sample(self.encode(history))[:, start:start + self.config.n_action_steps]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if not self._queues[ACTION]:
            self._queues[ACTION].extend(self.predict_action_chunk(batch).transpose(0, 1))
        else:
            batch = self._stack_images(batch)
            for key in (OBS_STATE, OBS_IMAGES):
                self._queues[key].append(batch[key])
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        cond = self.encode(self._stack_images(batch))
        actions = batch[ACTION]
        noise = torch.randn_like(actions)
        u = torch.distributions.Beta(self.config.time_beta_alpha, self.config.time_beta_beta).sample((actions.shape[0],))
        t = (self.config.time_max * (1 - u)).to(actions.device)
        x = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        loss = F.mse_loss(self.velocity(x, t, cond), actions - noise)
        return loss, {"loss": loss.item()}
