"""Conditional rectified-flow action transformer (no U-Net, no VLM, no ACT weights).

Two cameras pass through one ImageNet ResNet-18 (trainable, frozen batch norm) into spatial tokens
(15x20 per 480x640 image) with 2-D sinusoidal positions and a camera embedding. A state token and
a flow-time token join them as cross-attention memory for a noncausal transformer decoder over
the action chunk. Training follows linear conditional flow matching
(x_t = (1 - t) * noise + t * actions, target velocity actions - noise; Lipman et al. 2023,
https://arxiv.org/abs/2210.02747, as used by pi0, https://www.physicalintelligence.company/download/pi0.pdf).
Sampling integrates the velocity field with forward Euler from Gaussian noise, encoding the
images once per chunk.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class FlowConfig:
    chunk_size: int = 50
    action_dim: int = 7
    state_dim: int = 7
    hidden_dim: int = 512
    layers: int = 6
    heads: int = 8
    dropout: float = 0.1
    refinement_steps: int = 10
    height: int = 480
    width: int = 640

    def dictionary(self):
        return asdict(self)


def sinusoid(values, dim):
    frequency = torch.exp(torch.arange(dim // 2, device=values.device, dtype=torch.float32) * (-math.log(10000) / (dim // 2 - 1)))
    phase = values.float()[..., None] * frequency
    return torch.cat([phase.sin(), phase.cos()], -1)


def spatial_positions(h, w, dim, device):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.cat([sinusoid(y, dim // 2), sinusoid(x, dim // 2)], -1).reshape(h * w, dim)


class FlowPolicy(nn.Module):
    def __init__(self, config=None, pretrained_vision=False):
        super().__init__()
        self.config = c = config or FlowConfig()
        d = c.hidden_dim
        if d % 4 or d % c.heads:
            raise ValueError("hidden dimension must be divisible by 4 and by the head count")
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_vision else None, norm_layer=FrozenBatchNorm2d)
        self.vision = nn.Sequential(*list(backbone.children())[:-2])
        self.vision_projection = nn.Conv2d(512, d, 1)
        self.camera_embedding = nn.Parameter(torch.randn(2, d) * 0.02)
        self.state_encoder = nn.Sequential(nn.Linear(c.state_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.time_encoder = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.action_embedding = nn.Linear(c.action_dim, d)
        self.action_positions = nn.Parameter(torch.randn(c.chunk_size, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, c.heads, 4 * d, c.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, c.layers, norm=nn.LayerNorm(d))
        self.output = nn.Linear(d, c.action_dim)
        nn.init.zeros_(self.output.weight)  # start from zero velocity
        nn.init.zeros_(self.output.bias)
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1))

    def encode_condition(self, images, state):
        """images [B, 2, 3, H, W] in [0, 1]; state [B, state_dim] normalized -> memory tokens."""
        b, n, ch, h, w = images.shape
        if n != 2 or ch != 3:
            raise ValueError("expected ego and wrist RGB images")
        features = self.vision_projection(self.vision((images.reshape(b * n, ch, h, w) - self.image_mean) / self.image_std))
        fh, fw = features.shape[-2:]
        d = self.config.hidden_dim
        features = features.flatten(2).transpose(1, 2).reshape(b, n, fh * fw, d)
        features = features + spatial_positions(fh, fw, d, images.device)[None, None] + self.camera_embedding[None, :, None]
        return torch.cat([features.reshape(b, n * fh * fw, d), self.state_encoder(state)[:, None]], dim=1)

    def velocity(self, actions, tau, condition):
        time = self.time_encoder(sinusoid(tau * 1000, self.config.hidden_dim))[:, None]
        tokens = self.action_embedding(actions) + self.action_positions[None] + time
        return self.output(self.transformer(tokens, torch.cat([condition, time], dim=1)))

    def forward(self, images, state, actions, tau):
        return self.velocity(actions, tau, self.encode_condition(images, state))

    @torch.no_grad()
    def sample(self, images, state, generator=None, noise=None, steps=None):
        """Normalized action chunk [B, chunk, action_dim] by Euler integration from noise."""
        condition = self.encode_condition(images, state)
        steps = steps or self.config.refinement_steps
        shape = (len(state), self.config.chunk_size, self.config.action_dim)
        actions = torch.randn(shape, device=state.device, generator=generator) if noise is None else noise.clone()
        for i in range(steps):
            tau = torch.full((len(state),), i / steps, device=state.device)
            actions = actions + self.velocity(actions, tau, condition) / steps
        return actions


def flow_loss(model, images, state, actions, padding, tau=None, noise=None, metric_mask=None):
    """Masked velocity MSE; padded chunk steps (past the episode end) contribute nothing."""
    tau = torch.rand(len(actions), device=actions.device) if tau is None else tau
    noise = torch.randn_like(actions) if noise is None else noise
    noisy = (1 - tau[:, None, None]) * noise + tau[:, None, None] * actions
    error = (model(images, state, noisy, tau) - (actions - noise)).square()
    valid = ~padding.bool()
    loss = (error * valid[:, :, None]).sum() / (valid.sum().clamp_min(1) * actions.shape[-1])
    metrics = {"velocity_mse": float(loss.detach())}
    if metric_mask is not None:
        selected = valid & metric_mask[:, None].bool()
        metrics["metric_sum"] = float((error * selected[:, :, None]).sum().detach())
        metrics["metric_count"] = int(selected.sum()) * actions.shape[-1]
    return loss, metrics
