"""Flow-matching transformer policy that cross-attends to language-modulated visual tokens.

Context: every patch token of the ``full_token_cameras`` (the scene view) and, for the other cameras (the wrist views),
the class/register tokens plus the patch grid pooled to ``pooled_patch_grid`` squared, from a pretrained ViT (DINOv3 by
default) fine-tuned with the policy, plus a joint-state token. The instruction's frozen text embedding scales and shifts
every visual token (FiLM, Perez et al. 2018), starting at the identity. With ``text_tokens`` the instruction's word tokens
join the context too, and ``fusion_layers`` self-attention layers mix it so each image patch can attend to the words.

Action generator: the noisy chunk is ``horizon`` action tokens. Each block applies self-attention over the chunk, cross-
attention from the action tokens to the context and a feed-forward layer; the flow time drives adaptive layer norm with zero-
initialised gates (adaLN-Zero). Trained with flow matching (Lipman et al. 2023) and real-time chunking built in: the first
0..``rtc_max_prefix`` actions are given clean (flow time 1), so at run time the chunk can continue the actions still being
executed. Eval mode runs on an exponential moving average of the weights (``ema_decay``).
"""
from __future__ import annotations

import math
from collections import deque

import torch
import torch.utils.checkpoint
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.v2 as T  # noqa: N812
import torchvision.transforms.v2.functional as TF  # noqa: N812
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from torch import Tensor, nn
from transformers import AutoModel, CLIPTextModel, CLIPVisionModel

from .configuration_ditx import DiTXConfig

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


def time_embedding(t: Tensor, dim: int = 256) -> Tensor:
    """Sinusoidal embedding of flow times of any shape: (...,) -> (..., dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t[..., None] * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class Attention(nn.Module):
    """Multi-head attention; self-attention with rotary positions when ``context`` is None, cross-attention otherwise."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, context: Tensor | None = None, cos: Tensor | None = None, sin: Tensor | None = None,
                mask: Tensor | None = None) -> Tensor:
        """``mask`` (B, N) is True for the context tokens that may be attended to (False for text padding)."""
        b, t, d = x.shape
        src = x if context is None else context
        q = self.q(x).view(b, t, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(src).view(b, src.shape[1], 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if context is None:
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        # written out, not F.scaled_dot_product_attention: its memory-efficient kernel's backward gives ~1e19 gradients
        # here (RTX 5070 Ti, fp32, head dim 96) while the math path gives ~20; the attention is small (32 queries)
        w = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        if mask is not None:
            w = w.masked_fill(~mask[:, None, None], float("-inf"))
        w = F.dropout(w.softmax(dim=-1), self.dropout, self.training)
        return self.out((w @ v).transpose(1, 2).reshape(b, t, d))


class FusionLayer(nn.Module):
    """Pre-norm transformer layer over the whole context (visual, text and state tokens together), so each image patch
    can look at the instruction's words before the action tokens read it: per-patch grounding that FiLM's single
    scale and shift per channel cannot express."""

    def __init__(self, dim: int, ff_dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(ff_dim, dim))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, context=h, mask=mask)
        return x + self.ff(self.norm2(x))


class Block(nn.Module):
    """Self-attention over the chunk, cross-attention to the context, feed-forward; each shifted, scaled and gated by the
    per-token flow-time embedding (adaLN-Zero)."""

    def __init__(self, dim: int, ff_dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3))
        self.self_attn = Attention(dim, heads, dropout)
        self.cross_attn = Attention(dim, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.modulation = nn.Linear(dim, 9 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, c: Tensor, context: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None = None) -> Tensor:
        """x (B, H, D) action tokens; c (B, H, D) flow-time embedding per token; context (B, N, D)."""
        m = self.modulation(F.silu(c)).chunk(9, dim=-1)
        x = x + m[2] * self.self_attn(self.norm1(x) * (1 + m[1]) + m[0], cos=cos, sin=sin)
        x = x + m[5] * self.cross_attn(self.norm2(x) * (1 + m[4]) + m[3], context=context, mask=mask)
        return x + m[8] * self.mlp(self.norm3(x) * (1 + m[7]) + m[6])


class VisionEncoder(nn.Module):
    """Pretrained ViT (CLIP or DINO from Hugging Face): images -> all tokens (class [+ registers] + patches) and the count of
    leading non-patch tokens."""

    def __init__(self, name: str):
        super().__init__()
        self.is_clip = "clip" in name.lower()
        self.model = CLIPVisionModel.from_pretrained(name) if self.is_clip else AutoModel.from_pretrained(name)
        self.dim = self.model.config.hidden_size
        self.prefix = 1 + getattr(self.model.config, "num_register_tokens", 0)
        mean, std = (CLIP_MEAN, CLIP_STD) if self.is_clip else (IMAGENET_MEAN, IMAGENET_STD)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: Tensor) -> Tensor:
        pixels = (images - self.mean) / self.std
        if self.is_clip:
            native = self.model.config.image_size
            out = self.model(pixel_values=pixels, interpolate_pos_encoding=images.shape[-1] != native)
        else:
            out = self.model(pixel_values=pixels)
        return out.last_hidden_state


class DiTXPolicy(PreTrainedPolicy):
    config_class = DiTXConfig
    name = "humanoid_ditx"

    def __init__(self, config: DiTXConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        dim = config.hidden_dim
        self.cameras = list(config.image_features)
        self.action_dim = config.action_feature.shape[0]

        self.vision = VisionEncoder(config.vision_name)
        if config.grad_checkpoint:
            self.vision.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.vision_proj = nn.Linear(self.vision.dim, dim)
        self.camera_emb = nn.Parameter(torch.randn(len(self.cameras), dim) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(config.n_obs_steps, dim) * 0.02)
        self.context_norm = nn.LayerNorm(dim)
        self.state_proj = nn.Linear(config.robot_state_feature.shape[0], dim) if config.use_state else None

        text_cls = CLIPTextModel if "clip" in config.text_name.lower() else AutoModel
        self.text_encoder = text_cls.from_pretrained(config.text_name).requires_grad_(False).eval()
        text_dim = self.text_encoder.config.hidden_size
        # FiLM: z <- (1 + gamma(e)) * z + beta(e), shared across visual tokens; the last layer starts at zero (identity)
        self.film = nn.Sequential(nn.Dropout(config.film_dropout), nn.Linear(text_dim, dim), nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.context_drop = nn.Dropout(config.context_dropout)
        # instruction word tokens in the context (projected frozen text features + a type embedding), and fusion layers
        self.text_proj = nn.Linear(text_dim, dim) if config.text_tokens else None
        self.text_type = nn.Parameter(torch.randn(dim) * 0.02) if config.text_tokens else None
        self.fusion = nn.ModuleList(FusionLayer(dim, config.ff_dim, config.num_heads, config.dropout)
                                    for _ in range(config.fusion_layers))

        self.time_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_in = nn.Linear(self.action_dim, dim)
        self.blocks = nn.ModuleList(Block(dim, config.ff_dim, config.num_heads, config.dropout) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_mod = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.final_mod.weight)
        nn.init.zeros_(self.final_mod.bias)
        self.action_out = nn.Linear(dim, self.action_dim)

        head = dim // config.num_heads
        freqs = 1.0 / 10000 ** (torch.arange(0, head // 2) / (head // 2))
        angles = torch.arange(config.horizon)[:, None] * freqs[None]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

        jitter = config.color_jitter
        self.color = T.ColorJitter(brightness=jitter[0], contrast=jitter[1], saturation=jitter[2], hue=jitter[3])
        # EMA copies of the trained weights (buffers, so they are saved with the model); filled from the weights at the
        # first update, so a run started from a checkpoint without them averages from the loaded weights
        self._ema_names = [n for n, p in self.named_parameters() if p.requires_grad] if config.ema_decay > 0 else []
        for n in self._ema_names:
            self.register_buffer(self._ema_key(n), torch.zeros_like(self.get_parameter(n)))
        if self._ema_names:
            self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        self._ema_active = False
        self.reset()

    @staticmethod
    def _ema_key(name: str) -> str:
        return "ema__" + name.replace(".", "__")

    @torch.no_grad()
    def update(self):
        """Called by lerobot_train after each optimizer step."""
        if not self._ema_names:
            return
        if self._ema_active:
            raise RuntimeError("EMA update while the EMA weights are swapped in (eval mode)")
        n = int(self.ema_updates)
        decay = min(self.config.ema_decay, (1 + n) / (10 + n)) if n else 0.0
        for name in self._ema_names:
            self.get_buffer(self._ema_key(name)).lerp_(self.get_parameter(name).detach(), 1.0 - decay)
        self.ema_updates += 1

    @torch.no_grad()
    def _swap_ema(self):
        if int(self.ema_updates) == 0:
            raise RuntimeError("ema_decay is set but this model has no EMA weights yet (ema_updates = 0)")
        for name in self._ema_names:
            p, e = self.get_parameter(name), self.get_buffer(self._ema_key(name))
            tmp = p.detach().clone()
            p.copy_(e)
            e.copy_(tmp)
        self._ema_active = not self._ema_active

    def train(self, mode: bool = True):
        super().train(mode)
        # eval mode runs on the EMA weights, train mode on the trained ones
        if self._ema_names and int(self.ema_updates) and (not mode) != self._ema_active:
            self._swap_ema()
        self.text_encoder.eval()
        return self

    def get_optim_params(self) -> list[dict]:
        vision = [p for p in self.vision.parameters() if p.requires_grad]
        vision_ids = {id(p) for p in vision}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in vision_ids]
        return [{"params": rest}, {"params": vision, "lr": self.config.optimizer_lr * self.config.vision_lr_scale}]

    def reset(self):
        self._queues = {OBS_STATE: deque(maxlen=self.config.n_obs_steps), OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
                        ACTION: deque(maxlen=self.config.n_action_steps)}

    def _prepare(self, images: Tensor, size: int) -> Tensor:
        """(N, 3, H, W) in [0, 1] -> (N, 3, size, size). Training: random resized crop, small rotation and colour jitter
        per image; otherwise a plain resize."""
        if not (self.training and self.config.augment):
            return TF.resize(images, [size, size], antialias=True)
        cfg, out = self.config, []
        for img in images:
            top, left, h, w = T.RandomResizedCrop.get_params(img, scale=list(cfg.crop_scale), ratio=[1.0, 1.0])
            img = TF.resized_crop(img, top, left, h, w, [size, size], antialias=True)
            img = TF.rotate(img, float(torch.empty(1).uniform_(-cfg.rotate_deg, cfg.rotate_deg)))
            out.append(self.color(img))
        return torch.stack(out)

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None]:
        """Observation and task -> context tokens (B, N, D) and their key mask (B, N) or None.
        ``batch[OBS_IMAGES]``: (B, n_obs_steps, cameras, 3, H, W)."""
        cfg = self.config
        images = batch[OBS_IMAGES]
        b, steps = images.shape[:2]
        tokens = []
        for i, cam in enumerate(self.cameras):
            full = cam in cfg.full_token_cameras
            x = self._prepare(images[:, :, i].flatten(0, 1), cfg.full_image_size if full else cfg.image_size)
            h = self.vision(x)
            if not full and cfg.pooled_patch_grid:
                grid = h[:, self.vision.prefix:]
                side = math.isqrt(grid.shape[1])
                if side * side != grid.shape[1]:
                    raise ValueError(f"{cam}: {grid.shape[1]} patch tokens are not a square grid")
                grid = grid.transpose(1, 2).reshape(grid.shape[0], -1, side, side)
                pooled = F.adaptive_avg_pool2d(grid, cfg.pooled_patch_grid).flatten(2).transpose(1, 2)
                h = torch.cat([h[:, :self.vision.prefix], pooled], dim=1)
            elif not full:
                h = h[:, :self.vision.prefix]
            h = self.vision_proj(h).view(b, steps, h.shape[1], -1) + self.camera_emb[i] + self.step_emb[:, None]
            tokens.append(h.flatten(1, 2))
        visual = self.context_norm(torch.cat(tokens, dim=1))
        with torch.no_grad():
            mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
            out = self.text_encoder(input_ids=batch[OBS_LANGUAGE_TOKENS], attention_mask=mask)
            if isinstance(self.text_encoder, CLIPTextModel):
                text = out.pooler_output    # CLIP's sentence embedding: the end-of-text token, aligned with its images
            else:                           # mean over the real (unpadded) tokens
                text = (out.last_hidden_state * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        gamma, beta = self.film(text).chunk(2, dim=-1)
        context = (1 + gamma[:, None]) * visual + beta[:, None]
        if self.state_proj is not None:
            state = self.state_proj(batch[OBS_STATE]) + self.step_emb
            context = torch.cat([context, state], dim=1)
        keep = None
        if self.text_proj is not None:
            words = self.text_proj(out.last_hidden_state) + self.text_type
            keep = torch.cat([torch.ones(context.shape[:2], dtype=torch.bool, device=context.device), mask.bool()], dim=1)
            context = torch.cat([context, words], dim=1)
        fuse_mask = keep if keep is not None else torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        for layer in self.fusion:
            if self.config.grad_checkpoint and self.training:
                context = torch.utils.checkpoint.checkpoint(layer, context, fuse_mask, use_reentrant=False)
            else:
                context = layer(context, fuse_mask)
        return self.context_drop(context), keep

    def velocity(self, x: Tensor, t: Tensor, context: Tensor, mask: Tensor | None = None) -> Tensor:
        """x (B, H, A) noisy chunk; t (B, H) flow time of each action token; context (B, N, D) -> velocity (B, H, A)."""
        c = self.time_mlp(time_embedding(t))
        h = self.action_in(x)
        for block in self.blocks:
            h = block(h, c, context, self.cos, self.sin, mask)
        shift, scale = self.final_mod(F.silu(c)).chunk(2, dim=-1)
        return self.action_out(self.final_norm(h) * (1 + scale) + shift)

    @torch.no_grad()
    def sample(self, context: Tensor, prefix: Tensor | None = None, steps: int | None = None,
               generator: torch.Generator | None = None, mask: Tensor | None = None) -> Tensor:
        """Integrate from noise (t = 0) to actions (t = 1) with Euler steps: (B, horizon, action_dim). ``prefix`` (B, d, A),
        normalised actions already committed for the chunk's first d steps, is held clean throughout (real-time chunking)."""
        steps = steps or self.config.flow_steps
        b, horizon = context.shape[0], self.config.horizon
        x = torch.randn(b, horizon, self.action_dim, device=context.device, generator=generator)
        d = 0 if prefix is None else prefix.shape[1]
        if d:
            x[:, :d] = prefix
        for i in range(steps):
            t = torch.full((b, horizon), i / steps, device=context.device)
            t[:, :d] = 1.0
            v = self.velocity(x, t, context, mask)
            v[:, :d] = 0.0
            x = x + v / steps
        return x

    def _stack_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in self.cameras], dim=-4)
        return batch

    def _history(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        keys = (OBS_STATE, OBS_IMAGES) if self.config.use_state else (OBS_IMAGES,)
        for key in keys:
            queue = self._queues[key]
            queue.append(batch[key])
            while len(queue) < queue.maxlen:
                queue.append(batch[key])
        history = {key: torch.stack(list(self._queues[key]), dim=1) for key in keys}
        history[OBS_LANGUAGE_TOKENS], history[OBS_LANGUAGE_ATTENTION_MASK] = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        return history

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], prefix: Tensor | None = None, inference_delay: int | None = None,
                             prev_chunk_left_over: Tensor | None = None, **kwargs) -> Tensor:
        """The next ``n_action_steps`` normalised actions. ``prefix``: normalised actions the robot will still execute from
        the previous chunk while this one is computed (at most ``rtc_max_prefix``); they come back unchanged at the start.
        ``inference_delay`` + ``prev_chunk_left_over`` is the same thing in the form deploy/rtc_driver.py passes it."""
        if prev_chunk_left_over is not None and inference_delay:
            if prefix is not None:
                raise ValueError("give either prefix or prev_chunk_left_over, not both")
            d = inference_delay
            if d > self.config.rtc_max_prefix:   # trained with prefixes up to rtc_max_prefix: anchor on that many
                if not getattr(self, "_warned_delay", False):
                    print(f"[humanoid_ditx] inference delay {d} > rtc_max_prefix {self.config.rtc_max_prefix}: prefix cut to "
                          f"{self.config.rtc_max_prefix} (the RTC queue still skips the {d} executed actions)", flush=True)
                    self._warned_delay = True
                d = self.config.rtc_max_prefix
            left = prev_chunk_left_over if prev_chunk_left_over.dim() == 3 else prev_chunk_left_over[None]
            prefix = left[:, :d].to(self.camera_emb.device)
        history = self._history(self._stack_images(batch))
        start = self.config.n_obs_steps - 1
        if prefix is not None and start:
            raise ValueError("prefix needs n_obs_steps = 1 (the chunk must start at the current step)")
        context, mask = self.encode(history)
        return self.sample(context, prefix=prefix, mask=mask)[:, start:start + self.config.n_action_steps]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if not self._queues[ACTION]:
            self._queues[ACTION].extend(self.predict_action_chunk(batch).transpose(0, 1))
        else:
            batch = self._stack_images(batch)
            keys = (OBS_STATE, OBS_IMAGES) if self.config.use_state else (OBS_IMAGES,)
            for key in keys:
                self._queues[key].append(batch[key])
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        cfg = self.config
        # with a one-frame history lerobot squeezes the time axis off video frames (not off the state): put it back
        batch = {k: v.unsqueeze(1) if k in self.cameras and v.dim() == 4 else v for k, v in batch.items()}
        context, mask = self.encode(self._stack_images(batch))
        actions = batch[ACTION]
        b, horizon = actions.shape[:2]
        noise = torch.randn_like(actions)
        u = torch.distributions.Beta(cfg.time_beta_alpha, cfg.time_beta_beta).sample((b,)).to(actions.device)
        t = (cfg.time_max * (1 - u))[:, None].expand(b, horizon).clone()
        # real-time chunking: a random prefix of clean actions (flow time 1), left out of the loss
        d = torch.randint(0, cfg.rtc_max_prefix + 1, (b,), device=actions.device)
        given = torch.arange(horizon, device=actions.device)[None] < d[:, None]
        t[given] = 1.0
        x = (1 - t[..., None]) * noise + t[..., None] * actions
        err = (self.velocity(x, t, context, mask) - (actions - noise)).pow(2).mean(-1)
        if "action_is_pad" in batch:
            given = given | batch["action_is_pad"]
        weight = (~given).float()
        loss = (err * weight).sum() / weight.sum().clamp(min=1)
        return loss, {"loss": loss.item(), "film_scale": self.film[-1].weight.abs().mean().item()}
