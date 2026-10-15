"""Patch Policy (arXiv 2607.18236) on LeRobot's VQ-BeT: dense frozen ViT patch tokens and a block-causal trunk.

With PATCH_VQBET=1, importing this module changes ``--policy.type=vqbet`` in three ways, following the paper:
  * the image encoder is a frozen DINOv2 ViT-S/14 at 224 x 224 and returns all 256 patch tokens (384-d) of each
    image, with no pooling (VQ-BeT's own encoder squeezes an image into one vector);
  * each observation step enters the GPT as [patches of camera 1, patches of camera 2, ..., state, action query],
    with the GPT's learned position embedding over the flattened sequence;
  * the attention mask is block-causal: every token sees all tokens of its own step and of earlier steps.
The action chunk is read from the last token of each step, as before. The paper's sim policies use images only;
the robot state is kept here as one more token per step.

Train with ``train_patch_vqbet.py`` (lerobot-train with the patch on). Needs ``--policy.n_action_pred_token=1`` and
``--policy.gpt_block_size`` >= n_obs_steps * (cameras * 256 + 2). sim_policy_eval.py imports this module, so set
PATCH_VQBET=1 when evaluating such a checkpoint too.
"""
from __future__ import annotations

import os

import einops
import torch
from torch import nn

PATCHES, DIM, SIZE = 256, 384, 224
CACHE = os.path.expanduser("~/.cache/pioneer_backbones")


class PatchEncoder(nn.Module):
    """(B, 3, H, W) images in [0, 1] -> (B, 256, 384) patch tokens of a frozen DINOv2 ViT-S/14."""

    feature_dim = DIM

    def __init__(self, config):
        super().__init__()
        from transformers import Dinov2Model

        self.vit = Dinov2Model.from_pretrained("facebook/dinov2-small", cache_dir=CACHE).eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        return super().train(False)          # frozen: never in training mode

    @torch.no_grad()
    def forward(self, x):
        x = nn.functional.interpolate(x, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
        return self.vit(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 1:]


def _block_causal(model) -> None:
    """Replace the GPT's lower-triangular mask: a token attends to every token of its own step and earlier ones."""
    config = model.config
    per_step = model.num_images * PATCHES + 2
    if config.n_action_pred_token != 1:
        raise ValueError("Patch Policy reads one action chunk per observation step: set --policy.n_action_pred_token=1")
    if config.gpt_block_size < config.n_obs_steps * per_step:
        raise ValueError(f"--policy.gpt_block_size must be at least {config.n_obs_steps * per_step}")
    step = torch.arange(config.gpt_block_size) // per_step
    mask = (step[None, :] <= step[:, None]).float().view(1, 1, config.gpt_block_size, config.gpt_block_size)
    for module in model.policy.modules():
        if isinstance(getattr(module, "bias", None), torch.Tensor) and module.bias.dim() == 4:
            module.bias = mask.clone()
    model.per_step = per_step


def _attention(self, x):
    """CausalSelfAttention.forward with PyTorch's fused attention: the same result, without storing a (B, heads, T, T)
    matrix per layer (T is ~1000 here, against a few dozen in plain VQ-BeT)."""
    b, t, c = x.size()
    q, k, v = (y.view(b, t, self.gpt_n_head, c // self.gpt_n_head).transpose(1, 2) for y in self.c_attn(x).split(self.gpt_hidden_dim, dim=2))
    y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.bias[:, :, :t, :t].bool(),
                                                   dropout_p=self.attn_dropout.p if self.training else 0.0)
    return self.resid_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c)))


def _forward(self, batch, rollout: bool):
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
    patches = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
    patches = einops.rearrange(self.rgb_feature_projector(patches), "(b s n) p d -> b s (n p) d", b=batch_size, s=n_obs_steps)
    state = self.state_projector(batch[OBS_STATE])[:, :, None]
    query = einops.repeat(self.action_token, "1 1 d -> b s 1 d", b=batch_size, s=n_obs_steps)
    tokens = einops.rearrange(torch.cat([patches, state, query], dim=2), "b s t d -> b (s t) d")
    features = self.policy(tokens)[:, self.per_step - 1::self.per_step]      # the query token closing each step
    out = self.action_head(features)
    if rollout:
        return out["predicted_action"][:, n_obs_steps - 1, :].reshape(batch_size, self.config.action_chunk_size, -1)
    loss = self.action_head.loss_fn(out, batch[ACTION][:, self.select_target_actions_indices], reduction="mean")
    return out, loss


if os.environ.get("PATCH_VQBET"):
    import lerobot.policies.vqbet.modeling_vqbet as M

    _init = M.VQBeTModel.__init__

    def _patched_init(self, config):
        _init(self, config)
        _block_causal(self)

    def _validate_features(self) -> None:                         # as VQBeTConfig's, without its one-camera limit
        shapes = {ft.shape for ft in self.image_features.values()}
        if len(shapes) != 1:
            raise ValueError(f"all cameras must have the same image shape, got {shapes}")

    M.VQBeTConfig.validate_features = _validate_features
    M.VQBeTRgbEncoder = PatchEncoder
    M.VQBeTModel.__init__ = _patched_init
    M.VQBeTModel.forward = _forward
    import lerobot.policies.vqbet.vqbet_utils as U

    U.CausalSelfAttention.forward = _attention
