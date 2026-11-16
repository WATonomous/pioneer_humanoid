"""The RL token: a small encoder squeezes the frozen base's context tokens (B, N, D) into one vector z, trained by
reconstructing those tokens from z alone (the paper's information bottleneck). Only the encoder is used during RL.

Deviation from the paper: the decoder reconstructs all tokens in parallel from z plus learned position queries,
instead of autoregressively.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class RLToken(nn.Module):
    def __init__(self, in_dim: int, max_tokens: int, dim: int = 256, z_dim: int = 256, layers: int = 2, heads: int = 4):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        self.rl = nn.Parameter(torch.randn(1, 1, dim) * 0.02)          # the learned <rl> query token
        make = lambda: nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)  # noqa: E731
        self.encoder = nn.TransformerEncoder(make(), layers)
        self.to_z = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, z_dim))
        self.from_z = nn.Linear(z_dim, dim)
        self.queries = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        self.decoder = nn.TransformerEncoder(make(), layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, in_dim))

    def encode(self, context: Tensor, mask: Tensor | None = None) -> Tensor:
        """context (B, N, D), mask (B, N) True = real token -> z (B, z_dim)."""
        b, n, _ = context.shape
        x = torch.cat([self.rl.expand(b, -1, -1), self.inp(context) + self.pos[:, :n]], dim=1)
        pad = None if mask is None else torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~mask], dim=1)
        return self.to_z(self.encoder(x, src_key_padding_mask=pad)[:, 0])

    def loss(self, context: Tensor, mask: Tensor | None = None) -> Tensor:
        """Reconstruction MSE of the (layer-normed) context tokens from z, over real tokens only."""
        b, n, _ = context.shape
        z = self.encode(context, mask)
        h = self.decoder(self.queries[:, :n].expand(b, -1, -1) + self.from_z(z)[:, None])
        target = nn.functional.layer_norm(context, context.shape[-1:])
        err = (self.out(h) - target).pow(2).mean(-1)
        w = torch.ones_like(err) if mask is None else mask.float()
        return (err * w).sum() / w.sum()
