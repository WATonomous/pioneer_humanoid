"""Dense pretrained ViT patch tokens as ACT's vision backbone (the idea of Patch Policy, arXiv 2607.18236).

Importing this module registers ``torchvision.models.resnet_dinov2s``: a DINOv2 ViT-S/14 whose patch tokens are
returned as a (B, 384, h, w) feature map, shaped so LeRobot's ACT can use it in place of ResNet-18's layer4
(``--policy.vision_backbone=resnet_dinov2s``; the name keeps the "resnet" prefix ACTConfig insists on).
Images are resized to 168 x 224 (12 x 16 = 192 patch tokens per camera). Train with ``train_patch_act.py``;
sim_policy_eval.py / qplan.py import this module so such checkpoints load.
"""
from __future__ import annotations

import os

import torch
import torchvision
from torch import nn

SIZE = (168, 224)        # multiples of the 14 px patch
CACHE = os.path.expanduser("~/.cache/pioneer_backbones")


class _PatchTokens(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        from transformers import Dinov2Model
        self.vit = Dinov2Model.from_pretrained(name, cache_dir=CACHE)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=SIZE, mode="bilinear", antialias=True, align_corners=False)
        tokens = self.vit(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
        return tokens.transpose(1, 2).reshape(x.shape[0], -1, SIZE[0] // 14, SIZE[1] // 14)


def resnet_dinov2s(weights=None, norm_layer=None, replace_stride_with_dilation=None):
    model = nn.Module()
    model.layer4 = _PatchTokens("facebook/dinov2-small")
    model.fc = nn.Linear(384, 1)           # only its in_features is read (ACT sizes its input projection from it)
    return model


torchvision.models.resnet_dinov2s = resnet_dinov2s
