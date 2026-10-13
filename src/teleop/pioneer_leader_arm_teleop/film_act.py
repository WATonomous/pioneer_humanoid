"""Language conditioning inside ACT: FiLM on the visual features.

With ``ACT_FILM=1`` in the environment, importing this module makes LeRobot's ACT modulate every camera's ResNet
feature map with the instruction: the state's dimensions beyond the 7 joint values (the instruction encoding written
by add_subtask_state.py) go through a small MLP to a per-channel scale and shift, feature = feature * (1 + g) + b,
before the tokens enter the transformer. So what the policy sees depends on what it was told, instead of the
instruction being one token among ~160 that it can ignore (which is what it does: see the research notes).

Train with train_film_act.py (same arguments as lerobot-train). The rollout code turns the patch on by itself for a
checkpoint whose ``subtask_state.json`` has ``"film": true``.
"""
from __future__ import annotations

import os

import torch
from torch import nn

ARM_DIMS = 7


class _FilmBackbone(nn.Module):
    """ACT's backbone, with its output feature map scaled and shifted by the instruction."""

    def __init__(self, backbone: nn.Module, cond_dim: int, channels: int):
        super().__init__()
        self.backbone = backbone
        self.film = nn.Sequential(nn.Linear(cond_dim, 256), nn.ReLU(), nn.Linear(256, 2 * channels))
        nn.init.normal_(self.film[-1].weight, std=0.05)     # a small modulation from the start (ACT's lr is 1e-5:
        nn.init.zeros_(self.film[-1].bias)                  # from zero it would take most of the run to matter)
        self.cond = None

    def forward(self, img):
        out = self.backbone(img)
        if self.cond is not None:
            g, b = self.film(self.cond).chunk(2, dim=-1)
            out = dict(out)
            out["feature_map"] = out["feature_map"] * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return out


def _install() -> None:
    from lerobot.policies.act import modeling_act
    from lerobot.utils.constants import OBS_STATE

    if getattr(modeling_act.ACT, "_film_patched", False):
        return
    init, forward = modeling_act.ACT.__init__, modeling_act.ACT.forward

    def __init__(self, config):
        init(self, config)
        cond_dim = config.robot_state_feature.shape[0] - ARM_DIMS if config.robot_state_feature else 0
        if cond_dim > 0 and config.image_features:
            channels = self.encoder_img_feat_input_proj.in_channels
            self.backbone = _FilmBackbone(self.backbone, cond_dim, channels)

    def fwd(self, batch):
        if isinstance(getattr(self, "backbone", None), _FilmBackbone):
            self.backbone.cond = batch[OBS_STATE][:, ARM_DIMS:]
        return forward(self, batch)

    modeling_act.ACT.__init__, modeling_act.ACT.forward, modeling_act.ACT._film_patched = __init__, fwd, True


if os.environ.get("ACT_FILM") == "1":
    _install()
