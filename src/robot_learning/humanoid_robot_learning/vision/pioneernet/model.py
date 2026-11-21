"""PioneerNet v13: an RGB-D DINOv3 ViT-B/16 (86M) distilled from DINOv2-S (colour) and DeFM-S (depth).

Architecture: timm's DINOv3 ViT-B/16 plus a second patch embedding for depth whose output is added to the RGB patch
embedding (zero-initialised in training, so distillation started from plain DINOv3). Everything after the patch
embedding is the unchanged ViT, so the tokens have DINOv3's layout: 1 class token + 4 register tokens + the patch grid.

Inputs (``preprocess``):
    rgb    float (N, 3, H, W) in [0, 1]           -> resized to 256x320, ImageNet-normalised
    depth  float (N, H, W) or (N, 1, H, W) metres -> DeFM's 3-channel log encoding, resized, DeFM-normalised
           (0 = no reading), or None for a camera without a depth sensor: the depth input is then 0 after
           normalisation, the same blanking the encoder saw in training (colour or depth dropped per sample)

Outputs: ``forward`` -> (class token (N, 768), patch grid (N, 768, 16, 20));
         ``tokens``  -> all tokens (N, 5 + 320, 768), the layout of a Hugging Face ViT's last_hidden_state.

The network was trained and benchmarked FROZEN; fine-tuning it with a policy has cost robustness in our tests.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DEFM_MEAN, DEFM_STD = (0.24888, 0.49562, 0.492858), (0.139357, 0.271314, 0.297177)   # defm.utils.utils
TIMM_NAMES = {"dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m", "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m"}


def _resize(x: Tensor, hw: tuple[int, int]) -> Tensor:
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=True)


def defm_depth3(d: Tensor) -> Tensor:
    """(N, H, W) metres -> (N, 3, H, W): DeFM's metric-aware log encoding (defm.utils.preprocess_depth_batch without the
    resize / normalisation). Channel 1: log depth over 0..100 m, 2: over 0..9 m (clipped), 3: per-image min-max."""
    t = torch.nan_to_num(d, 0.0, 0.0, 0.0).clamp(0, 100.0)
    log = torch.log1p(t)
    c1 = log / math.log1p(100.0)
    c2 = (log / math.log1p(9.0)).clamp(0, 1)
    flat = log.flatten(1)
    mn, mx = flat.min(1)[0].view(-1, 1, 1), flat.max(1)[0].view(-1, 1, 1)
    den = mx - mn
    c3 = torch.where(den > 0, (log - mn) / den.clamp_min(1e-12), torch.zeros_like(log))
    return torch.stack([c1, c2, c3], 1)


class PioneerNet(nn.Module):
    """RGB-D ViT encoder; ``state_dict`` keys match the distillation checkpoints (``body.*``, ``pe_depth.*``, heads)."""

    def __init__(self, rgb_dim: int = 384, depth_dim: int = 384, hw: tuple[int, int] = (256, 320),
                 backbone: str = "dinov3_vitb16", pretrained_backbone: bool = False, **_unused):
        super().__init__()
        import timm
        self.body = timm.create_model(TIMM_NAMES[backbone], pretrained=pretrained_backbone, num_classes=0,
                                      dynamic_img_size=True)
        pe = self.body.patch_embed
        self.pe_depth = copy.deepcopy(pe)
        rgb_forward = pe.forward
        # the ViT sees the sum of the RGB and the depth patch embeddings of a 6-channel input
        pe.forward = lambda x: rgb_forward(x[:, :3]) + self.pe_depth(x[:, 3:])
        self.dim = self.body.num_features
        self.prefix = self.body.num_prefix_tokens          # class + register tokens (5 for DINOv3)
        self.patch = self.body.patch_embed.patch_size[0]
        self.hw = tuple(hw)
        self.grid_hw = (self.hw[0] // self.patch, self.hw[1] // self.patch)
        # distillation heads: only used in training (they map tokens to the teachers' feature spaces)
        self.head_rgb = nn.Conv2d(self.dim, rgb_dim, 1)
        self.head_depth = nn.Conv2d(self.dim, depth_dim, 1)
        self.g_rgb = nn.Linear(self.dim, rgb_dim)
        self.g_depth = nn.Linear(self.dim, depth_dim)
        self.register_buffer("mean_rgb", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std_rgb", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("mean_d", torch.tensor(DEFM_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std_d", torch.tensor(DEFM_STD).view(1, 3, 1, 1))

    def preprocess(self, rgb: Tensor, depth: Tensor | None = None) -> Tensor:
        """rgb float (N, 3, H, W) in [0, 1]; depth (N, H, W) / (N, 1, H, W) metres or None -> (N, 6, 256, 320)."""
        x = (_resize(rgb.float(), self.hw) - self.mean_rgb) / self.std_rgb
        if depth is None:
            z = torch.zeros_like(x)
        else:
            if depth.dim() == 4:
                depth = depth[:, 0]
            z = (_resize(defm_depth3(depth.float()), self.hw) - self.mean_d) / self.std_d
        return torch.cat([x, z], 1)

    def tokens(self, x: Tensor) -> Tensor:
        """Preprocessed (N, 6, h, w) -> all tokens (N, prefix + patches, dim)."""
        return self.body.forward_features(x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Preprocessed (N, 6, h, w) -> class token (N, dim) and patch grid (N, dim, h / 16, w / 16)."""
        t = self.tokens(x)
        h, w = x.shape[2] // self.patch, x.shape[3] // self.patch
        return t[:, 0], t[:, self.prefix:].transpose(1, 2).reshape(len(x), -1, h, w)

    def encode(self, rgb: Tensor, depth: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Convenience: raw rgb (+ depth) -> (class token, patch grid)."""
        return self(self.preprocess(rgb, depth))


def load_pioneernet(source: str | Path, device: str | torch.device = "cpu", freeze: bool = True) -> PioneerNet:
    """Load a PioneerNet checkpoint: a local ``.pt`` path, or ``hf:<repo_id>/<filename>`` on the Hugging Face Hub.
    The checkpoint is the dict the distillation writes: {"state", "arch", "rgb_dim", "depth_dim", teachers, ...}."""
    source = str(source)
    if source.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        repo, _, filename = source[3:].rpartition("/")
        source = hf_hub_download(repo_id=repo, filename=filename)
    ck = torch.load(source, map_location="cpu", weights_only=False)
    arch = ck.get("arch", {})
    if not arch.get("backbone", "resnet18").startswith("dinov3"):
        raise ValueError(f"{source}: backbone {arch.get('backbone', 'resnet18')!r}; this loader handles the DINOv3 ViT "
                         "PioneerNets (v10 and later)")
    model = PioneerNet(ck["rgb_dim"], ck["depth_dim"], **arch)
    model.load_state_dict(ck["state"])
    model.meta = {k: v for k, v in ck.items() if k != "state"}
    model = model.to(device).eval()
    if freeze:
        model.requires_grad_(False)
    return model
