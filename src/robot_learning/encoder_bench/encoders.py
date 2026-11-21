"""Encoder zoo with one interface, plus depth-sensor noise models.

    enc = load("defm_s14")              # nn.Module, frozen, on cuda
    g, tok = enc(enc.prep(batch))       # g: (B, C) global vector, tok: (B, C, h, w) patch grid

``enc.modality`` is "rgb" (input: uint8 (B, H, W, 3) tensor) or "depth" (input: float metres (B, H, W)).
All encoders see ~224 px high, 4:3-ish inputs: ViT/14 at 224x294 (16x21 patches), ViT/16 at 224x288
(14x18), CNNs at 224x288 (DeFM's BiFPN wants multiples of 32).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
REPO_MODELS = Path(__file__).resolve().parents[3] / "models"   # repo-level checkpoints, e.g. models/pioneernet_v5/
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIPN = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
HALF = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

# name -> (family, modality, description)
ZOO = {
    # --- RGB: what the team uses today --------------------------------------------------
    "r18_imnet":      ("tv_resnet18", "rgb", "ResNet-18 ImageNet (ACT's backbone)"),
    "clip_b16":       ("timm:vit_base_patch16_clip_224.openai", "rgb", "CLIP ViT-B/16 (humanoid_dit)"),
    "siglip_b16":     ("timm:vit_base_patch16_siglip_224.webli", "rgb", "SigLIP ViT-B/16 (pi0.5 / SmolVLA family)"),
    # --- RGB: self-supervised foundation models -----------------------------------------
    "dinov2_s14":     ("timm:vit_small_patch14_dinov2.lvd142m", "rgb", "DINOv2 ViT-S/14"),
    "dinov2_b14":     ("timm:vit_base_patch14_dinov2.lvd142m", "rgb", "DINOv2 ViT-B/14"),
    # --- Depth: DeFM ----------------------------------------------------------------------
    "defm_s14":       ("defm:defm_vit_s14", "depth", "DeFM ViT-S/14 (depth)"),
    "defm_l14":       ("defm:defm_vit_l14", "depth", "DeFM ViT-L/14 (depth)"),
    "defm_r18":       ("defm:defm_resnet18", "depth", "DeFM ResNet-18 (depth)"),
    "defm_r50":       ("defm:defm_resnet50", "depth", "DeFM ResNet-50 (depth)"),
    # --- Depth through RGB models (the DeFM paper's baselines: min-max depth -> 3 channels)
    "r18_imnet_d":    ("tv_resnet18", "depth", "ResNet-18 ImageNet on depth"),
    "dinov2_s14_d":   ("timm:vit_small_patch14_dinov2.lvd142m", "depth", "DINOv2 ViT-S/14 on depth"),
    # --- double input resolution (448x588 -> 32x42 patches): is patch size the accuracy limit? ----------
    # --- prior multi-teacher distilled encoders (closest related work; RGB only) -------------------
    "theia_s16":      ("theia:theaiinstitute/theia-small-patch16-224-cddsv", "rgb", "Theia-S/16 (CLIP+DINOv2+ViT+SAM+Depth-Anything distilled)"),
    "radio_b16":      ("radio:nvidia/C-RADIOv2-B", "rgb", "C-RADIOv2-B/16 (multi-teacher distilled)"),
    "cradio4_so400m": ("radio:nvidia/C-RADIOv4-SO400M", "rgb", "C-RADIOv4-SO400M/16 (NVIDIA, 2026)"),
    "siglino_30m":    ("siglino:tiiuae/siglino-30M", "rgb", "SigLino-30M (TII, 2026; SigLIP2+DINOv3 distilled)"),
    "siglino_70m":    ("siglino:tiiuae/siglino-70M", "rgb", "SigLino-70M (TII, 2026; SigLIP2+DINOv3 distilled)"),
    # --- 2025-26 generation (added 2026-11-14): DINOv3 and SigLIP2, all via timm (ungated weights) -----------------
    "dinov3_s16":     ("timm:vit_small_patch16_dinov3.lvd1689m", "rgb", "DINOv3 ViT-S/16 (Meta, 2025)"),
    "dinov3_b16":     ("timm:vit_base_patch16_dinov3.lvd1689m", "rgb", "DINOv3 ViT-B/16 (Meta, 2025)"),
    "dinov3_cnxt":    ("timm:convnext_tiny.dinov3_lvd1689m", "rgb", "DINOv3 ConvNeXt-T (Meta, 2025; distilled from DINOv3-7B)"),
    "siglip2_b16":    ("timm:vit_base_patch16_siglip_224.v2_webli", "rgb", "SigLIP2 ViT-B/16 (Google, 2025)"),
    "dinov2_s14_hr":  ("timm:vit_small_patch14_dinov2.lvd142m", "rgb", "DINOv2 ViT-S/14 @448"),
    "defm_s14_hr":    ("defm:defm_vit_s14", "depth", "DeFM ViT-S/14 @448 (depth)"),
}
HW_OVERRIDE = {"dinov2_s14_hr": (448, 588), "defm_s14_hr": (448, 588)}


def _resize(x, hw):
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=True)


def minmax_depth3(d: torch.Tensor) -> torch.Tensor:
    """(B,H,W) metres -> (B,3,H,W) in 0..1, per-image min-max over valid pixels (invalid = 0 stays 0)."""
    valid = d > 0
    big = torch.where(valid, d, torch.full_like(d, 1e9)).flatten(1).min(1)[0].view(-1, 1, 1)
    mx = d.flatten(1).max(1)[0].view(-1, 1, 1)
    x = torch.where(valid, (d - big) / (mx - big).clamp_min(1e-6), torch.zeros_like(d))
    return x.unsqueeze(1).expand(-1, 3, -1, -1)


def defm_depth3(d: torch.Tensor) -> torch.Tensor:
    """DeFM's metric-aware 3-channel log encoding (defm.utils.preprocess_depth_batch, without resize/normalise)."""
    t = torch.nan_to_num(d, 0.0, 0.0, 0.0).clamp(0, 100.0)
    log = torch.log1p(t)
    c1 = log / math.log1p(100.0)
    c2 = (log / math.log1p(9.0)).clamp(0, 1)
    flat = log.flatten(1)
    mn, mx = flat.min(1)[0].view(-1, 1, 1), flat.max(1)[0].view(-1, 1, 1)
    den = mx - mn
    c3 = torch.where(den > 0, (log - mn) / den.clamp_min(1e-12), torch.zeros_like(log))
    return torch.stack([c1, c2, c3], 1)


class Encoder(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        fam, self.modality, self.desc = ZOO[name]
        self.name, self.fam = name, fam
        if fam == "tv_resnet18":
            import torchvision
            m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            self.body = nn.Sequential(*list(m.children())[:-2])
            self.dim, self.hw, self.norm = 512, (224, 288), IMNET
            self.kind = "cnn"
        elif fam.startswith("timm:convnext"):   # CNN from timm: stride-32 feature map, like the ResNets
            import timm
            self.body = timm.create_model(fam[5:], pretrained=True, num_classes=0)
            self.dim, self.hw, self.norm, self.kind = self.body.num_features, (224, 288), IMNET, "cnn_timm"
        elif fam.startswith("timm:"):
            import timm
            self.body = timm.create_model(fam[5:], pretrained=True, num_classes=0, dynamic_img_size=True)
            patch = self.body.patch_embed.patch_size[0]
            self.hw = (224, 294) if patch == 14 else (224, 288)
            self.dim = self.body.num_features
            self.norm = CLIPN if "clip" in fam else HALF if "siglip" in fam else IMNET
            self.kind = "vit"
        elif fam.startswith("defm:"):
            from defm.model_factory import create_defm_model
            self.body = create_defm_model(fam[5:], pretrained=True)
            from defm.utils.utils import DEFM_MEAN, DEFM_STD
            self.norm = (DEFM_MEAN, DEFM_STD)
            if "vit" in fam:
                self.kind, self.hw, self.dim = "defm_vit", (224, 294), self.body.embed_dim
            else:
                self.kind, self.hw = "defm_cnn", (224, 288)
                self.dim = None  # set on first call
        elif fam.startswith("siglino:"):
            # trust_remote_code: official TII repo, user-approved baseline. Takes pre-cut 16x16 patches + grid shape.
            from transformers import AutoModel
            self.body = AutoModel.from_pretrained(fam.split(":", 1)[1], trust_remote_code=True)
            self.norm, self.dim, self.kind, self.hw = HALF, self.body.config.dim, "siglino", (224, 288)
        elif fam.startswith("theia:") or fam.startswith("radio:"):
            # trust_remote_code: official repos, user-approved baselines. Preprocessing is redone on the GPU.
            from transformers import AutoModel
            m = AutoModel.from_pretrained(fam.split(":", 1)[1], trust_remote_code=True)
            if fam.startswith("theia:"):
                self.body = m.backbone.model          # HF ViT (DeiT-S) inside Theia; translators unused
                self.norm, self.dim, self.kind = HALF, 384, "theia"   # Theia's processor: /255 then (x-.5)/.5
            else:
                self.body = m                         # RADIO normalises internally ("input_conditioner"): feed [0,1]
                with torch.no_grad():   # width differs per RADIO size (B 768, SO400M 1152, H 1280)
                    _, _sp = m(torch.rand(1, 3, 224, 288))
                self.norm, self.dim, self.kind = ([0.0] * 3, [1.0] * 3), _sp.shape[-1], "radio"
            self.hw = (224, 288)
        else:
            raise ValueError(fam)
        self.hw = HW_OVERRIDE.get(name, self.hw)
        mean, std = self.norm
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.eval().requires_grad_(False)

    def prep(self, x: torch.Tensor) -> torch.Tensor:
        """Raw batch on GPU -> normalised network input."""
        if self.modality == "rgb":
            x = x.permute(0, 3, 1, 2).float() / 255.0
        elif self.kind.startswith("defm"):
            x = defm_depth3(x.float())
        else:
            x = minmax_depth3(x.float())
        return (_resize(x, self.hw) - self.mean) / self.std

    def forward(self, x):
        if self.kind == "cnn":
            f = self.body(x)
            return f.mean((2, 3)), f
        if self.kind == "cnn_timm":
            f = self.body.forward_features(x)
            return f.mean((2, 3)), f
        if self.kind == "vit":
            t = self.body.forward_features(x)
            npre = self.body.num_prefix_tokens
            h, w = x.shape[2] // self.body.patch_embed.patch_size[0], x.shape[3] // self.body.patch_embed.patch_size[1]
            pt = t[:, npre:]
            g = t[:, 0] if npre > 0 and not getattr(self.body, "no_embed_class", False) and "siglip" not in self.fam else pt.mean(1)
            return g, pt.transpose(1, 2).reshape(len(x), -1, h, w)
        if self.kind == "theia":
            t = self.body(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
            h, w = x.shape[2] // 16, x.shape[3] // 16
            pt = t[:, 1:1 + h * w]
            return pt.mean(1), pt.transpose(1, 2).reshape(len(x), -1, h, w)
        if self.kind == "siglino":
            N, Cc, H, W = x.shape
            h, w = H // 16, W // 16
            pv = x.permute(0, 2, 3, 1).reshape(N, h, 16, w, 16, Cc).permute(0, 1, 3, 2, 4, 5).reshape(N, h * w, -1)
            shapes = torch.tensor([[h, w]] * N, device=x.device)
            with torch.autocast("cuda", enabled=False):   # its RoPE uses torch.polar, which has no bf16 kernel
                out = self.body.float()(pixel_values=pv.float(), spatial_shapes=shapes, compile=False)
            pt = out["patch_features"]["siglino"]
            return out["summary_features"]["siglino"], pt.transpose(1, 2).reshape(N, -1, h, w)
        if self.kind == "radio":
            summary, sp = self.body(x)
            h, w = x.shape[2] // 16, x.shape[3] // 16
            return summary, sp.transpose(1, 2).reshape(len(x), -1, h, w)
        if self.kind == "defm_vit":
            (pt, g), = self.body.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)
            return g, pt
        out = self.body(x)
        return out["global_backbone"], out["dense_bifpn"]["P4"]


class Q8:
    """Frozen token cache, int8 with a per-token absmax scale (rel. error ~0.4%, half of fp16's RAM).
    tokens (N, C, h, w) -> q int8 (N, C, h, w) + scale fp16 (N, 1, h, w)."""

    def __init__(self, n, shape):
        self.q = torch.empty((n, *shape), dtype=torch.int8)
        self.s = torch.empty((n, 1, *shape[1:]), dtype=torch.float16)

    def put(self, i, t):
        t = t.float()
        sc = t.abs().amax(1, keepdim=True).clamp_min(1e-6) / 127
        self.q[i:i + len(t)] = (t / sc).round().clamp(-127, 127).to(torch.int8).cpu()
        self.s[i:i + len(t)] = sc.half().cpu()

    def nbytes(self):
        return self.q.numel() + 2 * self.s.numel()

    def cuda(self):
        self.q, self.s = self.q.cuda(), self.s.cuda()
        return self

    def __len__(self):
        return len(self.q)

    @property
    def shape(self):
        return self.q.shape

    def gpu_copy(self):
        c = Q8.__new__(Q8)
        c.q, c.s = self.q.cuda(), self.s.cuda()
        return c

    def get(self, idx):
        """idx: CPU LongTensor -> float32 tokens on cuda."""
        if self.q.is_cuda:
            idx = idx.cuda()
            return self.q[idx].float() * self.s[idx].float()
        return self.q[idx].cuda(non_blocking=True).float() * self.s[idx].cuda(non_blocking=True).float()


def needs(enc) -> set:
    return {"rgb", "depth"} if enc.modality == "rgbd" else {enc.modality}


def run(enc, rgb=None, depth=None):
    """rgb: uint8 (B,H,W,3) cuda; depth: float metres (B,H,W) cuda -> (global, tokens) in bf16 autocast."""
    x = enc.prep(rgb, depth) if enc.modality == "rgbd" else enc.prep(rgb if enc.modality == "rgb" else depth)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return enc(x)


def load(name: str, ckpt: str | None = None):
    if name.endswith(":rgb"):
        # An RGB-D encoder on a camera with no depth sensor (the real wrist webcam): its depth input is blanked
        # the way modality dropout blanks it in training (0 after normalisation), so the stream needs RGB only.
        enc = load(name[:-4], ckpt)
        assert enc.modality == "rgbd", f"{name}: ':rgb' only applies to RGB-D encoders"
        base_prep = enc.prep

        def prep_rgb(rgb):
            x = base_prep(rgb, torch.ones(rgb.shape[:3], device=rgb.device))
            return torch.cat([x[:, :3], torch.zeros_like(x[:, 3:])], 1)
        enc.prep, enc.modality, enc.name = prep_rgb, "rgb", name
        return enc
    if name.startswith("pioneernet"):
        from distill import CKPT, PioneerNet
        # A checkpoint committed to the repo (models/<name>/<name>.pt) wins over the bench's own results folder.
        committed = REPO_MODELS / name / f"{name}.pt"
        path = ckpt or (str(committed) if committed.exists() else
                        str(CKPT) if name == "pioneernet" else str(CKPT).replace("pioneernet", name))
        sd = torch.load(path, map_location="cpu")
        enc = PioneerNet(sd["rgb_dim"], sd["depth_dim"], init=False, **sd.get("arch", {}))
        enc.load_state_dict(sd["state"])
        enc.name, enc.desc = name, f"PioneerNet RGB-D ResNet-18 (ours; {sd['rgb_teacher']} + {sd['depth_teacher']})"
        enc = enc.to(DEV).eval().requires_grad_(False)
        return enc
    enc = Encoder(name).to(DEV)
    if enc.dim is None:
        with torch.no_grad():
            g, t = enc(torch.zeros(1, 3, *enc.hw, device=DEV))
        enc.gdim, enc.dim = g.shape[1], t.shape[1]
    else:
        enc.gdim = enc.dim
    return enc


# ------------------------------------------------------------------------------------------------
# Depth sensor models (applied to clean sim depth, metres, (B, H, W) on GPU)
# ------------------------------------------------------------------------------------------------

def _edge_mask(d: torch.Tensor, thresh: float) -> torch.Tensor:
    dx = (d[:, :, 1:] - d[:, :, :-1]).abs()
    dy = (d[:, 1:, :] - d[:, :-1, :]).abs()
    e = torch.zeros_like(d, dtype=torch.bool)
    e[:, :, 1:] |= dx > thresh
    e[:, :, :-1] |= dx > thresh
    e[:, 1:, :] |= dy > thresh
    e[:, :-1, :] |= dy > thresh
    return e


def _blobs(shape, frac, scale, gen, device):
    """Random low-frequency dropout blobs covering ~frac of the image."""
    B, H, W = shape
    n = torch.rand(B, 1, max(2, H // scale), max(2, W // scale), generator=gen, device=device)
    n = F.interpolate(n, size=(H, W), mode="bicubic", align_corners=False)[:, 0]
    thr = torch.quantile(n.flatten(1), 1 - frac, dim=1).view(B, 1, 1)
    return n > thr


def stereo_noise(d: torch.Tensor, *, focal_px: float, baseline: float, sigma_disp: float, min_z: float, max_z: float,
                 edge_drop: float, hole_frac: float, gen: torch.Generator) -> torch.Tensor:
    """Active/passive stereo depth: disparity noise (sigma_z = z^2 sigma_d / (f b)), min/max range dropout,
    flying-pixel dropout at depth edges, random holes, 1 mm quantisation. Invalid pixels -> 0 (like librealsense)."""
    dev = d.device
    noise = torch.randn(d.shape, generator=gen, device=dev)
    # spatially correlated part (stereo matching windows): blur half of the noise
    corr = F.avg_pool2d(torch.randn(d.shape, generator=gen, device=dev).unsqueeze(1), 5, 1, 2)[:, 0] * 2.2
    sig = d ** 2 * sigma_disp / (focal_px * baseline)
    z = d + sig * (0.6 * noise + 0.8 * corr)
    bad = (d < min_z) | (d > max_z)
    bad |= _edge_mask(d, 0.02) & (torch.rand(d.shape, generator=gen, device=dev) < edge_drop)
    bad |= _blobs(d.shape, hole_frac, 24, gen, dev)
    z = torch.round(z * 1000) / 1000
    return torch.where(bad, torch.zeros_like(z), z)


# Chest RealSense D455 at 320x240: f ~ 0.5 * native 640 px focal -> ~190 px; baseline 95 mm; min-Z ~0.4 m.
def d455(d, gen):
    return stereo_noise(d, focal_px=190.0, baseline=0.095, sigma_disp=0.12, min_z=0.42, max_z=6.0,
                        edge_drop=0.5, hole_frac=0.04, gen=gen)


# Wrist: depth computed from a small passive stereo pair (no native depth), ~723 px focal at 640 -> 361 px at
# 320; baseline ~30 mm; worse matching (passive, textureless surfaces) -> larger disparity noise and holes.
def wrist_stereo(d, gen):
    return stereo_noise(d, focal_px=361.0, baseline=0.03, sigma_disp=0.35, min_z=0.07, max_z=2.0,
                        edge_drop=0.7, hole_frac=0.10, gen=gen)


NOISE = {"ego": d455, "wrist_left": wrist_stereo}
