"""Shared pieces: the frozen DINOv2 encoder, the predictor and the viewing decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F

HIST = 2  # frames of history the predictor sees
STRIDE = 5  # one world-model step = 5 recorded frames = 0.2 s at 25 fps
ACT_DIM = 7  # 6 arm joints + gripper
STATE_DIM = 7


class DinoEncoder:
    """Frozen DINOv2-small: an RGB frame -> 16x16 patch tokens (384 dims), average-pooled to tok x tok."""

    def __init__(self, tok: int, dev: str = "cuda"):
        from transformers import AutoModel
        if 16 % tok:
            raise ValueError(f"tok must divide 16, got {tok}")
        self.tok, self.dev = tok, dev
        self.net = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x):  # x (T, 3, H, W) float in [0, 1] on dev -> (T, tok*tok, 384) fp16
        x = F.interpolate(x, (224, 224), mode="bilinear", antialias=True)
        h = self.net(pixel_values=((x - self.mean) / self.std).half()).last_hidden_state[:, 1:]
        h = h.view(-1, 16, 16, 384).permute(0, 3, 1, 2)
        return F.avg_pool2d(h.float(), 16 // self.tok).flatten(2).transpose(1, 2).half()


class Predictor(nn.Module):
    """Next frame's tokens (as a change from the last frame), next arm state and next peg height, from the last HIST
    frames' tokens, the arm state and the next 0.2 s of actions (one shared action-and-state token)."""

    def __init__(self, tok: int, d=384, w=384, layers=6):
        super().__init__()
        nt = tok * tok
        self.inp = nn.Linear(d, w)
        self.pos = nn.Parameter(torch.zeros(1, HIST * nt, w))
        self.time = nn.Parameter(torch.zeros(1, HIST, 1, w))
        self.act = nn.Sequential(nn.Linear(STRIDE * ACT_DIM + STATE_DIM, w), nn.GELU(), nn.Linear(w, w))
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(w, 8, 4 * w, 0.0, batch_first=True,
                                                                       norm_first=True, activation="gelu"), layers)
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, d))
        self.state = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, STATE_DIM))
        self.peg = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 1))  # peg height above its start (m)
        self.nt = nt
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, zh, s, a):  # zh (B, HIST, NT, d), s (B, 7) normalised, a (B, 35) normalised
        B = zh.shape[0]
        x = (self.inp(zh) + self.time).reshape(B, HIST * self.nt, -1) + self.pos
        c = self.act(torch.cat([a, s], -1))[:, None]
        h = self.blocks(torch.cat([c, x], 1))
        return zh[:, -1] + self.out(h[:, 1 + (HIST - 1) * self.nt:]), self.state(h[:, 0]), self.peg(h[:, 0])[:, 0]


class Decoder(nn.Module):
    """Tokens -> 96x128 RGB, for looking at predictions only (trained on detached tokens)."""

    def __init__(self, tok: int, d=384):
        super().__init__()
        self.tok = tok

        def up(i, o):
            return nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.net = nn.Sequential(nn.Conv2d(d, 256, 1), up(256, 192), up(192, 128), up(128, 64), up(64, 32),
                                 nn.Conv2d(32, 3, 3, padding=1))

    def forward(self, zt):  # (B, NT, d) -> (B, 3, 96, 128)
        x = zt.transpose(1, 2).reshape(-1, zt.shape[-1], self.tok, self.tok)
        x = F.adaptive_avg_pool2d(x, 8)  # always decodes from 8x8
        return torch.sigmoid(F.interpolate(self.net(x), (96, 128), mode="bilinear"))
