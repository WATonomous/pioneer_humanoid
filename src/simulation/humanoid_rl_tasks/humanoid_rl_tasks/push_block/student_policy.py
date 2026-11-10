"""Minimal two-camera CNN+MLP student for push-block vision distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# camera name -> term name in the env's ``student_rgb`` group
CAMERA_OBS_TERMS = {"ego": "ego_rgb", "wrist": "wrist_rgb"}


def student_images(obs: dict) -> dict[str, torch.Tensor]:
    """NHWC image per camera from the env observation dict."""
    group = obs["student_rgb"]
    return {camera: group[term] for camera, term in CAMERA_OBS_TERMS.items()}


# Training-time image augmentation (see augment_rgb), applied to /255, mean-subtracted images.
AUG_SHIFT_PX = 4  # px
AUG_CONTRAST = (0.8, 1.2)
AUG_CHANNEL_GAIN = (0.9, 1.1)  # colour cast
AUG_PIXEL_NOISE_STD = 0.02  # in /255 units


def augment_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Random shift, contrast, colour cast and pixel noise for an NHWC batch, each sample drawn independently.
    Stacked frames (channels = 3 x frames) get the same shift, contrast and cast, so augmentation adds no fake motion.

    Brightness is not jittered: the env already subtracts each image's mean, which removes any global brightness shift."""
    n, h, w, c = rgb.shape
    x = rgb.permute(0, 3, 1, 2)
    shift = (torch.rand(n, 2, device=x.device) * 2 - 1) * AUG_SHIFT_PX
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = shift[:, 0] * 2 / w
    theta[:, 1, 2] = shift[:, 1] * 2 / h
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    contrast = torch.empty(n, 1, 1, 1, device=x.device).uniform_(*AUG_CONTRAST)
    gain = torch.empty(n, 3, 1, 1, device=x.device).uniform_(*AUG_CHANNEL_GAIN).repeat(1, c // 3, 1, 1)  # same on every stacked frame
    x = x * contrast * gain + torch.randn_like(x) * AUG_PIXEL_NOISE_STD
    return x.permute(0, 2, 3, 1)


class FrameStack:
    """Last ``k`` images per camera, stacked on the channel axis (oldest first), so the student can see motion.

    Call once per env step with the ``dones`` of the step that produced these images: an env that just reset starts
    with its first image repeated."""

    def __init__(self, k: int):
        self.k = k
        self.buffers: dict[str, list[torch.Tensor]] | None = None

    def __call__(self, images: dict[str, torch.Tensor], dones: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.k == 1:
            return images
        if self.buffers is None:
            self.buffers = {name: [img.clone() for _ in range(self.k)] for name, img in images.items()}
        else:
            for name, img in images.items():
                buf = self.buffers[name]
                buf.pop(0)
                buf.append(img.clone())
                if dones is not None and dones.any():
                    for old in buf[:-1]:
                        old[dones] = img[dones]
        return {name: torch.cat(buf, dim=-1) for name, buf in self.buffers.items()}


class CameraEncoder(nn.Module):
    """RGB CNN -> feature vector, for one camera. ``image_shape`` is (H, W, channels); channels is 3 x stacked frames."""

    def __init__(self, image_shape: tuple[int, int, int], cnn_feat_dim: int):
        super().__init__()
        h, w, c = image_shape
        self.in_channels = c
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.cnn(torch.zeros(1, c, h, w)).shape[-1]
        self.fc = nn.Sequential(nn.Linear(flat_dim, cnn_feat_dim), nn.ReLU(inplace=True))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[-1] == 4 and self.in_channels == 3:
            rgb = rgb[..., :3]
        x = rgb.permute(0, 3, 1, 2).contiguous() if rgb.shape[-1] == self.in_channels else rgb
        return self.fc(self.cnn(x))


class StudentVisionPolicy(nn.Module):
    """One CNN per camera + proprio MLP -> continuous joint actions."""

    def __init__(
        self,
        proprio_dim: int,
        num_actions: int,
        image_shapes: dict[str, tuple[int, int, int]],
        cnn_feat_dim: int = 128,
        mlp_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128, 64]

        self.image_shapes = dict(image_shapes)
        self.encoders = nn.ModuleDict({name: CameraEncoder(shape, cnn_feat_dim) for name, shape in image_shapes.items()})

        layers: list[nn.Module] = []
        in_dim = cnn_feat_dim * len(image_shapes) + proprio_dim
        for hidden in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ELU(inplace=True))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_actions))
        self.mlp = nn.Sequential(*layers)

    def forward(self, rgbs: dict[str, torch.Tensor], proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgbs: camera name -> (N, H, W, C) float image (already normalized by the env obs term).
            proprio: (N, D) student proprio vector.
        """
        feats = [self.encoders[name](rgbs[name]) for name in self.image_shapes]
        return self.mlp(torch.cat(feats + [proprio], dim=-1))
