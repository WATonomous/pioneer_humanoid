"""TrackNet network from TrackNetV3 (https://github.com/qaz812345/TrackNetV3, MIT License,
Copyright (c) 2024 qaz812345). Copied unchanged so our fine-tuned weights load into it.

Input: 8 frames plus their median background, RGB, 512x288 -> 27 channels.
Output: one 512x288 heatmap per input frame (8); bright spots are likely shuttles.
"""
import torch
import torch.nn as nn

WIDTH, HEIGHT = 512, 288
SEQ_LEN = 8


class Conv2DBlock(nn.Module):
    """Conv2D + BN + ReLU"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNet(nn.Module):
    """U-Net: three downsampling blocks, a bottleneck, three upsampling blocks with skips."""

    def __init__(self, in_dim=(SEQ_LEN + 1) * 3, out_dim=SEQ_LEN):
        super().__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pool, up = nn.MaxPool2d(2), nn.Upsample(scale_factor=2)
        x1 = self.down_block_1(x)                                  # (N,  64, 288, 512)
        x2 = self.down_block_2(pool(x1))                           # (N, 128, 144, 256)
        x3 = self.down_block_3(pool(x2))                           # (N, 256,  72, 128)
        x = self.bottleneck(pool(x3))                              # (N, 512,  36,  64)
        x = self.up_block_1(torch.cat([up(x), x3], dim=1))         # (N, 256,  72, 128)
        x = self.up_block_2(torch.cat([up(x), x2], dim=1))         # (N, 128, 144, 256)
        x = self.up_block_3(torch.cat([up(x), x1], dim=1))         # (N,  64, 288, 512)
        return self.sigmoid(self.predictor(x))                     # (N,   8, 288, 512)
