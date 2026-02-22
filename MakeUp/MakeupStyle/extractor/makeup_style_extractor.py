import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class StyleEncoder(nn.Module):
    def __init__(self, style_dim: int = 128):
        super().__init__()
        self.style_dim = style_dim

        self.encoder = nn.Sequential(
            ConvBlock(3, 32, stride=2),    # H/2
            ConvBlock(32, 64, stride=2),   # H/4
            ConvBlock(64, 128, stride=2),  # H/8
            ConvBlock(128, 256, stride=2), # H/16
            ConvBlock(256, 256, stride=1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, style_dim),
            nn.LayerNorm(style_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)           # (B, 256, h, w)
        feat = self.global_pool(feat)    # (B, 256, 1, 1)
        feat = feat.view(feat.size(0), -1)
        style_latent = self.fc(feat)
        return style_latent
