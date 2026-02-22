import torch.nn as nn


class StyleEncoder(nn.Module):
    def __init__(self, style_dim: int = 128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 64x64
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, 2, 1),  # 32x32
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1), # 16x16
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, 2, 1),# 8x8
            nn.ReLU(True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, 256, 1, 1)
            nn.Flatten(),              # (B, 256)
            nn.Linear(256, style_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)
        style = self.head(feat)
        return style
