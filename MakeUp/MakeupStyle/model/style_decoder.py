import torch.nn as nn

class StyleDecoder(nn.Module):
    def __init__(self, style_dim=128):
        super().__init__()

        self.fc = nn.Linear(style_dim, 256 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64x64
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 128x128
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 8, 8)
        return self.decoder(x)
