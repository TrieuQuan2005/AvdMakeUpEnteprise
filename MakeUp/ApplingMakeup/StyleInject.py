import torch
import torch.nn as nn


# =============================
# Style Injection (AdaIN-lite)
# =============================
class StyleInject(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, style):
        gamma, beta = self.fc(style).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


# =============================
# Generator
# =============================
class MakeupGenerator(nn.Module):
    def __init__(self, style_dim=128):
        super().__init__()

        # input: eye(3) + mask(1) = 4
        self.enc1 = nn.Conv2d(4, 64, 4, 2, 1)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)

        self.style = StyleInject(style_dim, 256)

        self.dec1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        self.act = nn.ReLU(True)
        self.out = nn.Sigmoid()

    def forward(self, eye, mask, style):
        # eye:  [B,3,H,W]
        # mask: [B,1,H,W]
        x = torch.cat([eye, mask], dim=1)

        e1 = self.act(self.enc1(x))
        e2 = self.act(self.enc2(e1))
        e3 = self.act(self.enc3(e2))

        e3 = self.style(e3, style)

        d1 = self.act(self.dec1(e3))
        d2 = self.act(self.dec2(d1))
        out = self.out(self.dec3(d2))

        # chỉ apply makeup trong mask
        out = out * mask + eye * (1 - mask)

        return out


# =============================
# Discriminator (PatchGAN)
# =============================
class MakeupDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # eye(3) + makeup(3) + mask(1) = 7
        self.net = nn.Sequential(
            nn.Conv2d(7, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, eye, makeup, mask):
        x = torch.cat([eye, makeup, mask], dim=1)
        return self.net(x)
