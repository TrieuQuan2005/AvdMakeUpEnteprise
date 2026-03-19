import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class FiLM(nn.Module):
    def __init__(self, emb_dim, num_features):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, num_features)
        self.beta = nn.Linear(emb_dim, num_features)

    def forward(self, x, emb):
        gamma = self.gamma(emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(emb).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta
# BUILDING BLOCK
class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, norm='None', sn=False):
        super().__init__()

        layers = []
        layers.append(nn.ReflectionPad2d(padding))

        conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding=0)

        if sn:
            conv = nn.utils.spectral_norm(conv)

        layers.append(conv)

        if norm == 'Instance':
            layers.append(nn.InstanceNorm2d(outplanes, affine=False))

        layers.append(nn.LeakyReLU(inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, 2, 2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.film1 = FiLM(emb_dim, 128)
        self.film2 = FiLM(emb_dim, 64)

        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, mask, target_emb):
        x_in = torch.cat([x, mask], dim=1)

        x1 = self.enc1(x_in)
        x2 = self.enc2(x1)

        x2 = self.film1(x2, target_emb)

        d1 = self.dec1(x2)

        d1_up = F.interpolate(d1, scale_factor=2)
        d2 = self.dec2(torch.cat([d1_up, x1], dim=1))

        d2_up = F.interpolate(d2, scale_factor=2)
        d3 = self.dec3(torch.cat([d2_up, x1], dim=1))

        d3 = self.film2(d3, target_emb)

        epsilon = 10.0
        perturb = torch.tanh(self.out_conv(d3)) * epsilon

        return perturb * mask


# DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, input_dim=3, ndf=64):
        super().__init__()

        self.model = nn.Sequential(
            LeakyReLUConv2d(input_dim, ndf * 2, 3, 2, 1, norm='Instance'),
            LeakyReLUConv2d(ndf * 2, ndf * 2, 3, 2, 1, norm='Instance'),
            LeakyReLUConv2d(ndf * 2, ndf * 2, 3, 2, 1, norm='Instance'),
            LeakyReLUConv2d(ndf * 2, ndf * 2, 1, 1, 0),
            nn.Conv2d(ndf * 2, 1, 1)
        )

    def forward(self, x):
        return self.model(x).view(-1)


# GAN NETWORK WRAPPER
class GanNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        self._init_weights()

    def forward(self, x):
        fake = self.generator(x)
        pred = self.discriminator(fake)
        return fake, pred

    def generate(self, x):
        return self.generator(x)

    def discriminate(self, x):
        return self.discriminator(x)

    def _init_weights(self, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__

            if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data)
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)

                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)