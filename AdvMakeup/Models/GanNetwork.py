import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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


# GENERATOR (Encoder + Decoder)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 64, 5, stride=2, padding=2)
        self.enc2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.bn_enc2 = nn.BatchNorm2d(128)

        # Decoder
        self.dec1 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_dec1 = nn.BatchNorm2d(128)

        self.dec2 = nn.Conv2d(256, 64, 3, padding=1)
        self.bn_dec2 = nn.BatchNorm2d(64)

        self.dec3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_dec3 = nn.BatchNorm2d(64)

        self.out_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        # ===== Encoder =====
        x1 = F.leaky_relu(self.enc1(x), 0.2)
        x2 = F.leaky_relu(self.bn_enc2(self.enc2(x1)), 0.2)

        # ===== Decoder =====
        d1 = F.relu(self.bn_dec1(self.dec1(x2)))

        d1_up = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)
        d1_cat = torch.cat([d1_up, x2], dim=1)

        d2 = F.relu(self.bn_dec2(self.dec2(d1_cat)))

        d2_up = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d2_cat = torch.cat([d2_up, x1], dim=1)

        d3 = F.relu(self.bn_dec3(self.dec3(d2_cat)))

        out = torch.tanh(self.out_conv(d3))

        return out


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