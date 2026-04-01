from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# FiLM
class FiLM(nn.Module):
    def __init__(self, emb_dim, num_features):
        super().__init__()

        self.gamma = nn.Linear(emb_dim, num_features)
        self.beta = nn.Linear(emb_dim, num_features)

    def forward(self, x, emb):
        gamma = self.gamma(emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(emb).unsqueeze(-1).unsqueeze(-1)

        return x * (1 + gamma) + beta


# Conv Block
class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, norm='None', sn=False):
        super().__init__()

        layers: List[nn.Module] = [nn.ReflectionPad2d(padding)]

        conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding=0)

        if sn:
            conv = nn.utils.spectral_norm(conv)

        layers.append(conv)

        if norm == 'Instance':
            layers.append(nn.InstanceNorm2d(outplanes))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


# Generator
class Generator(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()

        #Encoder
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

        #ResBlocks
        self.res = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )

        #FiLM
        self.film1 = FiLM(emb_dim, 128)
        self.film2 = FiLM(emb_dim, 64)

        #Decoder
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
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, mask, target_emb):

        # ensure mask 1 channel
        if mask.shape[1] != 1:
            mask = mask[:, :1]

        x_in = torch.cat([x, mask], dim=1)

        #Encoder
        x1 = self.enc1(x_in)
        x2 = self.enc2(x1)

        #Bottleneck
        x2 = self.res(x2)
        x2 = self.film1(x2, target_emb)

        #Decoder
        d1 = self.dec1(x2)
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)

        d2 = self.dec2(torch.cat([d1, x1], dim=1))
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)

        d3 = self.dec3(d2)
        d3 = self.film2(d3, target_emb)

        #Perturbation
        epsilon = 8 / 255.0
        perturb = torch.tanh(self.out_conv(d3)) * epsilon

        adv = x + perturb * mask
        adv = torch.clamp(adv, 0, 1)

        return adv, perturb


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim=3, ndf=64):
        super().__init__()

        self.model = nn.Sequential(
            LeakyReLUConv2d(input_dim, ndf, 4, 2, 1),
            LeakyReLUConv2d(ndf, ndf * 2, 4, 2, 1, norm='Instance'),
            LeakyReLUConv2d(ndf * 2, ndf * 4, 4, 2, 1, norm='Instance'),
            LeakyReLUConv2d(ndf * 4, ndf * 8, 4, 2, 1, norm='Instance'),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.model(x)


# GAN Wrapper
class GanNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        self._init_weights()


    # generate adv
    def generate(self, x, mask, emb):
        adv, perturb = self.generator(x, mask, emb)
        return adv, perturb


    # discriminate
    def discriminate(self, x):
        return self.discriminator(x)

    # weight init
    def _init_weights(self, init_type='kaiming', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__

            if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):

                if init_type == 'normal':
                    init.normal_(m.weight, 0.0, gain)

                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=gain)

                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight)

                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=gain)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

        self.apply(init_func)