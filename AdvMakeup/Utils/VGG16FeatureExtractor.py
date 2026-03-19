from collections import namedtuple

import nn
from torchvision import models

from AdvMakeup.Utils.utils import get_device


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, device=None):
        super().__init__()
        device = device or get_device()

        vgg = models.vgg16(pretrained=pretrained).features

        self.slice1 = nn.Sequential(*vgg[:4])
        self.slice2 = nn.Sequential(*vgg[4:9])
        self.slice3 = nn.Sequential(*vgg[9:16])
        self.slice4 = nn.Sequential(*vgg[16:23])

        for p in self.parameters():
            p.requires_grad = False

        self.to(device)

    def forward(self, x):
        h = self.slice1(x)
        relu1_2 = h

        h = self.slice2(h)
        relu2_2 = h

        h = self.slice3(h)
        relu3_3 = h

        h = self.slice4(h)
        relu4_3 = h

        VggOutputs = namedtuple("VggOutputs",["relu1_2", "relu2_2", "relu3_3", "relu4_3"])

        return VggOutputs(relu1_2, relu2_2, relu3_3, relu4_3)