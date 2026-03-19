import nn
import torch

from AdvMakeup.Utils.utils import get_device

#Chuẩn hóa màu
class MeanShift(nn.Conv2d):
    def __init__(self, device=None):
        device = device or get_device()

        super().__init__(3, 3, kernel_size=1)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        std = torch.tensor(rgb_std).to(device)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1).to(device) / std.view(3, 1, 1, 1)
        self.bias.data = -torch.tensor(rgb_mean).to(device) / std

        for p in self.parameters():
            p.requires_grad = False

        self.to(device)