import torch
import torch.nn as nn

from AdvMakeUp.Utils.utils import get_device

#Làm nổi bật viền
class LaplacianFilter(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        device = device or get_device()

        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.conv.weight.data = kernel.to(device)
        self.conv.weight.requires_grad = False

        self.to(device)

    def forward(self, x):
        return self.conv(x)


