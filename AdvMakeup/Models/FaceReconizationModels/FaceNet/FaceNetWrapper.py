import torch
import torch.nn as nn
import torch.nn.functional as F

from AdvMakeup.Models.FaceReconizationModels.FaceNet.FaceNet import InceptionResnetV1


class FaceNetWrapper(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.device = device

        self.model = InceptionResnetV1(
            pretrained='vggface2',
            classify=False
        ).eval().to(device)

        self.input_size = (160, 160)

    def preprocess(self, x):
        # x: [B, 3, H, W], range [0,255]
        x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        x = (x - 127.5) / 128.0
        return x

    def forward(self, x):
        x = self.preprocess(x)
        emb = self.model(x)
        emb = F.normalize(emb, dim=1)
        return emb

    @torch.no_grad()
    def get_embedding(self, x):
        return self.forward(x)