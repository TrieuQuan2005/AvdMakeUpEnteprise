import torch
import torch.nn as nn
import torch.nn.functional as f

"""class EmbeddingAttackLoss(nn.Module):
    def __init__(self, margin=0.4, lambda_self=0.0):
        super().__init__()
        self.margin = margin
        self.lambda_self = lambda_self

    def forward(self, e_adv, e_target, e_you):
        cos_adv_target = f.cosine_similarity(e_adv, e_target, dim=1)
        cos_adv_you    = f.cosine_similarity(e_adv, e_you, dim=1)

        loss_target = torch.clamp(self.margin - cos_adv_target, min=0)

        loss_self = 1.0 - cos_adv_you

        loss = loss_target + self.lambda_self * loss_self
        return loss.mean()"""


class EmbeddingAttackLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, e_adv, e_target, e_you):
        cos_t = f.cosine_similarity(e_adv, e_target, dim=1)
        cos_y = f.cosine_similarity(e_adv, e_you, dim=1)

        # Want: cos_t >= cos_y + margin
        loss = torch.clamp(self.margin + cos_y - cos_t, min=0)
        return loss.mean()
