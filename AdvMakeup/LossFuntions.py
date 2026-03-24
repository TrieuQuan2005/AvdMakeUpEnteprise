import torch
import torch.nn.functional as F

def cosine_loss(a, b):
    return 1 - F.cosine_similarity(a, b, dim=1).mean()

def region_loss(x_adv, x, mask):
    return ((1 - mask) * (x_adv - x)).abs().mean()

def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, target)