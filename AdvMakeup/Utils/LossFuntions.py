import torch
import torch.nn.functional as F
import torch.nn as nn



# IDENTITY LOSS (FaceNet)
def cosine_loss(a, b):
    return 1 - F.cosine_similarity(a, b, dim=1).mean()


# REGION LOSS (only makeup area)
def region_loss(x_adv, x, mask):
    mask = mask.expand_as(x_adv)
    return ((1 - mask) * (x_adv - x)).abs().mean()


# GAN LOSS
def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, target)


# GRAM MATRIX
def gram_matrix(x):
    B, C, H, W = x.size()
    f = x.view(B, C, H * W)
    return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)


# STYLE LOSS (makeup transfer)
def style_loss(vgg, x_adv, makeup):

    # preprocess for VGG
    x_adv = vgg.preprocess_vgg(x_adv * 255)
    makeup = vgg.preprocess_vgg(makeup * 255)

    feat_adv = vgg(x_adv)
    feat_make = vgg(makeup)

    loss = 0

    for fa, fm in zip(feat_adv, feat_make):
        loss += (gram_matrix(fa) - gram_matrix(fm)).abs().mean()

    return loss


# EDGE LOSS
class LaplacianFilter(nn.Module):
    def __init__(self):
        super().__init__()

        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32)

        kernel = kernel.unsqueeze(0).unsqueeze(0)

        self.register_buffer("kernel", kernel)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1)


def edge_loss(lap, x_adv, makeup):

    x_adv_gray = x_adv.mean(dim=1, keepdim=True)
    makeup_gray = makeup.mean(dim=1, keepdim=True)

    edge_adv = lap(x_adv_gray)
    edge_make = lap(makeup_gray)

    return (edge_adv - edge_make).abs().mean()


# SMOOTH LOSS
def smooth_loss(x):
    return (
        (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean() +
        (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean()
    )



# PERCEPTUAL LOSS (VGG)
def perceptual_loss(vgg, x_adv, x):

    x_adv = vgg.preprocess_vgg(x_adv * 255)
    x = vgg.preprocess_vgg(x * 255)

    feat_adv = vgg(x_adv)
    feat_x = vgg(x)

    loss = 0

    for fa, fx in zip(feat_adv, feat_x):
        loss += (fa - fx).abs().mean()

    return loss



# TOTAL LOSS
def total_loss(
    loss_id,
    loss_perc,
    loss_reg,
    loss_gan,
    loss_style,
    loss_edge,
    loss_smooth,
    w=None
):

    if w is None:
        w = dict(
            id=25,
            perc=0.3,
            reg=5,
            gan=1,
            style=8,
            edge=1.5,
            smooth=0.1
        )

    return (
        w["id"] * loss_id +
        w["perc"] * loss_perc +
        w["reg"] * loss_reg +
        w["gan"] * loss_gan +
        w["style"] * loss_style +
        w["edge"] * loss_edge +
        w["smooth"] * loss_smooth
    )