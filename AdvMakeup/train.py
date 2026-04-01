import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from AdvMakeup.DatasetManager.AttackDataset import AttackDataset

from AdvMakeup.Utils.LossFuntions import (
    cosine_loss,
    region_loss,
    gan_loss,
    perceptual_loss,
    smooth_loss,
    total_loss
)

from AdvMakeup.Models.FaceNetWrapper import FaceNetWrapper
from AdvMakeup.Models.GanNetwork import GanNetwork
from AdvMakeup.Models.VGG16FeatureExtractor import VGG16FeatureExtractor
from AdvMakeup.Utils.GetDevice import get_device


DEVICE = get_device()
torch.backends.cudnn.benchmark = True

SAVE_DIR = "./checkpoints"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "G_best.pth")
LAST_MODEL_PATH = os.path.join(SAVE_DIR, "G_last.pth")



# TRAIN
def train(
    gan,
    fr_model,
    vgg_model,
    dataset,
    epochs=6,
    batch_size=4,
    num_workers=2
):

    os.makedirs(SAVE_DIR, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    G = gan.generator
    D = gan.discriminator

    G.train()
    D.train()

    fr_model.eval()
    vgg_model.eval()

    for p in fr_model.parameters():
        p.requires_grad = False

    for p in vgg_model.parameters():
        p.requires_grad = False

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    best_cos = -1.0

    for epoch in range(epochs):

        pbar = tqdm(loader, desc=f"Epoch {epoch}")

        epoch_cos = 0
        step_count = 0

        for attacker_img, mask, victim_emb, makeup_img in pbar:

            attacker_img = attacker_img.to(DEVICE)
            mask = mask.to(DEVICE)
            victim_emb = victim_emb.to(DEVICE)
            makeup_img = makeup_img.to(DEVICE)

            
            # Generator forward
            x_adv = G(
                attacker_img,
                mask,
                victim_emb,
                makeup_img
            )

            x_adv = torch.clamp(x_adv, 0, 255)

            
            # Identity loss
            emb_adv = fr_model(x_adv)
            loss_id = cosine_loss(emb_adv, victim_emb)

            
            # Perceptual loss (USE UTILS)
            loss_perc = perceptual_loss(
                vgg_model,
                x_adv,
                attacker_img
            )

            
            # Smooth loss (USE UTILS)
            loss_smooth = smooth_loss(x_adv)

            
            # Region loss
            loss_reg = region_loss(
                x_adv,
                attacker_img,
                mask
            )

            
            # GAN loss
            pred_fake = D(x_adv)
            loss_gan_G = gan_loss(pred_fake, True)

            # TOTAL LOSS (USE UTILS)
            loss_G = total_loss(
                loss_id,
                loss_perc,
                loss_reg,
                loss_gan_G,
                loss_style=0,
                loss_edge=0,
                loss_smooth=loss_smooth
            )

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            
            # Train Discriminator
            pred_real = D(attacker_img)
            pred_fake = D(x_adv.detach())

            loss_D = (
                gan_loss(pred_real, True) +
                gan_loss(pred_fake, False)
            )

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            
            # metrics
            cos_sim = F.cosine_similarity(
                emb_adv,
                victim_emb
            ).mean().item()

            epoch_cos += cos_sim
            step_count += 1

            pbar.set_postfix({
                "G": f"{loss_G.item():.2f}",
                "id": f"{loss_id.item():.2f}",
                "cos": f"{cos_sim:.3f}"
            })

        epoch_cos /= step_count

        print(f"\nEpoch {epoch} avg cosine: {epoch_cos:.4f}")

        torch.save(
            G.state_dict(),
            os.path.join(SAVE_DIR, f"G_epoch_{epoch}.pth")
        )

        if epoch_cos > best_cos:
            best_cos = epoch_cos
            torch.save(G.state_dict(), BEST_MODEL_PATH)

    torch.save(G.state_dict(), LAST_MODEL_PATH)

    print("Training done")
    print("Best cosine:", best_cos)



# MAIN
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    dataset = AttackDataset(
        os.path.join(BASE_DIR, "dataset"),
        max_imgs_per_person=200
    )

    gan = GanNetwork(DEVICE)

    fr_model = FaceNetWrapper(device=DEVICE).eval()
    vgg_model = VGG16FeatureExtractor().to(DEVICE).eval()

    train(
        gan,
        fr_model,
        vgg_model,
        dataset,
        epochs=6,
        batch_size=4,
        num_workers=2
    )