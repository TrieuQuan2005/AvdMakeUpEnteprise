import os
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from AdvMakeup.AttackDataset import AttackDataset
from AdvMakeup.LossFuntions import gan_loss, region_loss, cosine_loss
from AdvMakeup.Models.FaceReconizationModels.FaceNet.FaceNetWrapper import FaceNetWrapper
from AdvMakeup.Models.GanNetwork import GanNetwork
from AdvMakeup.Utils.VGG16FeatureExtractor import VGG16FeatureExtractor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True

SAVE_DIR = "./checkpoints"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "G_best.pth")
LAST_MODEL_PATH = os.path.join(SAVE_DIR, "G_last.pth")


# =============================
# VGG preprocess
# =============================
def preprocess_vgg(x):
    x = x / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x - mean) / std


# =============================
# Load victim embedding
# =============================
def load_victim_embedding(img_path, fr_model):

    transform = T.Compose([
        T.Resize((160,160)),
        T.ToTensor()
    ])

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE) * 255.0

    with torch.no_grad():
        emb = fr_model(img)

    return emb


# =============================
# TRAIN
# =============================
def train(
    gan,
    fr_model,
    vgg_model,
    dataset,
    victim_emb,
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

    for p in fr_model.parameters():
        p.requires_grad = False

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.999))

    best_cos = -1.0

    for epoch in range(epochs):

        pbar = tqdm(loader, desc=f"Epoch {epoch}")

        epoch_cos = 0
        step_count = 0

        for attacker_img, mask, _ in pbar:

            attacker_img = attacker_img.to(DEVICE)
            mask = mask.to(DEVICE)

            victim_batch = victim_emb.repeat(
                attacker_img.size(0), 1
            )

            # =========================
            # Generator
            # =========================

            x_adv = G(attacker_img, mask, victim_batch)
            x_adv = torch.clamp(x_adv, 0, 255)

            emb_adv = fr_model(x_adv)

            loss_id = cosine_loss(
                emb_adv,
                victim_batch
            )

            # perceptual
            feat_adv = vgg_model(preprocess_vgg(x_adv))
            feat_real = vgg_model(preprocess_vgg(attacker_img))

            loss_perc = sum(
                (fa - fr).abs().mean()
                for fa, fr in zip(feat_adv, feat_real)
            )

            # smooth
            loss_smooth = (
                (x_adv[:, :, :-1, :] - x_adv[:, :, 1:, :]).abs().mean()
                +
                (x_adv[:, :, :, :-1] - x_adv[:, :, :, 1:]).abs().mean()
            )

            # region
            loss_reg = region_loss(
                x_adv,
                attacker_img,
                mask
            )

            # GAN
            pred_fake = D(x_adv)
            loss_gan_G = gan_loss(pred_fake, True)

            # TOTAL
            loss_G = (
                25 * loss_id +
                0.3 * loss_perc +
                0.1 * loss_smooth +
                5 * loss_reg +
                1 * loss_gan_G
            )

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # =========================
            # Train D
            # =========================

            pred_real = D(attacker_img)
            pred_fake = D(x_adv.detach())

            loss_D = (
                gan_loss(pred_real, True) +
                gan_loss(pred_fake, False)
            )

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            cos_sim = torch.nn.functional.cosine_similarity(
                emb_adv,
                victim_batch
            ).mean().item()

            epoch_cos += cos_sim
            step_count += 1

            if step_count % 100 == 0:

                print("\n========= DEBUG =========")
                print("adv diff:",
                      (x_adv - attacker_img).abs().mean().item())
                print("cosine:", cos_sim)
                print("=========================\n")

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


# =============================
# MAIN
# =============================

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    dataset = AttackDataset(
        os.path.join(BASE_DIR, "dataset"),
        max_imgs_per_person=200
    )

    gan = GanNetwork(DEVICE)

    fr_model = FaceNetWrapper(device=DEVICE).eval()
    vgg_model = VGG16FeatureExtractor().to(DEVICE).eval()

    victim_path = os.path.join(BASE_DIR, "Trieuquan.jpg")

    victim_emb = load_victim_embedding(
        victim_path,
        fr_model
    )

    train(
        gan,
        fr_model,
        vgg_model,
        dataset,
        victim_emb,
        epochs=6,
        batch_size=4,
        num_workers=2
    )