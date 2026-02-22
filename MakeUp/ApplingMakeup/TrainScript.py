import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from MakeUp.ApplingMakeup.StyleInject import MakeupGenerator, MakeupDiscriminator
from MakeUp.ApplingMakeup.trainer import MakeupGANTrainer
from MakeUp.ApplingMakeup.makeup_gan_dataset import MakeupGANDataset


# CONFIG
DATASET_ROOT = "..\\..\\Data\\MakeupGANDataset"
BATCH_SIZE = 8
EPOCHS = 100
IMAGE_SIZE = 128
STYLE_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_G = "makeup_generator.pth"
SAVE_D = "makeup_discriminator.pth"


# Transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# Dataset & Loader
dataset = MakeupGANDataset(
    root_dir=DATASET_ROOT,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True
)


# Models
G = MakeupGenerator(style_dim=STYLE_DIM)
D = MakeupDiscriminator()


# Trainer
trainer = MakeupGANTrainer(
    G=G,
    D=D,
    loader=loader,
    device=DEVICE
)


# Train
if __name__ == "__main__":
    print("====================================")
    print("   Makeup Application GAN Training   ")
    print("====================================")
    print(f"Device       : {DEVICE}")
    print(f"Dataset size : {len(dataset)}")
    print(f"Epochs       : {EPOCHS}")
    print("====================================")

    for epoch in range(EPOCHS):
        trainer.train_epoch()
        print(f"[Epoch {epoch+1}/{EPOCHS}] done")

        # Save checkpoint mỗi 10 epoch
        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), SAVE_G)
            torch.save(D.state_dict(), SAVE_D)
            print("💾 Checkpoint saved")

    torch.save(G.state_dict(), SAVE_G)
    torch.save(D.state_dict(), SAVE_D)
    print("🎉 Training completed")
