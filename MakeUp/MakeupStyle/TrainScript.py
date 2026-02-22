import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from MakeUp.MakeupStyle.trainer.style_trainer import StyleTrainer
from MakeUp.MakeupStyle.makeup_style_dataset import MakeupStyleDataset


# CONFIG
DATASET_ROOT = "..\\..\\Data\\StyleExtractorDts"
BATCH_SIZE = 16
EPOCHS = 50
STYLE_DIM = 128
IMAGE_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "style_encoder.pth"


# =====================================================
# Transform
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# =====================================================
# Dataset & Loader
# =====================================================
dataset = MakeupStyleDataset(
    root_dir=DATASET_ROOT,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True
)


# Train
if __name__ == "__main__":
    print("===================================")
    print(" Makeup Style Encoder Training ")
    print("===================================")
    print(f"Device      : {DEVICE}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Epochs      : {EPOCHS}")
    print("===================================")

    trainer = StyleTrainer(
        dataloader=dataloader,
        style_dim=STYLE_DIM,
        device=DEVICE
    )

    trainer.train(
        epochs=EPOCHS,
        save_path=SAVE_PATH
    )

    print("🎉 Training completed")
