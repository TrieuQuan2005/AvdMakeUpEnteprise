import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from MakeUp.MakeupStyle.model.style_encoder import StyleEncoder


# =========================
# CONFIG
# =========================
STYLE_ENCODER_PATH = "..\\MakeupStyle\\style_encoder.pth"
MAKEUP_IMAGE_DIR = "..\\..\\Data\\MakeupGANDataset\\makeup"
STYLE_SAVE_DIR = "..\\..\\Data\\MakeupGANDataset\\style"

IMAGE_SIZE = 128
STYLE_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Prepare
# =========================
os.makedirs(STYLE_SAVE_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# =========================
# Load Style Encoder
# =========================
encoder = StyleEncoder(style_dim=STYLE_DIM)
encoder.load_state_dict(
    torch.load(STYLE_ENCODER_PATH, map_location=DEVICE)
)
encoder.to(DEVICE)
encoder.eval()


# =========================
# Generate Style Vectors
# =========================
print("===================================")
print(" Generating Style Vectors ")
print("===================================")
print(f"Device : {DEVICE}")
print(f"Source : {MAKEUP_IMAGE_DIR}")
print(f"Target : {STYLE_SAVE_DIR}")
print("===================================")

with torch.no_grad():
    for fname in tqdm(os.listdir(MAKEUP_IMAGE_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_id = os.path.splitext(fname)[0]

        img = Image.open(
            os.path.join(MAKEUP_IMAGE_DIR, fname)
        ).convert("RGB")

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        style_vec = encoder(img_tensor)   # [1, 128]
        style_vec = style_vec.squeeze(0).cpu()

        torch.save(
            style_vec,
            os.path.join(STYLE_SAVE_DIR, f"{img_id}.pt")
        )

print("🎉 Style extraction completed!")
