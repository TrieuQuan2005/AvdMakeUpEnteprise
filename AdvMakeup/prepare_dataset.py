import os
import torch
import shutil
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# ====== CONFIG ======
INPUT_FACE_DIR = "faces"
OUTPUT_DIR = "dataset"

FACE_OUT = os.path.join(OUTPUT_DIR, "faces")
MAKEUP_OUT = os.path.join(OUTPUT_DIR, "makeup")
EMB_OUT = os.path.join(OUTPUT_DIR, "cache", "embeddings")
MASK_OUT = os.path.join(OUTPUT_DIR, "cache", "masks")

os.makedirs(FACE_OUT, exist_ok=True)
os.makedirs(MAKEUP_OUT, exist_ok=True)
os.makedirs(EMB_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

# ====== TRANSFORM ======
transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor() * 255.0  # giữ range [0,255] cho FaceNet wrapper
])

# ====== YOUR FUNCTIONS (plug vào đây) ======
def get_embedding(img_tensor):
    """
    img_tensor: (1,3,H,W)
    return: (1,512)
    """
    raise NotImplementedError

def get_eye_mask(img_pil):
    """
    img_pil: PIL image
    return: numpy array or PIL (H,W) with 0/1
    """
    raise NotImplementedError

# ====== MAIN ======
def process_faces():
    persons = os.listdir(INPUT_FACE_DIR)

    for person in tqdm(persons, desc="Processing persons"):
        person_path = os.path.join(INPUT_FACE_DIR, person)

        if not os.path.isdir(person_path):
            continue

        # create output folder
        out_person_path = os.path.join(FACE_OUT, person)
        os.makedirs(out_person_path, exist_ok=True)

        embeddings = []

        images = os.listdir(person_path)

        for img_name in images:
            img_path = os.path.join(person_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
            except:
                continue

            # ===== Save face (copy) =====
            shutil.copy(img_path, os.path.join(out_person_path, img_name))

            # ===== Compute embedding =====
            img_tensor = transform(img).unsqueeze(0)  # (1,3,H,W)
            emb = get_embedding(img_tensor)  # (1,512)
            embeddings.append(emb.squeeze(0).cpu())

            # ===== Compute mask =====
            mask = get_eye_mask(img)

            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            mask_img = Image.fromarray((mask * 255).astype("uint8"))

            mask_name = img_name.replace(".jpg", "_mask.png")
            mask_path = os.path.join(MASK_OUT, mask_name)

            mask_img.save(mask_path)

        # ===== Save embedding (mean embedding) =====
        if len(embeddings) > 0:
            embeddings = torch.stack(embeddings)
            mean_emb = embeddings.mean(dim=0)

            save_path = os.path.join(EMB_OUT, f"{person}.pt")
            torch.save(mean_emb, save_path)


if __name__ == "__main__":
    process_faces()