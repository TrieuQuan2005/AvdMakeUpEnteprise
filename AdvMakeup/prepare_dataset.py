import os
import cv2
import torch
import shutil
from PIL import Image
import mediapipe as mp
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

from AdvMakeup.Models.FaceReconizationModels.FaceNet.FaceNetWrapper import FaceNetWrapper
from EyeDetect.Services.EyeDetectorService import EyeDetectorService

# ====== CONFIG ======
INPUT_FACE_DIR = "faces"
OUTPUT_DIR = "dataset"

FACE_OUT = os.path.join(OUTPUT_DIR, "faces")
EMB_OUT = os.path.join(OUTPUT_DIR, "cache", "embeddings")
MASK_OUT = os.path.join(OUTPUT_DIR, "cache", "masks")

os.makedirs(FACE_OUT, exist_ok=True)
os.makedirs(EMB_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),                 # [0,1]
    T.Lambda(lambda x: x * 255.0) # -> [0,255]
])

facenet = FaceNetWrapper().to(device).eval()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

eyeDetector = EyeDetectorService(face_mesh)

def get_embedding(img_tensor):
    with torch.no_grad():
        emb = facenet.get_embedding(img_tensor)

        if emb is None:
            raise ValueError("Embedding is None")

        if torch.isnan(emb).any():
            raise ValueError("Embedding contains NaN")

        if emb.shape[-1] != 512:
            raise ValueError(f"Invalid embedding shape: {emb.shape}")

        return emb


def paste_mask(full_mask, small_mask, box):
    x1, y1 = int(box.x1), int(box.y1)

    H, W = full_mask.shape
    h, w = small_mask.shape

    x2 = min(x1 + w, W)
    y2 = min(y1 + h, H)

    full_mask[y1:y2, x1:x2] = np.maximum(
        full_mask[y1:y2, x1:x2],
        small_mask[:y2-y1, :x2-x1]
    )

    return full_mask


def get_eye_mask(img_pil):
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    result = eyeDetector.detect(img_bgr)
    if result is None:
        return None

    left_eye, right_eye = result

    H, W = img_bgr.shape[:2]
    full_mask = np.zeros((H, W), dtype=np.float32)

    full_mask = paste_mask(full_mask, left_eye.mask, left_eye.box)
    full_mask = paste_mask(full_mask, right_eye.mask, right_eye.box)

    full_mask = cv2.GaussianBlur(full_mask, (15, 15), 0)

    # normalize mask
    if full_mask.max() > 0:
        full_mask = full_mask / full_mask.max()

    return np.clip(full_mask, 0, 1)


def process_faces():
    persons = sorted(os.listdir(INPUT_FACE_DIR))

    for person in tqdm(persons, desc="Processing persons"):
        person_path = os.path.join(INPUT_FACE_DIR, person)

        if not os.path.isdir(person_path):
            continue

        out_face_person = os.path.join(FACE_OUT, person)
        out_mask_person = os.path.join(MASK_OUT, person)

        os.makedirs(out_face_person, exist_ok=True)
        os.makedirs(out_mask_person, exist_ok=True)

        embeddings = []
        valid_count = 0

        images = sorted(os.listdir(person_path))

        for img_name in images:
            img_path = os.path.join(person_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Cannot open {img_path}: {e}")
                continue

            shutil.copy(img_path, os.path.join(out_face_person, img_name))

            try:
                img_tensor = transform(img).unsqueeze(0).to(device)
                img_tensor = torch.clamp(img_tensor, 0, 255)

                emb = get_embedding(img_tensor)
                embeddings.append(emb.squeeze(0).cpu())

            except Exception as e:
                print(f"[ERROR] Embedding failed: {img_path} | {e}")
                continue

            try:
                mask = get_eye_mask(img)

                if mask is not None:
                    mask_img = Image.fromarray((mask * 255).astype("uint8"))
                    mask_name = os.path.splitext(img_name)[0] + "_mask.png"
                    mask_img.save(os.path.join(out_mask_person, mask_name))

            except Exception as e:
                print(f"[ERROR] Mask failed: {img_path} | {e}")

            valid_count += 1

        if len(embeddings) > 0:
            embeddings = torch.stack(embeddings)

            mean_emb = embeddings.mean(dim=0)

            torch.save(mean_emb, os.path.join(EMB_OUT, f"{person}.pt"))
            torch.save(embeddings, os.path.join(EMB_OUT, f"{person}_all.pt"))

        print(f"[INFO] {person}: {valid_count} images processed")


if __name__ == "__main__":
    process_faces()