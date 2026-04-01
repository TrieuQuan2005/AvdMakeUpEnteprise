import os
import random
import torch
from PIL import Image
import torchvision.transforms as T


class AttackDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        max_persons=None,
        max_imgs_per_person=None
    ):
        self.face_root = os.path.join(root, "faces")
        self.emb_root = os.path.join(root, "cache", "embeddings")
        self.mask_root = os.path.join(root, "cache", "masks")

        # ===== load persons =====
        persons = [
            p for p in os.listdir(self.face_root)
            if os.path.isdir(os.path.join(self.face_root, p))
        ]

        # ===== limit persons =====
        if max_persons is not None:
            persons = random.sample(persons, min(max_persons, len(persons)))

        self.persons = persons

        # ===== preload image list =====
        self.data = []
        for p in self.persons:
            folder = os.path.join(self.face_root, p)
            imgs = os.listdir(folder)

            if max_imgs_per_person is not None:
                imgs = random.sample(imgs, min(max_imgs_per_person, len(imgs)))

            for img in imgs:
                self.data.append((p, img))

        print(f"[DATASET] Persons: {len(self.persons)}")
        print(f"[DATASET] Images: {len(self.data)}")

        # ===== transforms =====
        self.img_transform = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor()
        ])

        self.mask_transform = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            try:
                # ===== attacker =====
                attacker_id, attacker_img_name = self.data[idx]
                attacker_path = os.path.join(
                    self.face_root, attacker_id, attacker_img_name
                )

                attacker_img = Image.open(attacker_path).convert("RGB")

                # ===== victim =====
                victim_id = random.choice(self.persons)
                while victim_id == attacker_id:
                    victim_id = random.choice(self.persons)

                victim_emb_path = os.path.join(
                    self.emb_root, f"{victim_id}.pt"
                )

                if not os.path.exists(victim_emb_path):
                    idx = random.randint(0, len(self.data) - 1)
                    continue

                victim_emb = torch.load(victim_emb_path, weights_only=True)

                # ===== mask =====
                mask_name = attacker_img_name.replace(".jpg", "_mask.png")
                mask_path = os.path.join(
                    self.mask_root, attacker_id, mask_name
                )

                if not os.path.exists(mask_path):
                    idx = random.randint(0, len(self.data) - 1)
                    continue

                mask = Image.open(mask_path).convert("L")

                # ===== transform =====
                attacker_img = self.img_transform(attacker_img) * 255.0
                mask = self.mask_transform(mask)
                mask = (mask > 0.5).float()

                return attacker_img, mask, victim_emb

            except Exception:
                idx = random.randint(0, len(self.data) - 1)
                continue