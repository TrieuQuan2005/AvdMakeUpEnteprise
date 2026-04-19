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
        max_imgs_per_person=None,
        max_makeups=None
    ):
        self.face_root = os.path.join(root, "faces")
        self.emb_root = os.path.join(root, "cache", "embeddings")
        self.mask_root = os.path.join(root, "cache", "masks")
        self.makeup_root = os.path.join(root, "makeups")

        # ===== load persons =====
        persons = [
            p for p in os.listdir(self.face_root)
            if os.path.isdir(os.path.join(self.face_root, p))
        ]

        if max_persons is not None:
            persons = random.sample(persons, min(max_persons, len(persons)))

        self.persons = persons

        # ===== preload attacker images =====
        self.data = []
        for p in self.persons:
            folder = os.path.join(self.face_root, p)
            imgs = os.listdir(folder)

            if max_imgs_per_person is not None:
                imgs = random.sample(imgs, min(max_imgs_per_person, len(imgs)))

            for img in imgs:
                self.data.append((p, img))

        # ===== load makeup images =====
        self.makeups = [
            f for f in os.listdir(self.makeup_root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if max_makeups is not None:
            self.makeups = random.sample(
                self.makeups,
                min(max_makeups, len(self.makeups))
            )

        print(f"[DATASET] Persons: {len(self.persons)}")
        print(f"[DATASET] Images: {len(self.data)}")
        print(f"[DATASET] Makeups: {len(self.makeups)}")

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
                # attacker image
                attacker_id, attacker_img_name = self.data[idx]

                attacker_path = os.path.join(
                    self.face_root,
                    attacker_id,
                    attacker_img_name
                )

                attacker_img = Image.open(attacker_path).convert("RGB")

                # attacker embedding
                attacker_emb_path = os.path.join(
                    self.emb_root,
                    f"{attacker_id}.pt"
                )

                if not os.path.exists(attacker_emb_path):
                    idx = random.randint(0, len(self.data) - 1)
                    continue

                attacker_emb = torch.load(
                    attacker_emb_path,
                    weights_only=True
                )

                # mask
                base = os.path.splitext(attacker_img_name)[0]
                mask_name = base + "_mask.png"

                mask_path = os.path.join(
                    self.mask_root,
                    attacker_id,
                    mask_name
                )

                if not os.path.exists(mask_path):
                    idx = random.randint(0, len(self.data) - 1)
                    continue

                mask = Image.open(mask_path).convert("L")

                # makeup
                makeup_name = random.choice(self.makeups)

                makeup_path = os.path.join(
                    self.makeup_root,
                    makeup_name
                )

                makeup_img = Image.open(makeup_path).convert("RGB")

                attacker_img = self.img_transform(attacker_img)
                makeup_img = self.img_transform(makeup_img)

                mask = self.mask_transform(mask)
                mask = (mask > 0.5).float()

                return attacker_img, mask, attacker_emb, makeup_img

            except Exception:
                idx = random.randint(0, len(self.data) - 1)
                continue