import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2


class MakeupGANDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.non_makeup_dir = os.path.join(root_dir, "non-makeup")
        self.makeup_dir = os.path.join(root_dir, "makeup")
        self.style_dir = os.path.join(root_dir, "style")
        self.mask_dir = os.path.join(root_dir, "mask")
        self.transform = transform

        self.ids = sorted([
            f.replace(".png", "")
            for f in os.listdir(self.non_makeup_dir)
            if f.endswith(".png")
        ])

        assert len(self.ids) > 0, "Dataset empty!"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        non_makeup = Image.open(
            os.path.join(self.non_makeup_dir, f"{id_}.png")
        ).convert("RGB")

        makeup = Image.open(
            os.path.join(self.makeup_dir, f"{id_}.png")
        ).convert("RGB")

        style = torch.load(
            os.path.join(self.style_dir, f"{id_}.pt")
        ).float().view(-1)

        mask = cv2.imread(
            os.path.join(self.mask_dir, f"{id_}.png"),
            cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.resize(mask, (128, 128))
        mask = torch.tensor(mask / 255.0).float().unsqueeze(0)

        if self.transform:
            non_makeup = self.transform(non_makeup)
            makeup = self.transform(makeup)

        return {
            "non_makeup": non_makeup,
            "makeup": makeup,
            "style": style,
            "mask": mask
        }
