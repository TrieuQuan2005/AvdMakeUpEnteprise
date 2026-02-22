import os
from PIL import Image
from torch.utils.data import Dataset


class MakeupStyleDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.samples = []

        makeup_dir = os.path.join(root_dir, "makeup")
        non_makeup_dir = os.path.join(root_dir, "non-makeup")

        for fname in os.listdir(makeup_dir):
            self.samples.append({
                "path": os.path.join(makeup_dir, fname),
                "is_makeup": 1
            })

        for fname in os.listdir(non_makeup_dir):
            self.samples.append({
                "path": os.path.join(non_makeup_dir, fname),
                "is_makeup": 0
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "is_makeup": sample["is_makeup"]
        }
