import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class GomradePatchDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, transform=None, is_train=True):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        for corner_type in ['TL', 'TR', 'BR', 'BL']:
            folder = self.root_dir / corner_type
            if folder.exists():
                for img_path in folder.glob("*.png"):
                    self.samples.append({"path": img_path, "type": corner_type})
        print(f"Loaded {len(self.samples)} patches from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(str(sample["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target_x, target_y = self.patch_size / 2, self.patch_size / 2
        if self.is_train:
            max_shift = 15
            shift_x = np.random.randint(-max_shift, max_shift)
            shift_y = np.random.randint(-max_shift, max_shift)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (self.patch_size, self.patch_size))
            target_x += shift_x
            target_y += shift_y
        target = np.array([target_x / self.patch_size, target_y / self.patch_size], dtype=np.float32)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.tensor(target)
