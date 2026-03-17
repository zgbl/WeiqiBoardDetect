import os
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
import numpy as np

class MultiClassPatchDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        self.transform = transform
        self.samples = []
        
        self.class_map = {
            'corner': 0,
            'edge': 1,
            'inner': 2,
            'outer': 3
        }
        
        for class_name, class_id in self.class_map.items():
            folder = self.root_dir / class_name
            if not folder.exists(): 
                print(f"Warning: Folder {folder} not found.")
                continue
            
            files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            for f in files:
                self.samples.append((f, class_id))
        
        print(f"Dataset Loaded: {len(self.samples)} samples. Classes: {self.class_map}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            return self.__getitem__(random.randint(0, len(self.samples)-1))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.is_train:
            # Basic Augmentation
            if random.random() > 0.5:
                img = cv2.flip(img, 1) # Horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 0) # Vertical flip
            
            # Simple rotation (90 deg steps)
            k = random.randint(0, 3)
            img = np.rot90(img, k).copy()
            
            # Brightness
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

        # To Tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Normalization (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        
        return img, torch.tensor(class_id, dtype=torch.long)
