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
        
        # 1. 明确定义类别，这里绝对不包含 'dirty'
        self.class_map = {
            'corner': 0,
            'edge': 1,
            'inner': 2,
            'outer': 3
        }
        
        print(f"Initializing Dataset. Explicitly EXCLUDING 'dirty' folder.")
        
        for class_name, class_id in self.class_map.items():
            folder = self.root_dir / class_name
            if not folder.exists(): 
                print(f"Warning: Expected folder {folder} not found.")
                continue
            
            # 搜索图片
            files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            for f in files:
                self.samples.append((f, class_id))
        
        # 特别检查：确保 samples 中没有任何路径包含 'dirty'
        original_len = len(self.samples)
        self.samples = [s for s in self.samples if "dirty" not in str(s[0]).lower()]
        if len(self.samples) != original_len:
            print(f"CRITICAL: Filtered out {original_len - len(self.samples)} files that accidentally contained 'dirty' in path.")

        print(f"Dataset Loaded: {len(self.samples)} samples. Classes: {list(self.class_map.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            return self.__getitem__(random.randint(0, len(self.samples)-1))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.is_train:
            # 基础增强：旋转和翻转
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
            
            k = random.randint(0, 3)
            img = np.rot90(img, k).copy()
            
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

        # 归一化/转 Tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        
        return img, torch.tensor(class_id, dtype=torch.long)
