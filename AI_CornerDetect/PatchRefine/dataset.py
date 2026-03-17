import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class GomradePatchDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, transform=None, is_train=True, sessions=None, use_synthetic=False):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        
        # 类别映射 TL:1, TR:2, BR:3, BL:4, BG:0
        corner_map = {'TL':1, 'TR':2, 'BR':3, 'BL':4}

        # 1. 加载真实数据 (暂时默认为正样本，由于没有负样本标注，我们只在训练集加合成负样本)
        if sessions is not None or not use_synthetic:
            for corner_type in ['TL', 'TR', 'BR', 'BL']:
                folder = self.root_dir / corner_type
                if not folder.exists(): continue
                
                files = list(folder.glob("*.png"))
                for img_path in files:
                    session_name = img_path.name.split('_')[0]
                    if sessions is not None and session_name not in sessions:
                        continue
                    self.samples.append({
                        "path": img_path, 
                        "type": "real", 
                        "class_id": corner_map[corner_type],
                        "gt": (64, 64)
                    })

        # 2. 加载合成数据 (包含正负样本)
        if use_synthetic:
            syn_dir = self.root_dir / "Synthetic"
            label_file = syn_dir / "labels.txt"
            if label_file.exists():
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 4:
                            img_name, cid, x, y = parts
                            self.samples.append({
                                "path": syn_dir / img_name,
                                "type": "synthetic",
                                "class_id": int(cid),
                                "gt": (float(x), float(y))
                            })
        
        print(f"Dataset ready: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def rotate_image_and_point(self, image, point, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        point_tri = np.array([point[0], point[1], 1])
        new_point = M.dot(point_tri)
        return rotated_img, new_point

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(str(sample["path"]))
        if img is None: # 防御性编程
            return self.__getitem__((idx + 1) % len(self.samples))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target_x, target_y = sample["gt"]
        class_id = sample["class_id"]
        
        if self.is_train:
            # 真实数据增加随机位移
            if sample["type"] == "real":
                max_shift = 20
                sx = np.random.randint(-max_shift, max_shift)
                sy = np.random.randint(-max_shift, max_shift)
                M_shift = np.float32([[1, 0, sx], [0, 1, sy]])
                img = cv2.warpAffine(img, M_shift, (self.patch_size, self.patch_size), borderMode=cv2.BORDER_REFLECT_101)
                target_x += sx
                target_y += sy
            
            # 合成负样本不旋转(也没意义)，正样本旋转
            if class_id > 0:
                angle = np.random.uniform(0, 360)
                img, (target_x, target_y) = self.rotate_image_and_point(img, (target_x, target_y), angle)

        # 归一化坐标
        target_coords = np.array([target_x / self.patch_size, target_y / self.patch_size], dtype=np.float32)
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
        return img, torch.tensor(class_id, dtype=torch.long), torch.tensor(target_coords, dtype=torch.float32)
