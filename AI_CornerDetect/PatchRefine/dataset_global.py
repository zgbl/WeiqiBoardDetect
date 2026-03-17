import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import yaml
import random

class GlobalBoardDataset(Dataset):
    def __init__(self, root_dirs, img_size=224, heatmap_size=56, is_train=True):
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.is_train = is_train
        self.samples = []
        
        for root in root_dirs:
            root = Path(root)
            if not root.exists(): continue
            for session in root.iterdir():
                if not session.is_dir(): continue
                ext_yml = session / "board_extractor_state.yml"
                if not ext_yml.exists(): continue
                
                with open(ext_yml, 'r') as f:
                    data = yaml.safe_load(f)
                if 'pts_clicks' not in data: continue
                pts = np.array(data['pts_clicks'], dtype=np.float32)
                
                # 排序点 TL, TR, BR, BL
                pts = self.sort_pts_clockwise(pts)
                
                for img_path in session.glob("*.png"):
                    self.samples.append({"img": img_path, "pts": pts})
        
        print(f"Dataset Loaded: {len(self.samples)} images. Augment={is_train}")

    def sort_pts_clockwise(self, pts):
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]
        sums = pts[:, 0] + pts[:, 1]
        tl_idx = np.argmin(sums)
        return np.roll(pts, -tl_idx, axis=0)

    def augment(self, img, pts):
        h, w = img.shape[:2]
        # 1. 随机旋转 (-30 到 30)
        angle = random.uniform(-30, 30)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        
        pts_ones = np.ones((4, 3))
        pts_ones[:, :2] = pts
        pts = M.dot(pts_ones.T).T
        
        # 2. 随机缩放与平移 (模拟拍摄距离和位置)
        scale = random.uniform(0.7, 1.1)
        tx = random.uniform(-w*0.05, w*0.05)
        ty = random.uniform(-h*0.05, h*0.05)
        M_st = np.float32([[scale, 0, tx], [0, scale, ty]])
        img = cv2.warpAffine(img, M_st, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        
        pts_ones[:, :2] = pts
        pts = M_st.dot(pts_ones.T).T
        
        # 3. 色彩增强
        if random.random() > 0.5:
            brightness = random.uniform(0.7, 1.3)
            img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
            
        return img, pts

    def create_heatmap(self, pts, h, w):
        heatmaps = np.zeros((4, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        sigma = 2.0 # 稍微增大半径，让 Loss 更容易下降
        for i, (px, py) in enumerate(pts):
            tx, ty = (px / w * self.heatmap_size), (py / h * self.heatmap_size)
            if 0 <= tx < self.heatmap_size and 0 <= ty < self.heatmap_size:
                grid_y, grid_x = np.mgrid[0:self.heatmap_size, 0:self.heatmap_size]
                dist_sq = (grid_x - tx)**2 + (grid_y - ty)**2
                heatmaps[i] = np.exp(-dist_sq / (2 * sigma**2))
        return heatmaps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(str(sample['img']))
        pts = sample['pts'].copy()
        
        if self.is_train:
            img, pts = self.augment(img, pts)
        
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        heatmaps = self.create_heatmap(pts, h, w)
        
        # 归一化: 严格按照 ImageNet 标准
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor, torch.tensor(heatmaps)
