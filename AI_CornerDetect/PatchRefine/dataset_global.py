import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import yaml

class GlobalBoardDataset(Dataset):
    def __init__(self, root_dirs, img_size=224, heatmap_size=56):
        self.img_size = img_size
        self.heatmap_size = heatmap_size
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
                pts = self.sort_points(pts)
                
                for img_path in session.glob("*.png"):
                    self.samples.append({"img": img_path, "pts": pts})
        
        print(f"Global dataset ready: {len(self.samples)} images.")

    def sort_points(self, pts):
        pts = pts[np.argsort(pts[:, 1]), :]
        top = pts[:2, :][np.argsort(pts[:2, 0]), :]
        bottom = pts[2:, :][np.argsort(pts[2:, 0]), :]
        return np.array([top[0], top[1], bottom[1], bottom[0]]) # TL, TR, BR, BL

    def create_heatmap(self, pts, h, w):
        # 4 个通道对应 4 个角
        heatmaps = np.zeros((4, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        stride = self.img_size / self.heatmap_size
        
        for i, (px, py) in enumerate(pts):
            # 将原图坐标映射到热力图尺度
            tx = (px / w * self.img_size) / stride
            ty = (py / h * self.img_size) / stride
            
            # 在 (tx, ty) 处画一个高斯园
            for y in range(self.heatmap_size):
                for x in range(self.heatmap_size):
                    dist_sq = (x - tx)**2 + (y - ty)**2
                    heatmaps[i, y, x] = np.exp(-dist_sq / (2 * 1.5**2)) # 半径为 1.5 的高斯核
        return heatmaps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(str(sample['img']))
        h, w = img.shape[:2]
        
        # 缩放图像
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 生成 4 通道热力图
        heatmaps = self.create_heatmap(sample['pts'], h, w)
        
        # 转换并归一化
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        # 标准 ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor, torch.tensor(heatmaps)
