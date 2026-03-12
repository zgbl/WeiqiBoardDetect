import os
import yaml
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class GomradeCornerDataset(Dataset):
    def __init__(self, root_dir, session_paths=None, transform=None):
        """
        Args:
            root_dir: Gomrade-dataset1 根路径
            session_paths: 可选，指定的 session 目录列表
            transform: pytorch 图像变换
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # 如果没传 session_paths，就遍历所有
        if session_paths is None:
            sessions = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        else:
            sessions = session_paths
            
        for session in sessions:
            ext_yml = session / "board_extractor_state.yml"
            if not ext_yml.exists():
                continue
                
            # 加载坐标点
            try:
                with open(ext_yml, 'r') as f:
                    data = yaml.safe_load(f)
                if "pts_clicks" not in data:
                    continue
                pts = np.array(data["pts_clicks"], dtype=np.float32) # (4, 2)
                
                # 查找该 session 下的所有图像
                img_files = list(session.glob("*.png")) + list(session.glob("*.jpg"))
                
                # --- 核心改进：强制排序点 ---
                # 无论点击顺序如何，统一为：[左上, 右上, 右下, 左下]
                pts_sorted = self.sort_points(pts)
                
                for img_path in img_files:
                    self.samples.append({
                        "img_path": img_path,
                        "pts": pts_sorted
                    })
            except Exception as e:
                print(f"Error loading {session}: {e}")
                
        print(f"Found {len(self.samples)} images with corner labels.")

    def sort_points(self, pts):
        """将 4 个点排序为 [TL, TR, BR, BL] (顺时针)"""
        # 按照 y 坐标排序，分出顶部两个点和底部两个点
        pts = pts[np.argsort(pts[:, 1]), :]
        top_two = pts[:2, :]
        bottom_two = pts[2:, :]
        
        # 顶部两个点按 x 排序 -> TL, TR
        tl, tr = top_two[np.argsort(top_two[:, 0]), :]
        # 底部两个点按 x 排序 -> BL, BR (注意：为了顺时针，底层顺序是 BR, BL)
        bl, br = bottom_two[np.argsort(bottom_two[:, 0]), :]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = str(sample["img_path"])
        pts = sample["pts"].copy()
        
        # 加载图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 归一化坐标 (0-1)
        # pts 是 4个 (x, y)，打平为 8 个维度的向量
        pts[:, 0] = pts[:, 0] / w
        pts[:, 1] = pts[:, 1] / h
        target = pts.flatten() # [x1, y1, x2, y2, x3, y3, x4, y4]
        
        if self.transform:
            img = self.transform(img)
        else:
            # 简单的转 Tensor 和缩放（如果没有传入 transform）
            img = cv2.resize(img, (224, 224))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, torch.tensor(target, dtype=torch.float32)

if __name__ == "__main__":
    # 测试代码
    ds = GomradeCornerDataset("/Users/tuxy/Codes/AI/Data/Gomrade-dataset1")
    if len(ds) > 0:
        img, target = ds[0]
        print(f"Image shape: {img.shape}")
        print(f"Target (normalized): {target}")
