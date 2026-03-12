import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path

# 导入本地模块
from utils.dataset import GomradeCornerDataset
from models.model_win import CornerRegressor

class AugmentedDataset(GomradeCornerDataset):
    """支持同步几何变换的增强数据集"""
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(str(sample["img_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 内部转 RGB
        h, w = img.shape[:2]
        pts = sample["pts"].copy()

        # --- 随机几何增强 ---
        if random.random() > 0.5:
            # 1. 随机透视变换 (模拟不同拍摄角度)
            # 这里简单实现：给 4 个角各加一个随机偏移，然后做 Perspective
            distortion_scale = 0.15
            start_points = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32)
            off = distortion_scale * min(w, h)
            end_points = start_points + np.random.uniform(-off, off, size=(4, 2)).astype(np.float32)
            
            M = cv2.getPerspectiveTransform(start_points, end_points)
            img = cv2.warpPerspective(img, M, (w, h))
            
            # 同时变换标签点
            pts_ones = np.ones((4, 3))
            pts_ones[:, :2] = pts
            pts_new = pts_ones @ M.T
            pts = pts_new[:, :2] / pts_new[:, 2:3]

        # 归一化和缩放
        img = cv2.resize(img, (224, 224))
        pts[:, 0] /= w
        pts[:, 1] /= h
        
        # 基础增强 (颜色)
        img_pil = transforms.ToPILImage()(img)
        img_tensor = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img_pil)

        return img_tensor, torch.tensor(pts.flatten(), dtype=torch.float32)

import cv2 # 脚本内部需要用

def train_opt1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="E:/Data/Gomrade/kaggle-gomrade/dataset1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Option 1: Regression + Heavy Augmentations ---")
    
    full_dataset = AugmentedDataset(args.data)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = CornerRegressor(backbone='resnet18', pretrained=True).to(device)
    criterion = nn.MSELoss() # 回归任务 MSE 通常更稳
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, targets in loop:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 验证部分省略（同 train_win.py）...
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, targets in val_loader:
                outputs = model(imgs.to(device))
                val_loss += criterion(outputs, targets.to(device)).item()
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'model_state_dict': model.state_dict()}, "checkpoints/best_model_opt1.pth")
            print(f"⭐ Saved Opt1: {avg_val_loss:.6f}")

if __name__ == "__main__":
    import argparse
    train_opt1()
