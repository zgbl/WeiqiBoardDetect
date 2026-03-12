import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np
import random
import argparse
from tqdm import tqdm
from pathlib import Path

# 导入本地模块
from utils.dataset import GomradeCornerDataset
from models.model_win import CornerRegressor

class HardcoreAugDataset(GomradeCornerDataset):
    """支持同步强几何变换的数据集"""
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(str(sample["img_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        pts = sample["pts"].copy()

        # 1. 强力几何增强 (只有训练时做，由代码逻辑控制)
        if self.transform is not None: # 假设给训练集传了 transform，给验证集传 None 或者简单 resize
            if random.random() > 0.3:
                # 随机透视变换 (Perspective Transform)
                distortion_scale = 0.2
                start_pts = np.float32([[0,0], [w,0], [w,h], [0,h]])
                off = distortion_scale * min(w, h)
                end_pts = start_pts + np.random.uniform(-off, off, size=(4, 2)).astype(np.float32)
                
                M = cv2.getPerspectiveTransform(start_pts, end_pts)
                img = cv2.warpPerspective(img, M, (w, h))
                
                # 标签点同步变换
                pts_h = np.column_stack([pts, np.ones(len(pts))])
                pts_transformed = pts_h @ M.T
                pts = pts_transformed[:, :2] / pts_transformed[:, 2:3]

            if random.random() > 0.5:
                # 随机平移/缩放
                tx = random.uniform(-0.1, 0.1) * w
                ty = random.uniform(-0.1, 0.1) * h
                scale = random.uniform(0.8, 1.2)
                M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                img = cv2.warpAffine(img, M, (w, h))
                
                # 标签同步
                pts_h = np.column_stack([pts, np.ones(len(pts))])
                pts = (pts_h @ M.T)

        # 2. 归一化和 Resize
        img_pil = transforms.ToPILImage()(img)
        img_pil = transforms.Resize((224, 224))(img_pil)
        
        # 即使不做几何增强，基本的颜色增强也要有
        if self.transform is not None:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img_pil)

        pts[:, 0] /= w
        pts[:, 1] /= h
        
        return img_tensor, torch.tensor(pts.flatten(), dtype=torch.float32)

def train_hardcore():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="E:/Data/Gomrade/kaggle-gomrade/dataset1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4) # 稍微调低一点学习率，防止猛冲
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Anti-Overfitting Hardcore Mode ---")
    
    root_dir = Path(args.data)
    all_sessions = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    
    # --- 核心改进：按 Session 切分数据集 ---
    random.shuffle(all_sessions)
    train_idx = int(0.8 * len(all_sessions))
    train_sessions = all_sessions[:train_idx]
    val_sessions = all_sessions[train_idx:]
    
    print(f"Splitting: {len(train_sessions)} sessions for Training, {len(val_sessions)} for Validation.")

    train_transform = transforms.Compose([
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = HardcoreAugDataset(root_dir, session_paths=train_sessions, transform=train_transform)
    val_dataset = HardcoreAugDataset(root_dir, session_paths=val_sessions, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = CornerRegressor(backbone='resnet18', pretrained=True).to(device)
    # 增加一个小 Dropout
    model.model.fc[2].p = 0.3 
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # 增加 L2 正则化
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                outputs = model(imgs.to(device))
                val_loss += criterion(outputs, targets.to(device)).item()
        
        avg_val = val_loss/len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train={train_loss/len(train_loader):.6f}, Val={avg_val:.6f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({'model_state_dict': model.state_dict()}, "checkpoints/best_model_hardcore.pth")
            print(f"⭐ Saved: {avg_val:.6f}")

if __name__ == "__main__":
    train_hardcore()
