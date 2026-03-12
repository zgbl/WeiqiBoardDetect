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
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(str(sample["img_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        pts = sample["pts"].copy()

        # 1. 强力几何增强
        if self.transform is not None:
            # 随机透视
            if random.random() > 0.4:
                distortion_scale = 0.12 # 稍微温和一点
                start_pts = np.float32([[0,0], [w,0], [w,h], [0,h]])
                off = distortion_scale * min(w, h)
                end_pts = start_pts + np.random.uniform(-off, off, size=(4, 2)).astype(np.float32)
                M = cv2.getPerspectiveTransform(start_pts, end_pts)
                img = cv2.warpPerspective(img, M, (w, h))
                
                pts_h = np.column_stack([pts, np.ones(len(pts))])
                pts_transformed = pts_h @ M.T
                pts = pts_transformed[:, :2] / pts_transformed[:, 2:3]

            # 随机平移/缩放/微小旋转
            if random.random() > 0.4:
                tx = random.uniform(-0.05, 0.05) * w
                ty = random.uniform(-0.05, 0.05) * h
                angle = random.uniform(-5, 5) # 稍微旋转
                scale = random.uniform(0.9, 1.1)
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                img = cv2.warpAffine(img, M, (w, h))
                
                pts_h = np.column_stack([pts, np.ones(len(pts))])
                pts = (pts_h @ M.T)
            
            # --- 核心改进：再次排序 ---
            # 确保无论怎么变，点 0 永远是视觉左上，点 1 永远是视觉右上...
            pts = self.sort_points(pts)

        # 2. Resize 和 Tensor 化
        # 限制坐标在 [0, 1] 范围内，防止增强出界
        pts[:, 0] = np.clip(pts[:, 0], 0, w)
        pts[:, 1] = np.clip(pts[:, 1], 0, h)
        
        img_pil = transforms.ToPILImage()(img)
        img_pil = transforms.Resize((224, 224))(img_pil)
        
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="Resume training from last best model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Anti-Overfitting V2 (with re-sorting) ---")
    
    root_dir = Path(args.data)
    all_sessions = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    random.seed(42) # 固定随机种子，让切分可重复
    random.shuffle(all_sessions)
    train_idx = int(0.85 * len(all_sessions)) # 稍微多给训练集一点
    train_sessions = all_sessions[:train_idx]
    val_sessions = all_sessions[train_idx:]
    
    print(f"Training on {len(train_sessions)} sessions, testing on {len(val_sessions)} new sessions.")

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
    
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 0

    # --- 核心改进：恢复训练逻辑 ---
    checkpoint_path = "checkpoints/best_model_hardcore_v2.pth"
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming with previous Best Val Loss: {best_val_loss:.6f}")
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from Epoch {start_epoch}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, targets in loop:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            
            if torch.isnan(loss):
                continue
                
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
        scheduler.step(avg_val)
        
        print(f"Epoch {epoch+1}: Train={train_loss/len(train_loader):.6f}, Val={avg_val:.6f} [Last Best: {best_val_loss:.6f}]")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"⭐ NEW BEST SAVED!")

if __name__ == "__main__":
    train_hardcore()
