import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from pathlib import Path

from dataset_global import GlobalBoardDataset
from model_global import GlobalKeypointModel

class WeightedMSELoss(nn.Module):
    """针对热力图关键点的加权 MSE，强制关注非零区域"""
    def __init__(self, pos_weight=20.0):
        super(WeightedMSELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        # 基础 MSE
        mse = (pred - target) ** 2
        # 对 target > 0 的区域应用更高权重
        weight = torch.ones_like(target) + (target > 0.01).float() * self.pos_weight
        return torch.mean(mse * weight)

def calculate_pck(outputs, targets, threshold=3.0):
    """
    计算关键点正确率。threshold 是在 56x56 热力图尺度下的像素距离。
    """
    batch_size = outputs.size(0)
    num_pts = outputs.size(1)
    
    # 提取预测和真实坐标
    correct = 0
    total = batch_size * num_pts
    
    for b in range(batch_size):
        for p in range(num_pts):
            out_hm = outputs[b, p].detach().cpu().numpy()
            tar_hm = targets[b, p].detach().cpu().numpy()
            
            # 如果 target 全是 0，跳过不计
            if np.max(tar_hm) < 0.1: continue
            
            py, px = np.unravel_index(np.argmax(out_hm), out_hm.shape)
            ty, tx = np.unravel_index(np.argmax(tar_hm), tar_hm.shape)
            
            dist = np.sqrt((px - tx)**2 + (py - ty)**2)
            if dist <= threshold:
                correct += 1
    return correct, total

def train_global_hardcore():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting HARDCORE Global Training on: {device}")

    DATA_DIRS = [
        r"E:\Data\Gomrade\kaggle-gomrade\dataset1",
        r"E:\Data\Gomrade\kaggle-gomrade\dataset2"
    ]
    
    # --- 修复 1: 初始化两个完全独立的 Dataset 实例 ---
    full_dataset = GlobalBoardDataset(DATA_DIRS, is_train=True) # 临时用于计算分割
    n = len(full_dataset)
    indices = list(range(n))
    np.random.shuffle(indices)
    split = int(0.9 * n)
    
    train_idx, val_idx = indices[:split], indices[split:]
    
    # 构建真正隔离的子集
    train_dataset = GlobalBoardDataset(DATA_DIRS, is_train=True)
    train_dataset.samples = [train_dataset.samples[i] for i in train_idx]
    
    val_dataset = GlobalBoardDataset(DATA_DIRS, is_train=False) # 验证集关闭增强
    val_dataset.samples = [val_dataset.samples[i] for i in val_idx]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # --- 修复 4: 降低学习率，增强稳定性 ---
    model = GlobalKeypointModel(num_points=4).to(device)
    criterion = WeightedMSELoss(pos_weight=50.0) # 使用加权 MSE
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_pck = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, heatmaps in pbar:
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, heatmaps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # --- 修复 3: 引入 PCK 评估指标 ---
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, heatmaps in val_loader:
                imgs, heatmaps = imgs.to(device), heatmaps.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, heatmaps).item()
                
                c, t = calculate_pck(outputs, heatmaps)
                val_correct += c
                val_total += t
        
        avg_pck = val_correct / val_total
        scheduler.step()
        print(f"Epoch {epoch+1}: Val Loss: {val_loss/len(val_loader):.6f}, PCK: {avg_pck:.2%}")

        # 以 PCK 为准保存模型
        if avg_pck > best_pck:
            best_pck = avg_pck
            torch.save(model.state_dict(), "checkpoints/best_global_model_hardcore.pth")
            print(f"⭐ Saved Best Model with PCK: {avg_pck:.2%}")

if __name__ == "__main__":
    train_global_hardcore()
