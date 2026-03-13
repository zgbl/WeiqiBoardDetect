import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path

# 导入本地模块
from dataset import GomradePatchDataset
from model import PatchDetector

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="E:/Data/Gomrade/Corners", help="Patch数据路径")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use_syn", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Robust Patch Detector on: {device}")

    # 1. 划分 Session
    root_path = Path(args.data)
    all_real_files = []
    for corner in ['TL', 'TR', 'BR', 'BL']:
        all_real_files.extend(list((root_path / corner).glob("*.png")))
    all_sessions = sorted(list(set([f.name.split('_')[0] for f in all_real_files])))
    
    np.random.seed(42)
    np.random.shuffle(all_sessions)
    split_idx = int(0.8 * len(all_sessions))
    train_sessions = all_sessions[:split_idx]
    val_sessions = all_sessions[split_idx:]

    # 2. 变换设置
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = GomradePatchDataset(args.data, transform=transform, is_train=True, sessions=train_sessions, use_synthetic=args.use_syn)
    val_dataset = GomradePatchDataset(args.data, transform=transform, is_train=False, sessions=val_sessions, use_synthetic=args.use_syn)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. 初始化模型和损失函数
    model = PatchDetector(backbone='resnet18').to(device)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for imgs, labels, coords in pbar:
            imgs, labels, coords = imgs.to(device), labels.to(device), coords.to(device)
            
            cls_out, reg_out = model(imgs)
            
            # 分类损失 (所有样本)
            loss_cls = criterion_cls(cls_out, labels)
            
            # 坐标回归损失 (仅对正样本计算, label > 0)
            mask = labels > 0
            if mask.sum() > 0:
                loss_reg = criterion_reg(reg_out[mask], coords[mask])
            else:
                loss_reg = 0.0
                
            loss = loss_cls + 10.0 * loss_reg # 增大回归的权重
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels, coords in val_loader:
                imgs, labels, coords = imgs.to(device), labels.to(device), coords.to(device)
                cls_out, reg_out = model(imgs)
                
                loss_cls = criterion_cls(cls_out, labels)
                mask = labels > 0
                loss_reg = criterion_reg(reg_out[mask], coords[mask]) if mask.sum() > 0 else 0
                
                val_loss += (loss_cls + 10.0 * loss_reg).item()
                
                # 统计分类准确率
                preds = torch.argmax(cls_out, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss: {avg_val_loss:.4f}, Acc: {accuracy:.2%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_patch_detector.pth")
            print("⭐ Saved Best Detector!")

if __name__ == "__main__":
    train()
