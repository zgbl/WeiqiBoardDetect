import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import time
import argparse
from pathlib import Path

# 导入本地模块
from utils.dataset import GomradeCornerDataset
from models.model_win import CornerRegressor

"""
Windows + CUDA 训练脚本
针对 NVIDIA 1080 优化。
Python 3.12 建议安装方式:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"""

def train_windows():
    # 0. 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="E:/Data/Gomrade/kaggle-gomrade/dataset1", help="Windows 上数据集的路径")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32) # 1080 8G 显存可以开到 32 甚至更高
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0) # Windows 下建议不要设置太高，否则容易内存报错
    args = parser.parse_args()

    # 1. 设备配置 (优先使用 CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on: {device} ---")
    if device.type == 'cuda':
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # 2. 数据准备
    img_size = 224
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 注意：确保这里传入的路径在 Windows 上是正确的
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data path {data_path} not found! Please check your D:/ or C:/ drive.")
        return

    full_dataset = GomradeCornerDataset(data_path, transform=transform)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Windows 下 num_workers 设置过大会导致 'BrokenPipeError', 这里加一个保护
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # 3. 初始化模型
    model = CornerRegressor(backbone='resnet18', pretrained=True).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 4. 训练
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (imgs, targets) in enumerate(loop):
            imgs, targets = imgs.to(device), targets.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device).float(), targets.to(device).float()
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"checkpoints/best_model_win.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)
            print(f"⭐ Saved Best Windows Model: {avg_val_loss:.6f}")

    print("Training Finished!")

# Windows 下 PyTorch 多进程必须放在 if __name__ == '__main__' 后面
if __name__ == "__main__":
    train_windows()
