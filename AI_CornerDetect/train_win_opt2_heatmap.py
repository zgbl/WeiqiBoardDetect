import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- 1. 热力图模型架构 ---
class HeatmapRegressor(nn.Module):
    def __init__(self):
        super(HeatmapRegressor, self).__init__()
        # 使用 ResNet18 作为特征提取器
        backbone = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2]) # 去掉全连接层和池化
        
        # 上采样头：将 7x7 的特征图逐步恢复到 56x56
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 7->14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 14->28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 28->56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1) # 输出 4 个通道，分别对应 4 个角
        )

    def forward(self, x):
        x = self.features(x)
        x = self.decoder(x)
        return x

# --- 2. 热力图数据集 ---
from utils.dataset import GomradeCornerDataset
class HeatmapDataset(GomradeCornerDataset):
    def __init__(self, root_dir, transform=None, heatmap_size=56, sigma=2):
        super().__init__(root_dir, transform)
        self.heatmap_size = heatmap_size
        self.sigma = sigma

    def __getitem__(self, idx):
        img_tensor, target_pts = super().__getitem__(idx)
        # target_pts 是 flat 的 (8,)
        pts = target_pts.reshape(4, 2).numpy()
        
        heatmaps = np.zeros((4, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        for i in range(4):
            # 将归一化坐标映射到热力图尺度
            mu_x = int(pts[i, 0] * self.heatmap_size)
            mu_y = int(pts[i, 1] * self.heatmap_size)
            heatmaps[i] = self.generate_heatmap(mu_x, mu_y)
        
        return img_tensor, torch.from_numpy(heatmaps)

    def generate_heatmap(self, mu_x, mu_y):
        size = self.heatmap_size
        x = np.arange(0, size, 1, np.float32)
        y = np.arange(0, size, 1, np.float32)
        x, y = np.meshgrid(x, y)
        heatmap = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * self.sigma**2))
        return heatmap

# --- 3. 训练主逻辑 ---
def train_heatmap():
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- Option 2: Heatmap Regression Mode ---")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = HeatmapDataset("E:/Data/Gomrade/kaggle-gomrade/dataset1", transform=transform)
    train_size = int(0.9 * len(dataset))
    val_dataset = random_split(dataset, [train_size, len(dataset)-train_size])[1]
    
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = HeatmapRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        for imgs, hms in loop:
            imgs, hms = imgs.to(device), hms.to(device)
            preds = model(imgs)
            loss = criterion(preds, hms)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(mse=loss.item())

        # 保存模型
        torch.save({'model_state_dict': model.state_dict()}, "checkpoints/best_model_heatmap.pth")

if __name__ == "__main__":
    train_heatmap()
