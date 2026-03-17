import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from dataset import MultiClassPatchDataset
from model import PatchClassifier

def fine_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on Hard Cases using: {device}")

    # 路径配置：只针对 False-Positive 文件夹
    HARDCASE_DIR = Path(r"E:\Data\Gomrade\MultiClassPatches\False-Positive")
    WEIGHTS_DIR = Path("weights")
    BEST_WEIGHTS = WEIGHTS_DIR / "best_classifier.pth"
    
    if not BEST_WEIGHTS.exists():
        print(f"Error: {BEST_WEIGHTS} not found. Hardcase training requires a base model.")
        return

    # 加载数据集 (只加载 False-Positive 里的硬样本)
    dataset = MultiClassPatchDataset(HARDCASE_DIR, is_train=True)
    if len(dataset) == 0:
        print("No samples found in False-Positive directory!")
        return
        
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True, num_workers=0)

    # 初始化模型并加载权重
    model = PatchClassifier(num_classes=4).to(device)
    print(f"Loading base weights from {BEST_WEIGHTS}")
    model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device))
    
    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    # 使用较小的学习率进行微调
    optimizer = optim.Adam(model.parameters(), lr=1e-5) 

    epochs = 3
    print(f"Starting 10 epochs of hardcase strengthening on {len(dataset)} samples...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)

    # 备份并保存
    backup_path = WEIGHTS_DIR / "best_classifier_pre_hardcase.pth"
    if not backup_path.exists():
        import shutil
        shutil.copy(str(BEST_WEIGHTS), str(backup_path))
        print(f"Backup created at {backup_path}")

    torch.save(model.state_dict(), str(BEST_WEIGHTS))
    print(f"Strengthened model saved to {BEST_WEIGHTS}")

if __name__ == "__main__":
    fine_tune()
