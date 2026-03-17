import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

from dataset import MultiClassPatchDataset
from model import PatchClassifier

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Training on: {device}")

    # 数据根目录 (由您手动清理后的 patches)
    ROOT_DIR = r"E:\Data\Gomrade\MultiClassPatches"
    
    # 第一次初始化仅为获取所有样本以进行分割
    full_dataset = MultiClassPatchDataset(ROOT_DIR, is_train=False)
    
    # --- 严格隔离测试集 ---
    n = len(full_dataset)
    indices = list(range(n))
    np.random.shuffle(indices)
    
    # 80/10/10 分割
    train_split = int(0.8 * n)
    val_split = int(0.9 * n)
    
    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]
    
    # 记录测试集路径，确保绝对参与不到训练
    split_info = {
        'test_set': [str(full_dataset.samples[i][0]) for i in test_idx]
    }
    with open("test_set_isolation.json", "w") as f:
        json.dump(split_info, f)
    
    # 重新构造训练和验证集
    train_dataset = MultiClassPatchDataset(ROOT_DIR, is_train=True)
    train_dataset.samples = [full_dataset.samples[i] for i in train_idx]
    
    val_dataset = MultiClassPatchDataset(ROOT_DIR, is_train=False)
    val_dataset.samples = [full_dataset.samples[i] for i in val_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = PatchClassifier(num_classes=4).to(device)
    
    # 支持接力训练：检查是否存在上一轮的权重
    weight_path = Path("weights/best_classifier.pth")
    if weight_path.exists():
        print(f"Relay Training Active: Loading weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print("Starting training from scratch.")

    criterion = nn.CrossEntropyLoss()
    # 因为是接力训练，学习率可以缩小一点，或者保持 1e-4
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    
    best_acc = 0.0
    epochs = 30

    
    os.makedirs("weights", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(acc=100.*correct/total)

        # 验证
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                v_total += labels.size(0)
                v_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * v_correct / v_total
        print(f"Epoch {epoch+1}: Val Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights/best_classifier.pth")
            print(f"⭐ Saved Best Model: {val_acc:.2f}%")

    print("\nTraining Complete.")
    print(f"Best Val Accuracy: {best_acc:.2f}%")
    print(f"Isolated test set saved in test_set_isolation.json")

if __name__ == "__main__":
    train()
