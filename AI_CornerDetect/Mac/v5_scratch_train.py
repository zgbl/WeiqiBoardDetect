import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
import sys

# 手动添加路径以导入 PatchClassifier
MAC_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(MAC_DIR)
CLASSIFIER_V1_DIR = os.path.join(PARENT_DIR, "Classifier_V1")

if CLASSIFIER_V1_DIR not in sys.path: sys.path.insert(0, CLASSIFIER_V1_DIR)

try:
    from model import PatchClassifier
except ImportError:
    if PARENT_DIR not in sys.path: sys.path.insert(0, PARENT_DIR)
    from Classifier_V1.model import PatchClassifier

# ========== 配置参数 ==========
# 数据根目录 (使用您清理后的 debug_extracted_patches)
DATA_DIR = "/Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/debug_extracted_patches"
# 输出的新权重路径
NEW_WEIGHTS_PATH = "/Users/tuxy/Codes/AI/WeiqiBoardDetect/data/Weights/4Classes-V5_Scratch.pth"

BATCH_SIZE = 16 # 数据量小，使用稍小的 batch_size
EPOCHS = 25
LR = 1e-4 # 从头训练建议使用较大初始学习率

# 类别映射与文件夹对齐
CLASS_MAP = {
    'Corner': 0,
    'Inner': 1,
    'Edge': 2,
    'Outer': 3
}
# =============================

class ScratchDataset(Dataset):
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.samples = []
        for cname, cid in CLASS_MAP.items():
            folder = self.root / cname
            if not folder.exists(): 
                print(f"Warning: Folder {folder} not found, skipping label {cid}")
                continue
            files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            for f in files:
                self.samples.append((f, cid))
        print(f"[*] Training dataset: {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = cv2.imread(str(fpath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 数据增强 (帮助应对过拟合)
        if np.random.rand() > 0.5: img = cv2.flip(img, 1) # 水平翻转
        if np.random.rand() > 0.5: img = cv2.flip(img, 0) # 垂直翻转
        img = np.rot90(img, np.random.randint(0,4)).copy()

        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = (img - mean) / std
        return img, label

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Training on: {device}")

    # 1. 初始化一个全新的 PatchClassifier (仅保留 ImageNet 初始特征，丢弃 V3/V4 权重)
    model = PatchClassifier(num_classes=4).to(device)
    print("[*] Model initialized from scratch (reset all local weights).")

    # 2. 加载整理后的这一小批数据
    dataset = ScratchDataset(DATA_DIR)
    if len(dataset) == 0:
        print("[Error] No data found in debug_extracted_patches. Please check folders.")
        return
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. 训练循环
    print(f"[*] Starting 25 Epochs of Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{total_loss/(total/BATCH_SIZE):.4f}", acc=f"{100.*correct/total:.2f}%")

    # 5. 保存 V5_Scratch
    os.makedirs(os.path.dirname(NEW_WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), NEW_WEIGHTS_PATH)
    print(f"\n[*] Training Complete. New model saved to: {NEW_WEIGHTS_PATH}")

if __name__ == "__main__":
    main()
