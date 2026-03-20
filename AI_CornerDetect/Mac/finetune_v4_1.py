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
# Classifier_V1 所在目录
CLASSIFIER_V1_DIR = os.path.join(PARENT_DIR, "Classifier_V1")

if CLASSIFIER_V1_DIR not in sys.path:
    sys.path.insert(0, CLASSIFIER_V1_DIR)

try:
    from model import PatchClassifier
except ImportError:
    # 另一种可能的路径结构
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    from Classifier_V1.model import PatchClassifier

# ========== 配置参数 ==========
DATA_DIR = "/Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/debug_extracted_patches"
V3_WEIGHTS = "/Users/tuxy/Codes/AI/WeiqiBoardDetect/data/Weights/4Classes-V3.pth"
V4_1_WEIGHTS = "/Users/tuxy/Codes/AI/WeiqiBoardDetect/data/Weights/4Classes-V4.1.pth"

BATCH_SIZE = 32
EPOCHS = 10
LR = 5e-6 # 微调建议使用更小的学习率

# 映射关系务必与您的手动清理文件夹对齐 (索引 1: Inner, 索引 2: Edge)
CLASS_MAP = {
    'Corner': 0,
    'Inner': 1,
    'Edge': 2,
    'Outer': 3
}
# =============================

class FineTuneDataset(Dataset):
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.samples = []
        for cname, cid in CLASS_MAP.items():
            folder = self.root / cname
            if not folder.exists(): continue
            files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            for f in files:
                self.samples.append((f, cid))
        print(f"[*] Finetune Dataset: {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = cv2.imread(str(fpath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 简单增强
        if np.random.rand() > 0.5: img = cv2.flip(img, 1)
        if np.random.rand() > 0.5: img = np.rot90(img, np.random.randint(0,4)).copy()

        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = (img - mean) / std
        return img, label

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # 1. 加载模型与 V3 权重
    model = PatchClassifier(num_classes=4).to(device)
    
    ckpt = torch.load(V3_WEIGHTS, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
    model_keys = model.state_dict().keys()
    new_sd = {}
    for k, v in state_dict.items():
        if k in model_keys: new_sd[k] = v
        elif "backbone." + k in model_keys: new_sd["backbone." + k] = v
        elif k.replace("backbone.", "") in model_keys: new_sd[k.replace("backbone.", "")] = v
    
    model.load_state_dict(new_sd, strict=False)
    print(f"[*] Resuming from V3: {V3_WEIGHTS}")

    # 2. 准备数据
    dataset = FineTuneDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 训练配置
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 4. 训练循环
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
            pbar.set_postfix(loss=total_loss/(total/BATCH_SIZE), acc=100.*correct/total)

    # 5. 保存 V4.1
    os.makedirs(os.path.dirname(V4_1_WEIGHTS), exist_ok=True)
    torch.save(model.state_dict(), V4_1_WEIGHTS)
    print(f"\n[*] Training Finished. Model saved to: {V4_1_WEIGHTS}")

if __name__ == "__main__":
    main()
