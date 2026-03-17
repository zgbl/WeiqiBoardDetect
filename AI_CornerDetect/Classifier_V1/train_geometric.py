import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
from pathlib import Path
import cv2
import random
from tqdm import tqdm

from model import PatchClassifier

# --- 1. 几何约束损失函数 ---

class GeometricConsistencyLoss(nn.Module):
    def __init__(self, lambda_line=0.5):
        super().__init__()
        self.lambda_line = lambda_line

    def forward(self, logits, coords, labels):
        """
        logits: (B*N, 4) 
        coords: (B, N, 2) 
        labels: (B, N)
        """
        B = coords.size(0)
        N = coords.size(1)
        logits = logits.view(B, N, 4)
        probs = F.softmax(logits, dim=-1)
        
        edge_p = probs[:, :, 1] # Edge 概率
        
        loss_geom = 0
        for i in range(B):
            b_coords = coords[i]
            b_labels = labels[i]
            b_edge_p = edge_p[i]
            
            # 获取真实的 4 个角点坐标
            gt_corners = b_coords[b_labels == 0]
            if len(gt_corners) < 4: continue
            
            # 排序角点以便连线
            center = gt_corners.mean(dim=0)
            angles = torch.atan2(gt_corners[:,1]-center[1], gt_corners[:,0]-center[0])
            gt_corners = gt_corners[torch.argsort(angles)]
            
            # 定义 4 条边线 (0-1, 1-2, 2-3, 3-0)
            lines = [(gt_corners[0], gt_corners[1]), (gt_corners[1], gt_corners[2]), 
                     (gt_corners[2], gt_corners[3]), (gt_corners[3], gt_corners[0])]
            
            # 几何约束：如果一个点的 Edge 概率很高，它距离这 4 条线的最小距离应该很小
            for j in range(N):
                p = b_coords[j]
                p_edge_prob = b_edge_p[j]
                
                # 计算到 4 条线的最小距离
                dists = []
                for a, b in lines:
                    pa = p - a
                    ba = b - a
                    t = torch.dot(pa, ba) / (torch.dot(ba, ba) + 1e-6)
                    t = torch.clamp(t, 0, 1)
                    dists.append(torch.norm(pa - t * ba))
                
                min_dist = torch.stack(dists).min()
                # 损失：概率 * 距离 (强制模型在离线远的地方不要给 Edge 高分)
                # 距离归一化：除以 100 像素以保持 Loss 数量级
                loss_geom += p_edge_prob * (min_dist / 100.0)

        return (loss_geom / B) * self.lambda_line

# --- 2. 棋盘单位的数据集 ---

class BoardGroupDataset(Dataset):
    def __init__(self, root_dirs, patch_size=128, n_samples=32):
        self.patch_size = patch_size
        self.n_samples = n_samples # 每张图采样 32 个 patch
        self.samples = []
        
        for root in root_dirs:
            root = Path(root)
            for session in root.iterdir():
                if not session.is_dir(): continue
                yml = session / "board_extractor_state.yml"
                if not yml.exists(): continue
                with open(yml, 'r') as f:
                    data = yaml.safe_load(f)
                if 'pts_clicks' not in data: continue
                pts = np.array(data['pts_clicks'], dtype=np.float32)
                
                imgs = list(session.glob("*.png")) + list(session.glob("*.jpg"))
                for img in imgs:
                    self.samples.append({'img': img, 'pts': pts})
        print(f"Geometric Dataset: {len(self.samples)} boards loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(str(s['img']))
        if img is None: return self.__getitem__((idx+1)%len(self.samples))
        h, w = img.shape[:2]
        pts = s['pts'].copy()
        
        patches = []
        coords = []
        labels = []
        
        # 1. 采样 4 个角点 (Corner)
        for p in pts:
            patches.append(self.extract_patch(img, p))
            coords.append(p)
            labels.append(0)
            
        # 2. 采样 12 个边点 (Edge)
        for i in range(4):
            p1, p2 = pts[i], pts[(i+1)%4]
            for _ in range(3):
                t = random.random()
                p = p1 * t + p2 * (1-t)
                patches.append(self.extract_patch(img, p))
                coords.append(p)
                labels.append(1)
                
        # 3. 采样 8 个内点 (Inner)
        center = pts.mean(axis=0)
        for _ in range(8):
            r1, r2 = random.random(), random.random()
            if r1+r2 > 1: r1, r2 = 1-r1, 1-r2
            p = (1-r1-r2)*pts[0] + r1*pts[1] + r2*pts[2] # 随便采个三角形内
            patches.append(self.extract_patch(img, p))
            coords.append(p)
            labels.append(2)
            
        # 4. 采样 8 个外点 (Outer)
        for _ in range(8):
            p = np.array([random.randint(0, w-1), random.randint(0, h-1)])
            patches.append(self.extract_patch(img, p))
            coords.append(p)
            labels.append(3)

        # 转换为 Tensor
        patches_tensor = torch.stack(patches)
        coords_tensor = torch.from_numpy(np.stack(coords)).float()
        labels_tensor = torch.tensor(labels).long()
        
        return patches_tensor, coords_tensor, labels_tensor

    def extract_patch(self, img, pt):
        h, w = img.shape[:2]
        half = self.patch_size // 2
        x, y = int(pt[0]), int(pt[1])
        x1, y1, x2, y2 = x-half, y-half, x+half, y+half
        
        patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        s_x1, s_y1 = max(0, x1), max(0, y1)
        s_x2, s_y2 = min(w, x2), min(h, y2)
        d_x1, d_y1 = s_x1-x1, s_y1-y1
        d_x2, d_y2 = d_x1+(s_x2-s_x1), d_y1+(s_y2-s_y1)
        
        if s_x2 > s_x1 and s_y2 > s_y1:
            patch[d_y1:d_y2, d_x1:d_x2] = img[s_y1:s_y2, s_x1:s_x2]
        
        # 归一化
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (t - mean) / std

# --- 3. 训练主循环 ---

def train_geometric():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT_DIRS = [r"E:\Data\Gomrade\kaggle-gomrade\dataset1", r"E:\Data\Gomrade\kaggle-gomrade\dataset2"]
    
    dataset = BoardGroupDataset(ROOT_DIRS)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0) # BatchSize 设小点，因为内部有很多 patch

    model = PatchClassifier(num_classes=4).to(device)
    # 接力训练
    weight_path = Path("weights/best_classifier.pth")
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("Loaded previous weights for geometric relay.")

    ce_criterion = nn.CrossEntropyLoss()
    geom_criterion = GeometricConsistencyLoss(lambda_line=1.0) # 几何损失权重
    optimizer = optim.Adam(model.parameters(), lr=5e-6) # 极低学习率微调

    for epoch in range(5): # 进行 5 轮几何强化
        model.train()
        pbar = tqdm(loader, desc=f"Geo-Epoch {epoch+1}")
        for patches, coords, labels in pbar:
            B, N, C, H, W = patches.shape
            patches = patches.view(-1, C, H, W).to(device)
            coords, labels = coords.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(patches)
            
            loss_ce = ce_criterion(logits, labels.view(-1))
            loss_geo = geom_criterion(logits, coords, labels)
            
            loss = loss_ce + loss_geo
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(ce=loss_ce.item(), geo=loss_geo.item())

    torch.save(model.state_dict(), "weights/best_classifier_geometric.pth")
    print("Geometric strengthened model saved!")

if __name__ == "__main__":
    train_geometric()
