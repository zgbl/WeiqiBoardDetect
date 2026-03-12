import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.dataset import GomradeCornerDataset
from torchvision import transforms

def visualize_dataset(data_path, num_samples=5):
    # 1. 直接可视化原始数据和标签
    transform_none = None
    dataset_raw = GomradeCornerDataset(data_path, transform=transform_none)
    
    if len(dataset_raw) == 0:
        print("Dataset is empty!")
        return

    os.makedirs("diagnostics", exist_ok=True)
    
    for i in range(min(num_samples, len(dataset_raw))):
        img_tensor, target = dataset_raw[i]
        
        # 将 tensor 转回 numpy
        # 注意：dataset_raw 在没有 transform 时返回的是 (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        h, w = img.shape[:2]
        pts = target.reshape(4, 2).numpy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        
        # 绘制
        for j, (px, py) in enumerate(pts):
            cv2.circle(img, (int(px), int(py)), 10, (0, 0, 255), -1)
            cv2.putText(img, str(j), (int(px), int(py)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        cv2.polylines(img, [pts.astype(np.int32)], True, (0, 255, 0), 2)
        
        save_path = f"diagnostics/sample_{i}_raw.jpg"
        cv2.imwrite(save_path, img)
        print(f"Saved raw diagnostic: {save_path}")

    # 2. 可视化模型真正看到的图像 (Transform 之后)
    img_size = 224
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_aug = GomradeCornerDataset(data_path, transform=transform)
    
    # 反标准化用于显示
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(min(num_samples, len(dataset_aug))):
        img_tensor, target = dataset_aug[i]
        
        # 反标准化
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 在 224x224 尺度上绘制点
        pts = target.reshape(4, 2).numpy()
        pts[:, 0] *= 224
        pts[:, 1] *= 224
        
        for j, (px, py) in enumerate(pts):
            cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)
            
        cv2.polylines(img, [pts.astype(np.int32)], True, (0, 255, 0), 1)
        
        save_path = f"diagnostics/sample_{i}_aug.jpg"
        cv2.imwrite(save_path, img)
        print(f"Saved augmented diagnostic: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="E:/Data/Gomrade/kaggle-gomrade/dataset1", help="Path to dataset")
    args = parser.parse_args()
    visualize_dataset(args.data)
