import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from pathlib import Path
import random

# 导入本地模块
from model import PatchCornerRegressor
from dataset import GomradePatchDataset

def verify_joint_model():
    # 0. 配置 (确保这里的文件名和 train.py 保存的一致)
    CHECKPOINT_PATH = "checkpoints/best_patch_model_joint.pth"
    DATA_PATH = r"E:\Data\Gomrade\Corners"
    OUTPUT_FOLDER = "validation_results"
    NUM_SAMPLES = 10
    PATCH_SIZE = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Model: {CHECKPOINT_PATH} on {device}")

    # 1. 加载模型
    model = PatchCornerRegressor(backbone='resnet18', pretrained=False).to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found! Did you run train.py?")
        return
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()

    # 2. 准备数据 (包含合成数据进行测试)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 开启用合成数据混合测试
    dataset = GomradePatchDataset(DATA_PATH, transform=transform, is_train=False, use_synthetic=True)
    if len(dataset) == 0: return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))

    for i, idx in enumerate(indices):
        sample = dataset.samples[idx]
        img_bgr = cv2.imread(str(sample["path"]))
        if img_bgr is None: continue
        
        # 为了测试鲁棒性，随机做一个位置抖动
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.cpu().numpy()[0]
            
        # 还原位置
        px, py = pred[0] * w, pred[1] * h
        gt_x, gt_y = sample["gt"]

        # 画图
        vis = img_bgr.copy()
        # 绿色：预测结果
        cv2.drawMarker(vis, (int(px), int(py)), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
        # 蓝色：标签真值
        cv2.circle(vis, (int(gt_x), int(gt_y)), 3, (255, 0, 0), -1)
        
        err = np.sqrt((px-gt_x)**2 + (py-gt_y)**2)
        cv2.putText(vis, f"Err: {err:.2f}px", (5, 20), 0, 0.5, (0, 255, 0), 1)

        save_path = f"{OUTPUT_FOLDER}/test_{i}_{sample['type']}.png"
        cv2.imwrite(save_path, vis)

    print(f"Visualized samples saved to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    verify_joint_model()
