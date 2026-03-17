import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from pathlib import Path
import yaml
import random

from model import PatchCornerRegressor

def test_on_full_board(image_path, model_path, output_path="board_test_result.png", jitter_px=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = PatchCornerRegressor(backbone='resnet18', pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. 读取图片
    full_img = cv2.imread(image_path)
    if full_img is None:
        print("Image not found")
        return
    h, w = full_img.shape[:2]

    # 3. 获取“带噪声”的初始坐标
    yml_path = Path(image_path).parent / "board_extractor_state.yml"
    if yml_path.exists():
        with open(yml_path, 'r') as f:
            data = yaml.safe_load(f)
            true_pts = np.array(data['pts_clicks'])
        # --- 核心：故意制造人工偏差 (作弊检测) ---
        print(f"Adding random jitter of +/-{jitter_px} pixels to simulate rough detection...")
        rough_pts = true_pts + np.random.randint(-jitter_px, jitter_px, size=true_pts.shape)
    else:
        # 如果没有 yml，就随便点四个点 (比如按比例)
        rough_pts = np.array([[w*0.1, h*0.1], [w*0.9, h*0.1], [w*0.9, h*0.9], [w*0.1, h*0.9]])

    # 4. 准备变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    vis_img = full_img.copy()
    size = 128
    half = size // 2

    for i, (rx, ry) in enumerate(rough_pts):
        # 裁剪 Patch
        x1, y1 = int(rx - half), int(ry - half)
        crop = np.zeros((size, size, 3), dtype=np.uint8)
        
        # 提取图像块
        c_x1, c_y1 = max(0, x1), max(0, y1)
        c_x2, c_y2 = min(w, x1 + size), min(h, y1 + size)
        if c_x2 > c_x1 and c_y2 > c_y1:
            crop[c_y1-y1:c_y2-y1, c_x1-x1:c_x2-x1] = full_img[c_y1:c_y2, c_x1:c_x2]
        
        # 推理
        input_t = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_t).cpu().numpy()[0]
        
        # 还原到全图坐标
        fx = x1 + pred[0] * size
        fy = y1 + pred[1] * size

        # --- 可视化对比 ---
        # 蓝色圆圈：故意给错的、带偏差的初始点 (Input to model)
        cv2.circle(vis_img, (int(rx), int(ry)), 8, (255, 0, 0), 2)
        # 绿色十字：模型纠偏后的最终结果 (Output of model)
        cv2.drawMarker(vis_img, (int(fx), int(fy)), (0, 255, 0), cv2.MARKER_CROSS, 25, 2)
        
        if yml_path.exists():
            # 红色小点：真实的 Label 位置 (仅供参考)
            cv2.circle(vis_img, (int(true_pts[i][0]), int(true_pts[i][1])), 3, (0, 0, 255), -1)

    cv2.imwrite(output_path, vis_img)
    print(f"Tested with jitter. Results saved to: {output_path}")
    print("Blue circles = Jittered Input, Green Crosses = Model Prediction, Red Dot = Ground Truth")

if __name__ == "__main__":
    # 随便换一张你没训练过的 session 图片试试！
    TEST_IMG = r"E:\Data\Gomrade\kaggle-gomrade\dataset1\1\11.png"
    MODEL = "checkpoints/best_patch_model_joint.pth"
    test_on_full_board(TEST_IMG, MODEL)
