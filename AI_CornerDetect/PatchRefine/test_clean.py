import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from pathlib import Path
import argparse

from model import PatchCornerRegressor

def test_unseen_image(image_path, model_path, output_path="testresult/clean_test_result1.png"):
    # 自动创建输出目录
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = PatchCornerRegressor(backbone='resnet18', pretrained=False).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. 加载图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    h, w = img.shape[:2]
    vis_img = img.copy()

    # 3. 自动设定“粗略搜寻区域”
    # 在 4:3 或 16:9 的图里，棋盘角通常分布在距边缘 10%-20% 的位置
    # 这里我们定义 4 个锚点作为输入的“第一直觉”
    margin_w = int(w * 0.15)
    margin_h = int(h * 0.15)
    
    rough_pts = [
        [margin_w, margin_h],         # 预期 TL
        [w - margin_w, margin_h],     # 预期 TR
        [w - margin_w, h - margin_h], # 预期 BR
        [margin_w, h - margin_h]      # 预期 BL
    ]

    # 4. 预处理变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Processing image: {image_path} ({w}x{h})")
    PATCH_SIZE = 128
    half = PATCH_SIZE // 2

    for i, (rx, ry) in enumerate(rough_pts):
        # 裁剪 Patch
        x1, y1 = int(rx - half), int(ry - half)
        x2, y2 = x1 + PATCH_SIZE, y1 + PATCH_SIZE
        
        # 提取 Patch (带边界填充)
        crop = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        c_x1, c_y1 = max(0, x1), max(0, y1)
        c_x2, c_y2 = min(w, x2), min(h, y2)
        if c_x2 > c_x1 and c_y2 > c_y1:
            crop[c_y1-y1:c_y2-y1, c_x1-x1:c_x2-x1] = img[c_y1:c_y2, c_x1:c_x2]
        
        # 模型推理
        input_t = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_t).cpu().numpy()[0]
        
        # 映射回原图坐标
        fx = x1 + pred[0] * PATCH_SIZE
        fy = y1 + pred[1] * PATCH_SIZE

        # 绘图：黄色是搜索区域，绿色是模型锁定的角点
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 1) # 黄色框
        cv2.drawMarker(vis_img, (int(fx), int(fy)), (0, 255, 0), cv2.MARKER_CROSS, 25, 2)
        
        # 打印一下结果
        names = ['TL', 'TR', 'BR', 'BL']
        print(f"Detected {names[i]} at: ({int(fx)}, {int(fy)})")

    cv2.imwrite(output_path, vis_img)
    print(f"\nSaved visualization to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="要测试的图片路径")
    parser.add_argument("--model", default="checkpoints/best_patch_model_joint.pth", help="模型路径")
    args = parser.parse_args()

    if args.img:
        test_unseen_image(args.img, args.model)
    else:
        # 如果你没指定图片，我默认随便找一张现有的但在脚本里不读 yml
        test_unseen_image(r"E:\Data\WeiqiPics\IMG20160706171004.jpg", args.model)
