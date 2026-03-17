import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from pathlib import Path
import time

# 导入我们的两个模型
from model_global import GlobalKeypointModel
from model import PatchDetector

class BoardDetectorPipeline:
    def __init__(self, global_pth, patch_pth):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 初始化全局模型
        self.global_model = GlobalKeypointModel(num_points=4).to(self.device)
        self.global_model.load_state_dict(torch.load(global_pth, map_location=self.device, weights_only=True))
        self.global_model.eval()
        
        # 2. 初始化精调模型
        self.patch_model = PatchDetector(backbone='resnet18').to(self.device)
        self.patch_model.load_state_dict(torch.load(patch_pth, map_location=self.device, weights_only=True))
        self.patch_model.eval()

        # 3. 图像变换
        self.global_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.patch_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect(self, image_path, out_vis="final_result.png"):
        img = cv2.imread(image_path)
        if img is None: return
        h_orig, w_orig = img.shape[:2]
        vis_img = img.copy()

        # --- 第一阶段: 全局粗定位 ---
        input_global = self.global_transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            heatmaps = self.global_model(input_global).cpu().numpy()[0] # (4, 56, 56)

        rough_pts = []
        for i in range(4):
            # 找到热力图最大值点
            hm = heatmaps[i]
            y_hm, x_hm = np.unravel_index(np.argmax(hm), hm.shape)
            # 映射回原图坐标 (56x56 -> 原图宽高)
            fx = x_hm / 56.0 * w_orig
            fy = y_hm / 56.0 * h_orig
            rough_pts.append((fx, fy))

        # --- 第二阶段: 局部像素精调 ---
        patch_size = 128
        half = patch_size // 2
        final_pts = []

        names = {0: 'TL', 1: 'TR', 2: 'BR', 3: 'BL'}
        colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (255, 255, 0), 3: (255, 0, 255)}

        print(f"\nFinal Detection Results for {Path(image_path).name}:")
        for i, (rx, ry) in enumerate(rough_pts):
            # 裁剪 Patch
            x1, y1 = int(rx - half), int(ry - half)
            crop = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            
            # 安全裁剪
            c_x1, c_y1 = max(0, x1), max(0, y1)
            c_x2, c_y2 = min(w_orig, x1 + patch_size), min(h_orig, y1 + patch_size)
            if c_x2 > c_x1 and c_y2 > c_y1:
                crop[c_y1-y1:c_y2-y1, c_x1-x1:c_x2-x1] = img[c_y1:c_y2, c_x1:c_x2]
            
            # 精调模型推理
            input_patch = self.patch_transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                cls_out, reg_out = self.patch_model(input_patch)
                # 其实此时分类已经不那么重要了，因为全局模型已经帮我们“看到”角点了
                # 但我们可以用来过滤掉太离谱的误报
                coords = reg_out.cpu().numpy()[0]
            
            # 计算最终坐标
            fx = x1 + coords[0] * patch_size
            fy = y1 + coords[1] * patch_size
            final_pts.append((fx, fy))

            # 可视化
            # 画粗定位框 (蓝色)
            cv2.rectangle(vis_img, (x1, y1), (x1+patch_size, y1+patch_size), (255, 0, 0), 2)
            # 画最终十字 (鲜艳色)
            cv2.drawMarker(vis_img, (int(fx), int(fy)), colors[i], cv2.MARKER_CROSS, 40, 3)
            print(f"[{names[i]}] -> ({int(fx)}, {int(fy)})")

        os.makedirs(os.path.dirname(out_vis), exist_ok=True)
        cv2.imwrite(out_vis, vis_img)
        print(f"Saved visualization to: {out_vis}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to image")
    args = parser.parse_args()

    pipeline = BoardDetectorPipeline(
        global_pth="checkpoints/best_global_model.pth", 
        patch_pth="checkpoints/best_patch_detector.pth"
    )
    
    pipeline.detect(args.img, out_vis="testresult/pipeline_final.png")
