import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from pathlib import Path
import time

from model_global import GlobalKeypointModel
from model import PatchDetector

class WeiqiBoardPipeline:
    def __init__(self, global_pth, patch_pth):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Models onto {self.device}...")
        
        self.global_model = GlobalKeypointModel(num_points=4).to(self.device)
        self.global_model.load_state_dict(torch.load(global_pth, map_location=self.device, weights_only=True))
        self.global_model.eval()
        
        self.patch_model = PatchDetector(backbone='resnet18').to(self.device)
        self.patch_model.load_state_dict(torch.load(patch_pth, map_location=self.device, weights_only=True))
        self.patch_model.eval()

        self.patch_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect(self, image_path, out_dir="testresult"):
        os.makedirs(out_dir, exist_ok=True)
        # --- 优化1：限制读入内存的大小，防止卡死 ---
        # 我们先读一张中等大小的图处理
        img_full = cv2.imread(image_path)
        if img_full is None: return
        h_orig, w_orig = img_full.shape[:2]
        
        # 2. 全局粗定位 (缩小型)
        scan_size = 224
        img_small = cv2.resize(img_full, (scan_size, scan_size))
        input_t = transforms.ToTensor()(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            heatmaps = self.global_model(input_t).cpu().numpy()[0] # (4, 56, 56)

        # --- 诊断2：保存并展示热力图本身 (关键！) ---
        hm_vis = np.max(heatmaps, axis=0) # 把4个通道合在一起看
        hm_vis = (hm_vis / (np.max(hm_vis) + 1e-9) * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
        cv2.imwrite(f"{out_dir}/GLOBAL_HEATMAP_DEBUG.png", hm_color)

        # 3. 提取 4 个点
        debug_canvas = np.zeros((128*4, 128*2, 3), dtype=np.uint8)
        names = ['TL', 'TR', 'BR', 'BL']
        
        for i in range(4):
            hm = heatmaps[i]
            max_conf = np.max(hm)
            y_hm, x_hm = np.unravel_index(np.argmax(hm), hm.shape)
            
            # 还原坐标
            rx, ry = x_hm / 56.0 * w_orig, y_hm / 56.0 * h_orig
            print(f"[{names[i]}] Global Conf: {max_conf:.4f} at ({int(rx)}, {int(ry)})")

            # 抠图 128x128 诊断
            patch_size = 128
            half = patch_size // 2
            x1, y1 = int(rx - half), int(ry - half)
            crop = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w_orig, x1+patch_size), min(h_orig, y1+patch_size)
            if cx2 > cx1 and cy2 > cy1:
                crop[cy1-y1:cy2-y1, cx1-x1:cx2-x1] = img_full[cy1:cy2, cx1:cx2]
            
            # 模型二次精调 (仅当有信号时)
            input_p = self.patch_transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, reg_out = self.patch_model(input_p)
                coords = reg_out.cpu().numpy()[0]
            
            # 保存到诊断拼图
            v128 = crop.copy()
            cv2.drawMarker(v128, (int(coords[0]*128), int(coords[1]*128)), (0,0,255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(v128, f"Conf:{max_conf:.4f}", (5, 20), 0, 0.5, (0,255,0), 1)
            debug_canvas[i*128:(i+1)*128, 0:128] = v128
            
        cv2.imwrite(f"{out_dir}/FINAL_DIAGNOSE.png", debug_canvas)
        print(f"Check {out_dir}/GLOBAL_HEATMAP_DEBUG.png to see if the model is ALIVE.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    args = parser.parse_args()
    pipeline = WeiqiBoardPipeline("checkpoints/best_global_model_robust.pth", "checkpoints/best_patch_detector.pth")
    pipeline.detect(args.img)
