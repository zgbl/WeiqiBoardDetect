import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from pathlib import Path
import time

from model import PatchDetector

def scan_with_debug(image_path, model_path, output_dir="testresult", scan_width=1536, confidence_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载模型
    model = PatchDetector(backbone='resnet18').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. 加载与预处理图片
    orig_img = cv2.imread(image_path)
    if orig_img is None: return
    
    # --- 重要：为了匹配 Patch 大小，我们先缩放图片 ---
    h_orig, w_orig = orig_img.shape[:2]
    scale = scan_width / w_orig
    scan_h = int(h_orig * scale)
    img = cv2.resize(orig_img, (scan_width, scan_h))
    
    # 准备一张热力图 (5层，对应 BG, TL, TR, BR, BL)
    heatmap = np.zeros((scan_h, scan_width, 5), dtype=np.float32)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    PATCH_SIZE = 128
    STRIDE = 32 # 缩小步长以获得更细腻的热力图
    
    print(f"Scanning resized image {scan_width}x{scan_h} (Scale: {scale:.2f})...")
    
    # 为了速度，我们这里每隔一段打印一次
    total_steps = ((scan_h - PATCH_SIZE)//STRIDE + 1) * ((scan_width - PATCH_SIZE)//STRIDE + 1)
    step = 0
    
    with torch.no_grad():
        for y in range(0, scan_h - PATCH_SIZE, STRIDE):
            for x in range(0, scan_width - PATCH_SIZE, STRIDE):
                crop = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                input_t = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                
                cls_out, _ = model(input_t)
                probs = torch.softmax(cls_out, dim=1)[0].cpu().numpy()
                
                # 在热力图对应区域填充得分 (取最大值或累加)
                heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = np.maximum(heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE], probs)
                
                step += 1
                if step % 500 == 0:
                    print(f"Progress: {step}/{total_steps}...")

    # 3. 后处理与可视化
    names = {1: 'TL', 2: 'TR', 3: 'BR', 4: 'BL'}
    # 颜色：绿色，黄色，青色，紫色
    colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (255, 255, 0), 4: (255, 0, 255)}
    vis_img = img.copy()
    
    # 创建一张全类别的混合热力图
    combined_heatmap = np.max(heatmap[:, :, 1:], axis=2)
    heatmap_vis = (combined_heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    # 将热力图叠加到原图上 (50% 透明度)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    
    print("\n--- Summary of Best Candidates (Top 1 for each) ---")
    for cid in range(1, 5):
        channel = heatmap[:, :, cid]
        max_val = np.max(channel)
        y_max, x_max = np.unravel_index(np.argmax(channel), channel.shape)
        
        print(f"[{names[cid]}] Max Conf: {max_val:.4f} at ({x_max}, {y_max})")
        
        # 无论分数高低，都画出来方便排查
        # 画在原图上
        cv2.drawMarker(vis_img, (x_max, y_max), colors[cid], cv2.MARKER_CROSS, 30, 2)
        cv2.putText(vis_img, f"{names[cid]}:{max_val:.2f}", (x_max, y_max-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[cid], 2)
        # 同时也画在叠加图上
        cv2.drawMarker(overlay, (x_max, y_max), (255, 255, 255), cv2.MARKER_STAR, 20, 1)

    # 保存最终大合集
    cv2.imwrite(f"{output_dir}/scan_debug_result.png", vis_img)
    cv2.imwrite(f"{output_dir}/scan_overlay_heatmap.png", overlay)
    
    print(f"\nSaved 'scan_debug_result.png' (Markers on original)")
    print(f"Saved 'scan_overlay_heatmap.png' (Heatmap overlay)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--model", default="checkpoints/best_patch_detector.pth")
    parser.add_argument("--width", type=int, default=1280, help="Resize image to this width for scanning")
    args = parser.parse_args()

    scan_with_debug(args.img, args.model, scan_width=args.width)
