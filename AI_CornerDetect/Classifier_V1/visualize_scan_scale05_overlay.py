import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model import PatchClassifier

def visualize_scan_scale05_overlay():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 路径配置
    INPUT_DIR = Path(r"E:\Data\WeiqiPics\Test1")
    OUTPUT_DIR = Path(r"E:\Data\WeiqiPics\Test1\output_scale05_overlay")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = PatchClassifier(num_classes=4).to(device)
    model_path = Path("weights/best_classifier.pth")
    if not model_path.exists():
        print(f"Model weights not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 参数设置：按您的策略，取 256x256 的区域，resize 到 128x128
    crop_size = 256
    target_size = 128
    stride = 128 
    half_crop = crop_size // 2

    # ImageNet 归一化参数
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    img_files = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png"))
    print(f"Scanning with Crop={crop_size} (Scale 0.5) and Overlay for {len(img_files)} images.")

    for img_path in tqdm(img_files):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        
        # 创建一个覆盖层用于半透明方块
        overlay = img_bgr.copy()
        display_img = img_bgr.copy()
        h, w = img_bgr.shape[:2]
        
        # 遍历全图
        for y in range(half_crop, h - half_crop, stride):
            for x in range(half_crop, w - half_crop, stride):
                # 1. 截区域
                x1, y1 = x - half_crop, y - half_crop
                x2, y2 = x + half_crop, y + half_crop
                patch_large = img_bgr[y1:y2, x1:x2]
                
                # 2. Resize
                patch_resized = cv2.resize(patch_large, (target_size, target_size), interpolation=cv2.INTER_AREA)
                
                # 3. 预处理与判断
                patch_rgb = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)
                patch_tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).float() / 255.0
                patch_tensor = (patch_tensor.to(device) - mean) / std
                
                with torch.no_grad():
                    output = model(patch_tensor.unsqueeze(0))
                    probs = torch.softmax(output, dim=1)
                    conf, pred_id = torch.max(probs, 1)
                
                conf = conf.item()
                pred_id = pred_id.item()
                
                # 4. 绘图标记
                if conf > 0.8:
                    if pred_id == 0: # Corner
                        # 浅紫色方块 (BGR: 255, 0, 255 是紫色，浅紫色稍微调淡)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 255), -1)
                        cv2.putText(display_img, "C", (x - 20, y + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 255), 5)
                    elif pred_id == 1: # Edge
                        # 浅绿色方块 (BGR: 0, 255, 0 是绿色，浅绿色)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (144, 238, 144), -1)
                        cv2.putText(display_img, "E", (x - 20, y + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 0), 3)

        # 混合原图与覆盖层，实现半透明效果 (alpha=0.3)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)

        # 保存结果
        save_path = OUTPUT_DIR / f"scan_overlay_s05_{img_path.name}"
        cv2.imwrite(str(save_path), display_img)

    print(f"\nOverlay scan complete! Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_scan_scale05_overlay()
