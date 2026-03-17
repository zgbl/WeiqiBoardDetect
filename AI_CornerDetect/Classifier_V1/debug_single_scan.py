import torch
import cv2
import numpy as np
from pathlib import Path
from model import PatchClassifier
from tqdm import tqdm

def debug_single_image_scan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 路径配置
    TARGET_IMG_PATH = Path(r"E:\Data\WeiqiPics\Test1\IMG20170715153712.jpg")
    OUTPUT_DIR = Path(r"E:\Data\WeiqiPics\Test1\output1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TARGET_IMG_PATH.exists():
        print(f"Error: Target image {TARGET_IMG_PATH} not found.")
        return

    # 加载模型
    model = PatchClassifier(num_classes=4).to(device)
    model_path = Path("weights/best_classifier.pth")
    if not model_path.exists():
        print(f"Model weights not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 参数设置
    patch_size = 128
    stride = 64 
    half = patch_size // 2

    # 类别映射
    class_map = {0: 'C', 1: 'E', 2: 'I', 3: 'O'}
    colors = {
        'C': (0, 0, 255),   # 红色
        'E': (0, 255, 0),   # 绿色
        'I': (255, 255, 0), # 青色
        'O': (200, 200, 200) # 灰色
    }

    # ImageNet 归一化参数
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    img_bgr = cv2.imread(str(TARGET_IMG_PATH))
    if img_bgr is None:
        print("Failed to load image.")
        return
    
    h, w = img_bgr.shape[:2]
    print(f"Scanning {TARGET_IMG_PATH.name} ({w}x{h})...")

    patch_idx = 0
    # 我们遍历全图切片
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            x1, y1 = x - half, y - half
            x2, y2 = x + half, y + half
            patch_bgr = img_bgr[y1:y2, x1:x2].copy()
            
            # 预处理
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            patch_tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).float() / 255.0
            patch_tensor = (patch_tensor.to(device) - mean) / std
            
            with torch.no_grad():
                output = model(patch_tensor.unsqueeze(0))
                probs = torch.softmax(output, dim=1)
                conf, pred_id = torch.max(probs, 1)
            
            conf_val = conf.item()
            pred_char = class_map[pred_id.item()]
            
            # 在切片正中间标上判断的字母和置信度
            label_text = f"{pred_char} ({conf_val:.2f})"
            color = colors.get(pred_char, (255, 255, 255))
            
            # 画一个半透明层背景让文字更清晰
            overlay = patch_bgr.copy()
            cv2.rectangle(overlay, (5, 5), (120, 35), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, patch_bgr, 0.5, 0, patch_bgr)
            
            cv2.putText(patch_bgr, label_text, (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 保存该切片
            save_name = f"patch_{patch_idx:04d}_{pred_char}_{conf_val:.2f}.png"
            cv2.imwrite(str(OUTPUT_DIR / save_name), patch_bgr)
            patch_idx += 1

    print(f"\nDone! Extracted and labeled {patch_idx} patches to: {OUTPUT_DIR}")

if __name__ == "__main__":
    debug_single_image_scan()
