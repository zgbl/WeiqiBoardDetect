import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model import PatchClassifier

def visualize_scan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 路径配置
    INPUT_DIR = Path(r"E:\Data\WeiqiPics\Test1")
    OUTPUT_DIR = Path(r"E:\Data\WeiqiPics\Test1\output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    stride = 64 # 步长，越小越精细但运行越慢
    half = patch_size // 2

    # ImageNet 归一化参数
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    img_files = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png"))
    print(f"Found {len(img_files)} images to scan.")

    for img_path in tqdm(img_files):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        
        display_img = img_bgr.copy()
        h, w = img_bgr.shape[:2]
        
        # 准备数据批次以提高效率（可选，这里用简单循环）
        for y in range(half, h - half, stride):
            for x in range(half, w - half, stride):
                # 裁剪 Patch
                x1, y1 = x - half, y - half
                x2, y2 = x + half, y + half
                patch_bgr = img_bgr[y1:y2, x1:x2]
                
                # 预处理
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                patch_tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).float() / 255.0
                patch_tensor = (patch_tensor.to(device) - mean) / std
                
                with torch.no_grad():
                    output = model(patch_tensor.unsqueeze(0))
                    probs = torch.softmax(output, dim=1)
                    conf, pred_id = torch.max(probs, 1)
                
                conf = conf.item()
                pred_id = pred_id.item()
                
                # 只标出高置信度的识别结果 (门槛 0.8)
                if conf > 0.8:
                    if pred_id == 0: # Corner
                        cv2.putText(display_img, "C", (x - 10, y + 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3) # 红色 C
                    elif pred_id == 1: # Edge
                        cv2.putText(display_img, "E", (x - 10, y + 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # 绿色 E

        # 保存结果
        save_path = OUTPUT_DIR / f"scan_{img_path.name}"
        cv2.imwrite(str(save_path), display_img)

    print(f"\nVisualization complete! Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_scan()
