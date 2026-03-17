import torch
import cv2
import numpy as np
import shutil
from pathlib import Path
from model import PatchClassifier
from tqdm import tqdm

def debug_sort_raw_patches():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 路径配置
    TARGET_IMG_PATH = Path(r"E:\Data\WeiqiPics\Test1\IMG20170715153712.jpg")
    OUTPUT_BASE = Path(r"E:\Data\WeiqiPics\Test1\output2")
    
    # 确保输出目录干净且存在分类子目录
    categories = ['corner', 'edge', 'inner', 'outer', 'unsure']
    if OUTPUT_BASE.exists():
        shutil.rmtree(OUTPUT_BASE)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    for cat in categories:
        (OUTPUT_BASE / cat).mkdir(exist_ok=True)

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

    # 类别映射词典
    class_name_map = {0: 'corner', 1: 'edge', 2: 'inner', 3: 'outer'}

    # ImageNet 归一化参数
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    img_bgr = cv2.imread(str(TARGET_IMG_PATH))
    if img_bgr is None:
        print("Failed to load image.")
        return
    
    h, w = img_bgr.shape[:2]
    print(f"Slicing and sorting patches for {TARGET_IMG_PATH.name}...")

    patch_idx = 0
    # 遍历全图切片
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            x1, y1 = x - half, y - half
            x2, y2 = x + half, y + half
            patch_bgr = img_bgr[y1:y2, x1:x2].copy()
            
            # 这里存的是没有任何标记的干净图
            # 预处理仅用于模型判断
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            patch_tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).float() / 255.0
            patch_tensor = (patch_tensor.to(device) - mean) / std
            
            with torch.no_grad():
                output = model(patch_tensor.unsqueeze(0))
                probs = torch.softmax(output, dim=1)
                conf, pred_id = torch.max(probs, 1)
            
            conf_val = conf.item()
            
            # 判断逻辑子目录
            if conf_val > 0.7: # 门槛稍降，尽可能细分
                dest_cat = class_name_map[pred_id.item()]
            else:
                dest_cat = 'unsure'
            
            # 保存原图到对应子目录
            save_name = f"p_{patch_idx:04d}_conf_{conf_val:.2f}.png"
            cv2.imwrite(str(OUTPUT_BASE / dest_cat / save_name), patch_bgr)
            patch_idx += 1

    print(f"\nCompleted! Total {patch_idx} raw patches sorted into {OUTPUT_BASE}")

if __name__ == "__main__":
    debug_sort_raw_patches()
