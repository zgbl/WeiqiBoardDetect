import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import shutil
import cv2
import sys
import numpy as np

# ========== 参数设置 ==========
PATCH_DIR = "/Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/debug_extracted_patches"
# 您也可以根据需要手动修改这里的权重路径和类别定义
WEIGHTS_PATH = "/Users/tuxy/Codes/AI/WeiqiBoardDetect/data/Weights/4Classes-V3.pth"
# 索引映射 (基于之前的推断：1: Inner, 2: Edge)
CLASSES = ['Corner', 'Inner', 'Edge', 'Outer'] 
# =============================

# 如果 V3 出现大范围识别错误，说明映射还是不对，这时请尝试交换这里：
# CLASSES = ['Corner', 'Edge', 'Inner', 'Outer'] # 交换后试试

def main():
    # 动态导入模型
    MAC_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(MAC_DIR)
    sys.path.append(PARENT_DIR)
    
    # 手动添加 Classifier_V1 路径到 Python
    CLASSIFIER_V1_DIR = os.path.join(PARENT_DIR, "Classifier_V1")
    if str(CLASSIFIER_V1_DIR) not in sys.path:
        sys.path.insert(0, str(CLASSIFIER_V1_DIR))
    
    try:
        from model import PatchClassifier
    except ImportError:
        print("[Error] Could not find PatchClassifier in Classifier_V1/model.py")
        return

    # 1. 加载模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Running on device: {device}")
    
    model = PatchClassifier(num_classes=4)
    model.to(device)

    # 稳健加载
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
    model_keys = model.state_dict().keys()
    new_sd = {}
    for k, v in state_dict.items():
        if k in model_keys: new_sd[k] = v
        elif "backbone." + k in model_keys: new_sd["backbone." + k] = v
        elif k.replace("backbone.", "") in model_keys: new_sd[k.replace("backbone.", "")] = v
    
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    print(f"[*] Loaded weights from {WEIGHTS_PATH}")

    # 2. 预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 创建输出子文件夹
    for cname in CLASSES:
        path = os.path.join(PATCH_DIR, cname)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"[+] Created: {path}")

    # 4. 开始分类
    files = [f for f in os.listdir(PATCH_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"[*] Found {len(files)} patches to classify.")

    count = 0
    with torch.no_grad():
        for fname in files:
            fpath = os.path.join(PATCH_DIR, fname)
            try:
                img_pil = Image.open(fpath).convert("RGB")
            except:
                continue
            
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            conf, pred = torch.max(probs, 0)
            
            label = CLASSES[pred.item()]
            
            # 移动到对应的类文件夹
            dst_path = os.path.join(PATCH_DIR, label, fname)
            shutil.copy2(fpath, dst_path)
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count} / {len(files)} ...")

    print(f"\n[Done] Successfully classified {count} patches.")
    print(f"Check the subfolders in {PATCH_DIR} for results.")

if __name__ == "__main__":
    main()
