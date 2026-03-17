import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_unlabeled_patches(data_roots, output_dir, patch_size=128, stride=64):
    """
    由于这些真实图片没有 label，我们用滑动窗口（或随机裁剪）来切 patch
    这里采用滑动窗口方式，以便覆盖全图进行测试
    """
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    half = patch_size // 2
    total_patches = 0

    for root_path in data_roots:
        root = Path(root_path)
        if not root.exists():
            print(f"Warning: Data root {root} does not exist. Skipping.")
            continue
            
        img_files = list(root.glob("*.jpg")) + list(root.glob("*.png")) + list(root.glob("*.jpeg"))
        
        for img_path in tqdm(img_files, desc=f"Processing {root.name}"):
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            h, w = img.shape[:2]
            
            # 使用滑动窗口切整个图片
            # stride 越小能切出的图越多。这里默认为 patch_size 的一半
            patch_count_for_img = 0
            for y in range(half, h - half, stride):
                for x in range(half, w - half, stride):
                    x1, y1 = x - half, y - half
                    x2, y2 = x + half, y + half
                    
                    patch = img[y1:y2, x1:x2]
                    
                    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                        continue # 边缘可能会有形状不对的，直接跳过
                        
                    save_name = f"{img_path.stem}_p_{patch_count_for_img}.png"
                    save_path = output_base / save_name
                    cv2.imwrite(str(save_path), patch)
                    
                    total_patches += 1
                    patch_count_for_img += 1

    print(f"\nExtraction complete!")
    print(f"Total {total_patches} patches saved to {output_base}")

if __name__ == "__main__":
    DATA_ROOTS = [
        r"E:\Data\WeiqiPics\Set1", 
        r"E:\Data\WeiqiPics\Set2"
    ]
    OUTPUT_DIR = r"E:\Data\WeiqiPics\patches"
    
    # 尺寸必须和训练时保持绝对一致 (128x128)
    extract_unlabeled_patches(DATA_ROOTS, OUTPUT_DIR, patch_size=128, stride=64)
