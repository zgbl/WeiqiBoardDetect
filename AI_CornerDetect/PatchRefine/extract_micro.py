import os
import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

def extract_micro_dataset(root_dirs, output_dir=r"E:\Data\Gomrade\MicroCorners"):
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    print("Extracting 64x64 Micro Patches...")
    
    for root in root_dirs:
        root = Path(root)
        if not root.exists(): continue
        for session in root.iterdir():
            if not session.is_dir(): continue
            ext_yml = session / "board_extractor_state.yml"
            if not ext_yml.exists(): continue
            
            with open(ext_yml, 'r') as f:
                data = yaml.safe_load(f)
            pts = np.array(data['pts_clicks'], dtype=np.float32)

            for img_path in session.glob("*.png"):
                img = cv2.imread(str(img_path))
                if img is None: continue
                h, w = img.shape[:2]
                
                for i, (px, py) in enumerate(pts):
                    # 关键：我们只抠 64x64 的极小范围
                    size = 64
                    half = size // 2
                    
                    # 此时为了训练模型，我们故意在真值周围加一点点随机抖动 (-5 到 5 像素)
                    # 这样模型才能学会“如何把偏了的点拉回来”
                    jitter_x = px + np.random.randint(-10, 10)
                    jitter_y = py + np.random.randint(-10, 10)
                    
                    x1, y1 = int(jitter_x - half), int(jitter_y - half)
                    crop = np.zeros((size, size, 3), dtype=np.uint8)
                    
                    # 裁剪
                    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x1+size), min(h, y1+size)
                    if cx2 > cx1 and cy2 > cy1:
                        crop[cy1-y1:cy2-y1, cx1-x1:cx2-x1] = img[cy1:cy2, cx1:cx2]
                    
                    # 计算相对于这个 64x64 Patch 的真实坐标 (Label)
                    target_x = (px - x1) / size
                    target_y = (py - y1) / size
                    
                    name = f"micro_{count}_{i}.png"
                    cv2.imwrite(str(output_dir / name), crop)
                    
                    # 保存 label: 文件名 x y
                    with open(output_dir / "labels.txt", "a") as f:
                        f.write(f"{name} {target_x} {target_y}\n")
                
                count += 1
    print(f"Done. Extracted {count*4} micro patches to {output_dir}")

if __name__ == "__main__":
    DATA_DIRS = [
        r"E:\Data\Gomrade\kaggle-gomrade\dataset1",
        r"E:\Data\Gomrade\kaggle-gomrade\dataset2"
    ]
    extract_micro_dataset(DATA_DIRS)
