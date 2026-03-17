import cv2
import numpy as np
from pathlib import Path
import random

def generate_background_noise(size=128):
    """生成完全没有角点的干扰图"""
    r_base, g_base, b_base = random.randint(180, 240), random.randint(140, 200), random.randint(80, 160)
    img = np.full((size, size, 3), [b_base, g_base, r_base], dtype=np.uint8)
    
    # 模拟木纹
    for _ in range(8):
        c = random.randint(-15, 15)
        color = [max(0, b_base+c), max(0, g_base+c), max(0, r_base+c)]
        p1 = (random.randint(-size, size), random.randint(-size, size))
        p2 = (p1[0] + size*2, p1[1] + random.randint(-10, 10))
        cv2.line(img, p1, p2, color, random.randint(1, 2), cv2.LINE_AA)

    noise_type = random.choice(['clean', 'line', 'cross'])
    if noise_type == 'line': # 画一根穿过的直线 (不是角)
        cv2.line(img, (0, random.randint(0, size)), (size, random.randint(0, size)), (0,0,0), random.randint(1,3))
    elif noise_type == 'cross': # 画一个十字交叉 (让你担心的干扰项)
        cx, cy = random.randint(20, 108), random.randint(20, 108)
        cv2.line(img, (0, cy), (size, cy), (0,0,0), random.randint(1,3))
        cv2.line(img, (cx, 0), (cx, size), (0,0,0), random.randint(1,3))
        
    return img

def generate_l_corner(size=128, corner_type='TL'):
    """生成真实的 L 角点"""
    r_base, g_base, b_base = random.randint(180, 240), random.randint(140, 200), random.randint(80, 160)
    img = np.full((size, size, 3), [b_base, g_base, r_base], dtype=np.uint8)
    # 木纹... (同上)
    cx, cy = np.random.randint(size//4, 3*size//4, size=2)
    directions = {'TL': [(1, 0), (0, 1)], 'TR': [(-1, 0), (0, 1)], 'BR': [(-1, 0), (0, -1)], 'BL': [(1, 0), (0, -1)]}
    for dx, dy in directions[corner_type]:
        cv2.line(img, (cx, cy), (int(cx + size*2*dx), int(cy + size*2*dy)), (0,0,0), random.randint(2,4), cv2.LINE_AA)
    return img, (cx, cy)

def create_advanced_dataset(output_dir, count=5000):
    output_dir = Path(output_dir)
    if output_dir.exists(): import shutil; shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    label_file = output_dir / "labels.txt"
    with open(label_file, "w") as f:
        # 1. 生成正样本 (4种角)
        for i in range(count):
            ctype = random.choice(['TL', 'TR', 'BR', 'BL'])
            img, (x, y) = generate_l_corner(corner_type=ctype)
            name = f"pos_{ctype}_{i}.png"
            cv2.imwrite(str(output_dir/name), img)
            # 格式: 文件名 类别ID(1-4) x y
            cid = {'TL':1, 'TR':2, 'BR':3, 'BL':4}[ctype]
            f.write(f"{name} {cid} {x} {y}\n")
            
        # 2. 生成大量负样本 (类别ID为 0)
        for i in range(count):
            img = generate_background_noise()
            name = f"neg_{i}.png"
            cv2.imwrite(str(output_dir/name), img)
            f.write(f"{name} 0 0 0\n")

if __name__ == "__main__":
    create_advanced_dataset(r"E:\Data\Gomrade\Corners\Synthetic")
