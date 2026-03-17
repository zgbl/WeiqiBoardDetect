import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def get_quad_info(pts):
    # Sort points clockwise
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    
    sums = pts[:, 0] + pts[:, 1]
    tl_idx = np.argmin(sums)
    pts = np.roll(pts, -tl_idx, axis=0)
    
    tl, tr, br, bl = pts
    
    mid_t = (tl + tr) / 2
    mid_r = (tr + br) / 2
    mid_b = (br + bl) / 2
    mid_l = (bl + tl) / 2
    
    edges = [mid_t, mid_r, mid_b, mid_l]
    inner = [center]
    
    return {
        'corners': pts,
        'edges': edges,
        'inner': inner
    }

def is_inside_quad(p, quad):
    def cross_product(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for i in range(4):
        if cross_product(quad[i], quad[(i+1)%4], p) < 0:
            return False
    return True

def save_scaled_patch(img, pt, target_size, scale, save_path):
    h, w = img.shape[:2]
    # 在原图中需要裁剪的区域大小
    crop_size = int(target_size / scale)
    half = crop_size // 2
    
    x, y = int(pt[0]), int(pt[1])
    x1, y1, x2, y2 = x - half, y - half, x + half, y + half
    
    patch = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(w, x2), min(h, y2)
    
    dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
    
    if src_x2 > src_x1 and src_y2 > src_y1:
        patch[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    
    # 缩放回 target_size (默认 128x128)
    if scale != 1.0:
        patch = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
    cv2.imwrite(str(save_path), patch)

def extract_multiscale_patches(data_roots, output_base, target_size=128, scales=[0.75, 0.5]):
    output_base = Path(output_base)
    
    categories = ['corner', 'edge', 'inner', 'outer']
    for cat in categories:
        (output_base / cat).mkdir(parents=True, exist_ok=True)

    total_counts = {scale: {cat: 0 for cat in categories} for scale in scales}

    for root_path in data_roots:
        root = Path(root_path)
        if not root.exists(): continue
        sessions = [d for d in root.iterdir() if d.is_dir()]
        
        for session in tqdm(sessions, desc=f"Processing {root.name}"):
            ext_yml = session / "board_extractor_state.yml"
            if not ext_yml.exists(): continue
            
            try:
                with open(ext_yml, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if "pts_clicks" not in data: continue
                pts = np.array(data["pts_clicks"], dtype=np.float32)
                if len(pts) != 4: continue
                
                info = get_quad_info(pts)
                img_files = list(session.glob("*.png")) + list(session.glob("*.jpg"))
                
                for img_path in img_files:
                    img = cv2.imread(str(img_path))
                    if img is None: continue
                    h, w = img.shape[:2]
                    
                    for scale in scales:
                        prefix = f"s{int(scale*100)}"
                        # 1. Corner
                        for i, pt in enumerate(info['corners']):
                            save_scaled_patch(img, pt, target_size, scale, output_base / 'corner' / f"{session.name}_{img_path.stem}_{prefix}_c{i}.png")
                            total_counts[scale]['corner'] += 1
                            
                        # 2. Edge
                        for i, pt in enumerate(info['edges']):
                            save_scaled_patch(img, pt, target_size, scale, output_base / 'edge' / f"{session.name}_{img_path.stem}_{prefix}_e{i}.png")
                            total_counts[scale]['edge'] += 1
                            
                        # 3. Inner
                        for i, pt in enumerate(info['inner']):
                            save_scaled_patch(img, pt, target_size, scale, output_base / 'inner' / f"{session.name}_{img_path.stem}_{prefix}_i{i}.png")
                            total_counts[scale]['inner'] += 1
                            
                            for j in range(2):
                                r1, r2 = random.random(), random.random()
                                if r1 + r2 > 1: r1, r2 = 1-r1, 1-r2
                                p_rand = (1-r1-r2)*info['corners'][0] + r1*info['corners'][1] + r2*info['corners'][2]
                                save_scaled_patch(img, p_rand, target_size, scale, output_base / 'inner' / f"{session.name}_{img_path.stem}_{prefix}_ir{j}.png")
                                total_counts[scale]['inner'] += 1

                        # 4. Outer
                        tries = 0
                        outer_pts = 0
                        while outer_pts < 5 and tries < 50:
                            tries += 1
                            rx = random.randint(0, w-1)
                            ry = random.randint(0, h-1)
                            if not is_inside_quad([rx, ry], info['corners']):
                                save_scaled_patch(img, [rx, ry], target_size, scale, output_base / 'outer' / f"{session.name}_{img_path.stem}_{prefix}_o{outer_pts}.png")
                                total_counts[scale]['outer'] += 1
                                outer_pts += 1
                                
            except Exception as e:
                print(f"Error in {session}: {e}")

    print("\nExtraction complete!")
    for scale in scales:
        print(f"Scale {scale}:")
        for cat, count in total_counts[scale].items():
            print(f"  {cat}: {count} patches")

if __name__ == "__main__":
    DATA_ROOTS = [r"E:\Data\Gomrade\kaggle-gomrade\dataset1", r"E:\Data\Gomrade\kaggle-gomrade\dataset2"]
    OUTPUT_DIR = r"E:\Data\Gomrade\MultiClassPatches"
    
    # 提取多尺度，并且输出到同样的目录，方便一起训练
    # 依然生成 128x128 的图片，但是截取的原图视野更广
    extract_multiscale_patches(DATA_ROOTS, OUTPUT_DIR, target_size=128, scales=[0.75, 0.5])
