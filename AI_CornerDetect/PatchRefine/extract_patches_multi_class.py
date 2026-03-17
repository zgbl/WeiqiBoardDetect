import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def get_quad_info(pts):
    """
    pts: 4x2 array of [x, y]
    Returns useful points for sampling
    """
    # Sort points clockwise to ensure we know which is which (TL, TR, BR, BL)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    
    # Identify TL (min sum of x+y is a good heuristic after rotation-aware sorting)
    sums = pts[:, 0] + pts[:, 1]
    tl_idx = np.argmin(sums)
    pts = np.roll(pts, -tl_idx, axis=0)
    
    tl, tr, br, bl = pts
    
    # Midpoints of edges
    mid_t = (tl + tr) / 2
    mid_r = (tr + br) / 2
    mid_b = (br + bl) / 2
    mid_l = (bl + tl) / 2
    
    edges = [mid_t, mid_r, mid_b, mid_l]
    
    # Center
    inner = [center]
    
    return {
        'corners': pts,
        'edges': edges,
        'inner': inner
    }

def is_inside_quad(p, quad):
    """Check if point p is inside convex quadrilateral quad using cross products."""
    def cross_product(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for i in range(4):
        if cross_product(quad[i], quad[(i+1)%4], p) < 0:
            return False
    return True

def extract_multi_class_patches(data_roots, output_base, patch_size=128):
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    categories = ['corner', 'edge', 'inner', 'outer']
    for cat in categories:
        (output_base / cat).mkdir(exist_ok=True)

    half = patch_size // 2
    total_count = {cat: 0 for cat in categories}

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
                    
                    # 1. Corner Patches
                    for i, pt in enumerate(info['corners']):
                        save_patch(img, pt, patch_size, output_base / 'corner' / f"{session.name}_{img_path.stem}_c{i}.png")
                        total_count['corner'] += 1
                        
                    # 2. Edge Patches
                    for i, pt in enumerate(info['edges']):
                        save_patch(img, pt, patch_size, output_base / 'edge' / f"{session.name}_{img_path.stem}_e{i}.png")
                        total_count['edge'] += 1
                        
                    # 3. Inner Patches
                    for i, pt in enumerate(info['inner']):
                        # Center of board
                        save_patch(img, pt, patch_size, output_base / 'inner' / f"{session.name}_{img_path.stem}_i{i}.png")
                        total_count['inner'] += 1
                        # Sample 2 more random inner points
                        for j in range(2):
                            # Simple way to get inner point: weighted average
                            r1, r2 = random.random(), random.random()
                            if r1 + r2 > 1:
                                r1, r2 = 1-r1, 1-r2
                            # Triangle 1: TL, TR, BR
                            p_rand = (1-r1-r2)*info['corners'][0] + r1*info['corners'][1] + r2*info['corners'][2]
                            save_patch(img, p_rand, patch_size, output_base / 'inner' / f"{session.name}_{img_path.stem}_ir{j}.png")
                            total_count['inner'] += 1

                    # 4. Outer Patches
                    # Sample 5 points that are outside the quad but inside the image
                    tries = 0
                    outer_pts = 0
                    while outer_pts < 5 and tries < 50:
                        tries += 1
                        rx = random.randint(0, w-1)
                        ry = random.randint(0, h-1)
                        if not is_inside_quad([rx, ry], info['corners']):
                            save_patch(img, [rx, ry], patch_size, output_base / 'outer' / f"{session.name}_{img_path.stem}_o{outer_pts}.png")
                            total_count['outer'] += 1
                            outer_pts += 1
                            
            except Exception as e:
                print(f"Error in {session}: {e}")

    print("Extraction complete!")
    for cat, count in total_count.items():
        print(f"  {cat}: {count} patches")

def save_patch(img, pt, patch_size, save_path):
    h, w = img.shape[:2]
    half = patch_size // 2
    x, y = int(pt[0]), int(pt[1])
    x1, y1, x2, y2 = x - half, y - half, x + half, y + half
    
    patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(w, x2), min(h, y2)
    
    dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
    
    if src_x2 > src_x1 and src_y2 > src_y1:
        patch[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    
    cv2.imwrite(str(save_path), patch)

if __name__ == "__main__":
    DATA_ROOTS = [r"E:\Data\Gomrade\kaggle-gomrade\dataset1", r"E:\Data\Gomrade\kaggle-gomrade\dataset2"]
    OUTPUT_DIR = r"E:\Data\Gomrade\MultiClassPatches"
    extract_multi_class_patches(DATA_ROOTS, OUTPUT_DIR, 128)
