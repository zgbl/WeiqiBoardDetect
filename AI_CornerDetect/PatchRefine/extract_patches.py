import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def sort_points(pts):
    """将 4 个点排序为 [TL, TR, BR, BL]"""
    pts = pts[np.argsort(pts[:, 1]), :]
    top_two = pts[:2, :]
    bottom_two = pts[2:, :]
    tl, tr = top_two[np.argsort(top_two[:, 0]), :]
    bl, br = bottom_two[np.argsort(bottom_two[:, 0]), :]
    return {'TL': tl, 'TR': tr, 'BR': br, 'BL': bl}

def extract_corner_patches(data_roots, output_base, patch_size=128):
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    half = patch_size // 2
    total_extracted = 0

    for root_path in data_roots:
        root = Path(root_path)
        if not root.exists(): continue
        sessions = [d for d in root.iterdir() if d.is_dir()]
        for session in tqdm(sessions, desc=f"Extracting {root.name}"):
            ext_yml = session / "board_extractor_state.yml"
            if not ext_yml.exists(): continue
            try:
                with open(ext_yml, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if "pts_clicks" not in data: continue
                pts = np.array(data["pts_clicks"], dtype=np.float32)
                sorted_pts = sort_points(pts)
                img_files = list(session.glob("*.png")) + list(session.glob("*.jpg"))
                for img_path in img_files:
                    img = cv2.imread(str(img_path))
                    if img is None: continue
                    h, w = img.shape[:2]
                    for name, pt in sorted_pts.items():
                        x, y = int(pt[0]), int(pt[1])
                        x1, y1, x2, y2 = x - half, y - half, x + half, y + half
                        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                        src_x1, src_y1 = max(0, x1), max(0, y1)
                        src_x2, src_y2 = min(w, x2), min(h, y2)
                        dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
                        dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
                        if src_x2 > src_x1 and src_y2 > src_y1:
                            patch[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
                        save_path = output_base / name / f"{session.name}_{img_path.stem}_{name}.png"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(save_path), patch)
                        total_extracted += 1
            except Exception as e:
                print(f"Error {session}: {e}")
    print(f"Extracted {total_extracted} patches to {output_base}")

if __name__ == "__main__":
    DATA_ROOTS = [r"E:\Data\Gomrade\kaggle-gomrade\dataset1", r"E:\Data\Gomrade\kaggle-gomrade\dataset2"]
    OUTPUT_DIR = r"E:\Data\Gomrade\Corners"
    extract_corner_patches(DATA_ROOTS, OUTPUT_DIR, 128)
