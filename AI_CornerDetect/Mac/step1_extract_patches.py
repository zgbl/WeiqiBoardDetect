import cv2
import os
import numpy as np

# ========== 参数设置 ==========
INPUT_DIR = "/Users/tuxy/Codes/AI/Data/EmptyBoard"
OUTPUT_DIR = "/Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/debug_extracted_patches"
PATCH_SIZE = 128
STRIDE = 64 # 有重叠地切片
# =============================

image_exts = (".png", ".jpg", ".jpeg")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[*] Created folder: {OUTPUT_DIR}")

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_exts)]
    print(f"[*] Found {len(files)} images in {INPUT_DIR}")

    total_count = 0
    for fname in files:
        fpath = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        print(f"  - Processing {fname} ({w}x{h})")
        
        count = 0
        for y in range(0, h - PATCH_SIZE, STRIDE):
            for x in range(0, w - PATCH_SIZE, STRIDE):
                patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_name = f"{os.path.splitext(fname)[0]}_y{y}_x{x}.png"
                out_path = os.path.join(OUTPUT_DIR, patch_name)
                cv2.imwrite(out_path, patch)
                count += 1
        
        print(f"    -> Extracted {count} patches.")
        total_count += count

    print(f"\n[Done] Total extracted {total_count} patches in {OUTPUT_DIR}")
    print("Next step: Run step2_classify_to_folders.py")

if __name__ == "__main__":
    main()
