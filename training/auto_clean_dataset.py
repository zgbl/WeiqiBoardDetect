import os
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

def calculate_brightness(path):
    try:
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB"))
            return np.mean(arr) # Average pixel value (0-255)
    except:
        return None

def auto_clean(suspects_csv, target_root, confidence_threshold=0.9):
    """
    Cleans a dataset based on:
    1. Model suspects (prediction != target AND confidence > threshold)
    2. Physical heuristics (Label=W must be bright, Label=B must be dark)
    """
    df = pd.read_csv(suspects_csv)
    target_root = Path(target_root)
    
    # 1. Create Dirty Folders
    dirty_model_dir = target_root / "DIRTY_MODEL_CONFIRMED"
    dirty_phys_dir = target_root / "DIRTY_HEURISTIC_FAIL"
    dirty_model_dir.mkdir(exist_ok=True)
    dirty_phys_dir.mkdir(exist_ok=True)

    print(f"[CLEAN] Processing {len(df)} suspects from {suspects_csv}...")
    
    model_count = 0
    phys_count = 0
    
    # 2. Process Suspects Category (Model Driven)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning Model Suspects"):
        path = Path(row['path'])
        if not path.exists():
            continue
            
        conf = float(row['confidence'])
        if conf >= confidence_threshold:
            # Move to model-confirmed dirty folder
            target_path = dirty_model_dir / f"{row['target']}_to_{row['pred']}_{path.name}"
            shutil.move(str(path), str(target_path))
            model_count += 1

    # 3. Process Heuristics Category (Dataset-wide Physical Scan)
    # Scan all patches in B and W folders for obvious brightness mismatches
    print("\n[CLEAN] Running Physical Heuristic Scan (Brightness/Mean)...")
    for cls in ["B", "W"]:
        cls_dir = target_root / cls
        if not cls_dir.exists(): continue
        
        files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        for f_path in tqdm(files, desc=f"Scanning {cls}", leave=False):
            mean_v = calculate_brightness(f_path)
            if mean_v is None: continue
            
            is_dirty = False
            reason = ""
            
            if cls == "W" and mean_v < 110: # White stone but too dark (threshold from ChatGPT/common sense)
                is_dirty = True
                reason = f"W_too_dark_{int(mean_v)}"
            elif cls == "B" and mean_v > 150: # Black stone but too bright (glare/empty)
                is_dirty = True
                reason = f"B_too_bright_{int(mean_v)}"
            
            if is_dirty:
                target_path = dirty_phys_dir / f"{reason}_{f_path.name}"
                shutil.move(str(f_path), str(target_path))
                phys_count += 1

    print(f"\n[DONE] Cleaning Complete.")
    print(f"  - Model Confirmed Dirty: {model_count} (Confidence > {confidence_threshold})")
    print(f"  - Physical Heuristic Dirty: {phys_count}")
    print(f"  - Files moved to: {dirty_model_dir} and {dirty_phys_dir}")

if __name__ == "__main__":
    # Settings for Dataset2
    suspects_path = r"D:\Codes\WeiqiBoardDetect\WeiqiBoardDetect-main\training\checkpoints\suspects_latest.csv"
    data_root = r"D:\Codes\Data\Gomrade\dataset2\patches"
    
    if os.path.exists(suspects_path):
        auto_clean(suspects_path, data_root)
    else:
        print(f"Error: Suspects file not found at {suspects_path}")
