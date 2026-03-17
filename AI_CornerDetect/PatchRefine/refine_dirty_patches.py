import torch
import cv2
import os
import shutil
from pathlib import Path
from model_multi_class import PatchClassifier
from tqdm import tqdm

def refine_dirty():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    BASE_DIR = Path(r"E:\Data\Gomrade\MultiClassPatches")
    DIRTY_DIR = BASE_DIR / "dirty"
    OUTPUT_BASE = BASE_DIR # Move to existing categories
    
    if not DIRTY_DIR.exists():
        print(f"Error: {DIRTY_DIR} does not exist.")
        return

    # Load model
    model = PatchClassifier(num_classes=4).to(device)
    model_path = "checkpoints/best_patch_classifier.pth"
    if not Path(model_path).exists():
        print(f"Error: {model_path} not found. Train the model first.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Classes
    class_map = {0: 'corner', 1: 'edge', 2: 'inner', 3: 'outer'}
    
    # Process
    files = list(DIRTY_DIR.glob("*.png")) + list(DIRTY_DIR.glob("*.jpg"))
    print(f"Found {len(files)} dirty patches.")

    moved_count = {name: 0 for name in class_map.values()}

    with torch.no_grad():
        for img_path in tqdm(files):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            
            # Preprocess
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            
            # Normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
            # Predict
            output = model(img_tensor.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            conf, pred_id = torch.max(probs, 1)
            
            conf = conf.item()
            pred_id = pred_id.item()
            
            # Threshold for moving (only move if confident)
            if conf > 0.8:
                target_cat = class_map[pred_id]
                target_path = OUTPUT_BASE / target_cat / img_path.name
                
                # Ensure no overwrite or suffix if needed
                if not target_path.exists():
                    shutil.move(str(img_path), str(target_path))
                    moved_count[target_cat] += 1
                else:
                    # Rename if exists
                    new_name = f"refined_{img_path.name}"
                    shutil.move(str(img_path), str(OUTPUT_BASE / target_cat / new_name))
                    moved_count[target_cat] += 1

    print("\nRefinement Complete!")
    for cat, count in moved_count.items():
        print(f"  Moved to {cat}: {count}")

if __name__ == "__main__":
    refine_dirty()
