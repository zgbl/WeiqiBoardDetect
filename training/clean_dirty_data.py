import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms

# Import from training scripts
try:
    from train_win import build_model, detect_device
except ImportError:
    print("[ERROR] Could not import from train_win.py. Ensure you are running this in the training/ directory.")
    exit(1)

IDX_TO_CLASS = {0: "B", 1: "W", 2: "E"}

def get_transforms(patch_size=48):
    return transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def main():
    parser = argparse.ArgumentParser(description="Clean patches directly using inference (No RAM preload, SSD optimized).")
    parser.add_argument("--model",   default=r"D:\Codes\WeiqiBoardDetect\WeiqiBoardDetect-main\training\latest_model.pt", help="Path to the model checkpoint")
    parser.add_argument("--data",    default=r"E:\Data\Gomrade\patches", help="Path to the patches root directory (containing B, W, E)")
    parser.add_argument("--arch",    default="MobileNetV3", help="Model architecture")
    parser.add_argument("--size",    type=int, default=48, help="Patch size")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be moved without moving")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {args.data}")
        return

    # Load Model
    device = detect_device()
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading model weights from: {args.model}")
    
    model = build_model(args.arch, num_classes=3, patch_size=args.size).to(device)
    try:
        ckpt = torch.load(args.model, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
        
    model.eval()
    val_transforms = get_transforms(args.size)

    total_moved = 0

    # Process each class directory directly, one file at a time
    with torch.no_grad():
        for true_class in ["B", "W", "E"]:
            cls_dir = data_dir / true_class
            if not cls_dir.exists():
                print(f"[WARNING] Class directory not found: {cls_dir}")
                continue
                
            files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            if not files:
                continue
                
            print(f"\n[INFO] Scanning directory: {cls_dir} ({len(files)} patches)")
            moved_this_class = 0
            
            for file_path in tqdm(files, desc=f"Evaluating {true_class}"):
                try:
                    # Load completely into memory to free the file handle (prevent WinError 32)
                    with open(file_path, 'rb') as f:
                        with Image.open(f) as img:
                            img_rgb = img.convert("RGB")
                    
                    # Transform and move to GPU
                    tensor = val_transforms(img_rgb).unsqueeze(0).to(device)
                    
                    # Infer
                    outputs = model(tensor)
                    _, pred_idx = torch.max(outputs, 1)
                    pred_class = IDX_TO_CLASS[pred_idx.item()]
                    
                    # Logic: If predicted class is different from true class, move it
                    # E.g. If predicting E for a B patch, move to B/E/
                    if pred_class != true_class:
                        dest_dir = cls_dir / pred_class
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_dir / file_path.name
                        
                        if args.dry_run:
                            if moved_this_class < 5:
                                tqdm.write(f"[DRY-RUN] Move: {file_path.name} -> {true_class}/{pred_class}/")
                        else:
                            shutil.move(str(file_path), str(dest_path))
                            
                        moved_this_class += 1
                        total_moved += 1
                        
                except Exception as e:
                    tqdm.write(f"[ERROR] Failed processing {file_path.name}: {e}")
            
            print(f"  -> Moved {moved_this_class} suspicious patches from {true_class}.")

    if args.dry_run:
        print(f"\n[DONE] Dry-run complete. Would have moved {total_moved} files totally.")
    else:
        print(f"\n[DONE] Successfully moved {total_moved} files into prediction-based subfolders.")

if __name__ == "__main__":
    main()
