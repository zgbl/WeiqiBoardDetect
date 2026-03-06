import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import shutil

# Import project-specific modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_win import build_model, detect_device, CLASSES
from train_win import PatchDataset, load_config, gp

def main():
    parser = argparse.ArgumentParser(description="Identify dirty data (mislabeled samples) using a trained model.")
    parser.add_argument("--config",     default="config_windows.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", required=True,                help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--output",     default="dirty_data_report",   help="Directory to save report and images")
    parser.add_argument("--threshold",  type=float, default=0.9,       help="Confidence threshold to flag as highly suspicious")
    parser.add_argument("--batch_size", type=int,   default=128,      help="Batch size for inference")
    parser.add_argument("--move",       action="store_true",           help="Move suspicious files to DIRTY folders automatically")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = detect_device()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Parameters
    patches_raw = gp(cfg, "data", "patches_dir")
    patches_dirs = [Path(p) for p in patches_raw] if isinstance(patches_raw, list) else [Path(patches_raw)]
    gomrade_dirs = gp(cfg, "data", "gomrade_dir")
    
    arch        = gp(cfg, "model", "arch",        default="StoneCNN")
    patch_size  = gp(cfg, "model", "patch_size",  default=48)
    num_classes = gp(cfg, "model", "num_classes", default=3)

    # 2. Load Model
    print(f"[MODEL] Loading {arch} from {args.checkpoint}...")
    model = build_model(arch, num_classes, patch_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3. Load Dataset (Full set, no split)
    print(f"[DATA] Loading patches from {len(patches_dirs)} directories:")
    for d in patches_dirs:
        print(f"  - SOURCE: {d}")
        if args.move:
            # We explicitly show where the DIRTY folder will be
            print(f"    ➔ TARGET: {d / 'DIRTY'}")
    
    if args.move:
        print(f"\n[MOVE] Immediate mode enabled: Suspicious patches will be moved to /DIRTY subfolders as detected.")
    
    print("[DATA] Preparing all patches for analysis...")
    val_transforms = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # We use PatchDataset but bypass the split by requesting all samples
    full_ds = PatchDataset(patches_dirs, gomrade_dirs, split="all", transform=val_transforms)
    loader = DataLoader(full_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=2)

    # 4. Inference loop
    print(f"[INFERENCE] Analyzing {len(full_ds):,} samples...")
    results = []
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            # Map batch indices back to dataset indices
            start_idx = i * int(args.batch_size)
            for j in range(len(imgs)):
                ds_idx = start_idx + j
                img_path, true_label = full_ds.samples[ds_idx]
                pred_label = preds[j].item()
                conf = confidences[j].item()
                
                # Flag if prediction != label and confidence is high
                is_suspicious = (pred_label != true_label) and (conf >= args.threshold)
                
                results.append({
                    "path": img_path,
                    "true_label": CLASSES[true_label],
                    "pred_label": CLASSES[pred_label],
                    "confidence": conf,
                    "is_suspicious": is_suspicious,
                    "loss": -np.log(max(probs[j][true_label].item(), 1e-10))
                })

                # IMMEDIATELY move if suspicious and --move is set
                if args.move and is_suspicious:
                    src = Path(img_path)
                    if src.exists():
                        dest_dir = src.parent / "DIRTY"
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest = dest_dir / src.name
                        try:
                            shutil.move(str(src), str(dest))
                        except Exception as e:
                            # Print briefly to not flood tqdm
                            tqdm.write(f"[ERROR] Failed to move {src.name}: {e}")

    # 5. Export Report
    df = pd.DataFrame(results)
    report_csv = output_path / "dirty_data_candidates.csv"
    df.to_csv(report_csv, index=False)
    
    suspicious_df = df[df["is_suspicious"] == True].sort_values(by="confidence", ascending=False)
    print(f"\n[SUMMARY] Found {len(suspicious_df):,} highly suspicious samples (Confidence > {args.threshold} but Label mismatch)")
    if args.move:
        print(f"[SUMMARY] These samples have been moved to their respective DIRTY folders.")
    print(f"[SUMMARY] Detailed report saved to: {report_csv}")

    # 6. Generate Visualization Grid (Top offenders from the list)
    if len(suspicious_df) > 0:
        # Note: Since files are MOVED, we need to point to the NEW paths for visualization
        top_n = min(64, len(suspicious_df))
        top_offenders = suspicious_df.head(top_n)
        
        print(f"[VISUAL] Generating grid for top {top_n} suspicious samples...")
        grid_imgs = []
        
        for _, row in top_offenders.iterrows():
            # Adjust path if it was moved
            current_path = Path(row["path"])
            if args.move:
                current_path = current_path.parent / "DIRTY" / current_path.name
            
            if current_path.exists():
                img = Image.open(current_path).convert("RGB").resize((patch_size, patch_size))
                img_tensor = transforms.ToTensor()(img)
                grid_imgs.append(img_tensor)
            
        if grid_imgs:
            grid = utils.make_grid(grid_imgs, nrow=8, padding=2)
            grid_pil = transforms.ToPILImage()(grid)
            grid_pil.save(output_path / "top_dirty_samples.png")
            print(f"[VISUAL] Grid saved to: {output_path / 'top_dirty_samples.png'}")

if __name__ == "__main__":
    main()
