"""
train-win-ram.py — Industrial Grade Weiqi Training Pipeline
================================================================
Upgrades based on Experiment 8 (94.8% Acc) and ChatGPT suggestions:
  ✅ Full RAM Pre-loading with Memory Safety (.copy())
  ✅ Auto-Cleaning: Saves 'suspects.csv' (pred != label & conf > 0.9)
  ✅ Data Filtering: Auto-ignores 'Label=W' samples with <5% white area
  ✅ Early Stopping (Patience=10)
  ✅ Windows Stability Fix (num_workers=0)
"""

import os
import sys
import json
import yaml
import argparse
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from model_win import build_model, detect_device, CLASSES

# --- Logging Utils ---
class TeeLogger:
    def __init__(self, log_path, mode="w", encoding="utf-8"):
        self.terminal = sys.__stdout__
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, mode, encoding=encoding, buffering=1)
        print(f"[LOG] Logging to: {log_path}")

    def write(self, msg):
        self.terminal.write(msg)
        self.log_file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self): return self.terminal.isatty()
    def close(self): sys.stdout = self.terminal; self.log_file.close()

def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists(): return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def gp(cfg, *keys, default=None):
    v = cfg
    for k in keys:
        if not isinstance(v, dict) or k not in v: return default
        v = v[k]
    return v if v is not None else default

def calculate_white_ratio(pil_img):
    """Simple brightness heuristic for PIL image to filter noise."""
    arr = np.array(pil_img)
    # Average of RGB channels, normalize to 0-1
    brightness = np.mean(arr, axis=2) / 255.0
    return np.mean(brightness > 0.7)

# --- Dataset ---
class PatchDataset(Dataset):
    CLASS_TO_IDX = {"B": 0, "W": 1, "E": 2}

    def __init__(self, patches_dirs, gomrade_dirs=None, split="train", transform=None,
                 val_ratio=0.15, test_ratio=0.05, seed=42, subset_ratio=1.0, 
                 preload_to_ram=True, white_area_filter=True):
        self.transform = transform
        self.samples = []             # List of (path, label)
        self.cached_images = {}      # Map path -> PIL Image
        self.preload_to_ram = preload_to_ram
        self.white_area_filter = white_area_filter

        if not isinstance(patches_dirs, list): patches_dirs = [patches_dirs]
        self.patches_dirs = [Path(p) for p in patches_dirs]

        # 1. Session Isolation
        session_names = set()
        if gomrade_dirs:
            if not isinstance(gomrade_dirs, list): gomrade_dirs = [gomrade_dirs]
            for g_dir in gomrade_dirs:
                if Path(g_dir).exists():
                    for sub in Path(g_dir).iterdir():
                        if sub.is_dir() and sub.name != "patches": session_names.add(sub.name)

        all_files = []
        for p_dir in self.patches_dirs:
            if not p_dir.exists(): continue
            for cls_name, cls_idx in self.CLASS_TO_IDX.items():
                cls_dir = p_dir / cls_name
                if not cls_dir.exists(): continue
                files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
                for f_path in files:
                    fname = f_path.name
                    matching_session = "unknown"
                    if session_names:
                        sorted_sessions = sorted(list(session_names), key=len, reverse=True)
                        for s_name in sorted_sessions:
                            if fname.startswith(s_name + "_"): matching_session = s_name; break
                    else:
                        matching_session = fname.rsplit('_', 3)[0]
                    all_files.append((str(f_path), cls_idx, matching_session))
        
        if not all_files: raise RuntimeError(f"No patches found!")

        # 2. Split
        sessions_map = {}
        for path, label, s_name in all_files:
            if s_name not in sessions_map: sessions_map[s_name] = []
            sessions_map[s_name].append((path, label))
        
        unique_sessions = sorted(list(sessions_map.keys()))
        random.Random(seed).shuffle(unique_sessions)

        n_sess = len(unique_sessions)
        n_test = max(1, int(n_sess * test_ratio))
        n_val  = max(1, int(n_sess * val_ratio))
        n_train = n_sess - n_test - n_val

        if split == "train": selected = unique_sessions[:n_train]
        elif split == "val": selected = unique_sessions[n_train:n_train + n_val]
        elif split == "test": selected = unique_sessions[n_train + n_val:]
        else: selected = unique_sessions

        temp_samples = []
        for s_name in selected: temp_samples.extend(sessions_map[s_name])
        
        # 3. Filtering and Preloading
        ignored_count = 0
        print(f"  [{split:5s}] Loading/Filtering {len(temp_samples):,} images...")
        for path, label in tqdm(temp_samples, desc=f"RAM {split}", leave=False, file=sys.stdout):
            with Image.open(path) as img:
                pil_img = img.convert("RGB")
                
                # Rule-based filter: White stones must have brightness
                if self.white_area_filter and label == 1: # 1 = W
                    ratio = calculate_white_ratio(pil_img)
                    if ratio < 0.05: # Threshold from suggestion
                        ignored_count += 1
                        continue
                
                if self.preload_to_ram:
                    self.cached_images[path] = pil_img.copy()
                
                self.samples.append((path, label))

        if ignored_count > 0:
            print(f"  [FILTER] Ignored {ignored_count} samples due to white area rule.")

        if split == "train" and 0.0 < subset_ratio < 1.0:
            random.Random(seed).shuffle(self.samples)
            self.samples = self.samples[:int(len(self.samples) * subset_ratio)]

        counts = [0, 0, 0]
        for _, idx in self.samples: counts[idx] += 1
        self.class_counts = counts
        print(f"  [{split:5s}] Total={len(self.samples):,} B={counts[0]} W={counts[1]} E={counts[2]}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if self.preload_to_ram and path in self.cached_images:
            img = self.cached_images[path]
        else:
            img = Image.open(path).convert("RGB")
        
        if self.transform: img = self.transform(img)
        else: img = transforms.ToTensor()(img)
        return img, label, path

# --- Utils ---
def make_weighted_sampler(dataset, strategy="sqrt"):
    counts = dataset.class_counts; total = sum(counts)
    if strategy == "equal": class_w = [total / max(c, 1) for c in counts]
    elif strategy == "sqrt": class_w = [np.sqrt(total / max(c, 1)) for c in counts]
    else: return None
    weights = [class_w[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights, len(weights))

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=0, total_epochs=0):
    model.train(); total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f" Train {epoch:3d}/{total_epochs}", unit="batch", ncols=90, file=sys.stdout)
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(imgs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            outputs = model(imgs); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(1) == labels).sum().item()
        total += batch_size
        pbar.set_postfix({"loss": f"{total_loss/total:.4f}", "acc": f"{correct/total:.4f}"}, refresh=False)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc=" Val "):
    model.eval(); total_loss, correct, total = 0.0, 0, 0
    suspects = []
    
    for imgs, labels, paths in tqdm(loader, desc=desc, unit="batch", ncols=90, leave=False, file=sys.stdout):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs); loss = criterion(outputs, labels)
        
        probs = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(1)
        
        # Suspect Detection (Confidence > 0.9 and Wrong)
        for p, l, pr, c in zip(paths, labels, preds, confs):
            if pr != l and c > 0.9:
                suspects.append({
                    "path": p,
                    "target": CLASSES[l.item()],
                    "pred": CLASSES[pr.item()],
                    "confidence": f"{c.item():.4f}"
                })

        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        
    return total_loss / total, correct / total, suspects

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Industrial Grade Weiqi Training Pipeline")
    parser.add_argument("--config", default="config_windows.yaml")
    parser.add_argument("--no-ram", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    patches_dirs = [Path(p) for p in gp(cfg, "data", "patches_dir")] if isinstance(gp(cfg, "data", "patches_dir"), list) else [Path(gp(cfg, "data", "patches_dir"))]
    output_dir = Path(gp(cfg, "model", "output_dir", default="checkpoints"))
    arch = gp(cfg, "model", "arch", default="StoneCNN")
    patch_size = gp(cfg, "model", "patch_size", default=48)
    epochs = gp(cfg, "train", "epochs", default=60)
    batch_size = gp(cfg, "train", "batch_size", default=64)
    lr = gp(cfg, "train", "lr", default=0.001)
    num_workers = 0 if not args.no_ram else gp(cfg, "train", "num_workers", default=2)
    sampling = gp(cfg, "train", "sampling_strategy", default="sqrt")
    use_h_filter = gp(cfg, "heuristic", "white_area_filter", default=True)

    device = detect_device()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(gp(cfg, "logging", "log_dir", default="log")) / f"train_industrial_{ts}.log"
    tee = TeeLogger(log_path)
    sys.stdout = tee; sys.stderr = tee
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INDUSTRIAL MODE] Arch: {arch} | Epochs: {epochs} | P-Size: {patch_size}")
    print(f"Memory Safety Enabled | Suspect Detection Active | W-Area Filtering: {use_h_filter}")

    train_tf = transforms.Compose([
        transforms.Resize((patch_size, patch_size)), transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), transforms.RandomRotation(15), 
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.05), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((patch_size, patch_size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = PatchDataset(patches_dirs, gp(cfg, "data", "gomrade_dir"), "train", train_tf, preload_to_ram=not args.no_ram, white_area_filter=use_h_filter)
    val_ds = PatchDataset(patches_dirs, gp(cfg, "data", "gomrade_dir"), "val", val_tf, preload_to_ram=not args.no_ram, white_area_filter=False)
    
    sampler = make_weighted_sampler(train_ds, sampling)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(arch, 3, patch_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    
    start_epoch, best_val_acc = 1, 0.0
    latest_ckpt = output_dir / "latest_model.pt"
    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"]); start_epoch = ckpt["epoch"] + 1; best_val_acc = ckpt.get("best_val_acc", 0.0)
        optimizer.load_state_dict(ckpt["optimizer"])

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # Early Stopping init
    patience, no_improve = 10, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n[START] Training for {epochs} epochs...\n")
    for epoch in range(start_epoch, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, epochs)
        val_loss, val_acc, suspects = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_acc"].append(tr_acc); history["val_acc"].append(val_acc)
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)

        star = "*" if val_acc > best_val_acc else " "
        print(f"  {star} [{epoch:3d}/{epochs}] Train {tr_acc:.4f} Val {val_acc:.4f} Best {best_val_acc:.4f} (Suspects: {len(suspects)})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc; no_improve = 0
            torch.save({"model": model.state_dict(), "val_acc": val_acc, "arch": arch, "patch_size": patch_size}, output_dir / "best_model.pt")
            # Save latest suspects list on improvement
            if suspects:
                pd.DataFrame(suspects).to_csv(output_dir / "suspects_latest.csv", index=False)
        else:
            no_improve += 1

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_val_acc": best_val_acc}, latest_ckpt)
        if no_improve >= patience:
            print(f"\n[EARLY STOP] No improve for {patience} epochs. Breaking.")
            break

    try:
        from train_win import save_training_plot as save_p
        save_p(history, output_dir)
    except:
        save_training_plot(history, output_dir)
        
    print(f"\n[DONE] Best Val Acc: {best_val_acc:.4f}")
    if suspects:
        print(f"  [TIPS] Suspect samples exported to: {output_dir}/suspects_latest.csv")

if __name__ == "__main__":
    main()
