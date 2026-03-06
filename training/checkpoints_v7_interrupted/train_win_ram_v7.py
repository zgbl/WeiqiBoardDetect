"""
train-win-ram.py — RAM-Optimized training script for WeiqiBoardDetect
================================================================
Based on train-win.py, but loads the ENTIRE dataset into RAM to bypass HDD I/O bottlenecks.
Use this version if your HDD is slow and you have >16GB of RAM.

Difference from train-win.py:
  🚀 Pre-loads all JPEG/PNG patches into memory during initialization.
  🚀 Drastically reduces training time on mechanical hard drives.
"""

import os
import sys
import json
import yaml
import argparse
import time
import random
import numpy as np
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

# --- Same TeeLogger and Utils as train-win.py ---
class TeeLogger:
    def __init__(self, log_path, mode="w", encoding="utf-8"):
        self.terminal = sys.__stdout__
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, mode, encoding=encoding, buffering=1)
        print(f"[LOG] Logging to: {log_path}")
    def write(self, msg):
        self.terminal.write(msg); self.log_file.write(msg)
    def flush(self):
        self.terminal.flush(); self.log_file.flush()
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

# ──────────────────────────────────────────────────────────────
# RAM-Optimized Dataset
# ──────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    """
    Loads classification patches and caches them in RAM.
    """
    CLASS_TO_IDX = {"B": 0, "W": 1, "E": 2}

    def __init__(self, patches_dirs, gomrade_dirs=None, split="train", transform=None,
                 val_ratio=0.15, test_ratio=0.05, seed=42, subset_ratio=1.0, 
                 preload_to_ram=True):
        self.transform = transform
        self.samples = [] # List of (path, label)
        self.cached_images = {} # Map path -> PIL Image (if preloading)
        self.preload_to_ram = preload_to_ram

        if not isinstance(patches_dirs, list):
            patches_dirs = [patches_dirs]
        self.patches_dirs = [Path(p) for p in patches_dirs]

        # 1. Session Isolation (Leakage Fix)
        session_names = set()
        if gomrade_dirs:
            if not isinstance(gomrade_dirs, list): gomrade_dirs = [gomrade_dirs]
            for g_dir in gomrade_dirs:
                if Path(g_dir).exists():
                    for sub in Path(g_dir).iterdir():
                        if sub.is_dir() and sub.name != "patches":
                            session_names.add(sub.name)

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
                            if fname.startswith(s_name + "_"):
                                matching_session = s_name; break
                    else:
                        matching_session = fname.rsplit('_', 3)[0]
                    all_files.append((str(f_path), cls_idx, matching_session))
        
        if not all_files:
            raise RuntimeError(f"No patches found in {self.patches_dirs}!")

        # 2. Group and Split
        sessions_map = {}
        for path, label, s_name in all_files:
            if s_name not in sessions_map: sessions_map[s_name] = []
            sessions_map[s_name].append((path, label))
        
        unique_sessions = sorted(list(sessions_map.keys()))
        rng = random.Random(seed); rng.shuffle(unique_sessions)

        n_sess = len(unique_sessions)
        n_test_sess = max(1, int(n_sess * test_ratio))
        n_val_sess  = max(1, int(n_sess * val_ratio))
        n_train_sess = n_sess - n_test_sess - n_val_sess

        if split == "train": selected_sessions = unique_sessions[:n_train_sess]
        elif split == "val": selected_sessions = unique_sessions[n_train_sess:n_train_sess + n_val_sess]
        elif split == "test": selected_sessions = unique_sessions[n_train_sess + n_val_sess:]
        else: selected_sessions = unique_sessions

        for s_name in selected_sessions:
            self.samples.extend(sessions_map[s_name])
        
        if split == "train" and 0.0 < subset_ratio < 1.0:
            rng.shuffle(self.samples)
            self.samples = self.samples[:max(100, int(len(self.samples) * subset_ratio))]

        # 3. RAM PRELOADING
        if self.preload_to_ram:
            print(f"  [{split:5s}] Pre-loading {len(self.samples):,} images to RAM...")
            for path, _ in tqdm(self.samples, desc=f"RAM {split}", leave=False):
                # We store as PIL Image to allow on-the-fly augmentation
                self.cached_images[path] = Image.open(path).convert("RGB")
        
        counts = [0, 0, 0]
        for _, idx in self.samples: counts[idx] += 1
        self.class_counts = counts
        print(f"  [{split:5s}] Sessions={len(selected_sessions)}/{n_sess}  Total={len(self.samples):,}  B={counts[0]} W={counts[1]} E={counts[2]}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if self.preload_to_ram and path in self.cached_images:
            img = self.cached_images[path]
        else:
            img = Image.open(path).convert("RGB")
        
        if self.transform: img = self.transform(img)
        else: img = transforms.ToTensor()(img)
        return img, label

# --- The rest of core training logic (Copied from train-win.py for consistency) ---
def make_weighted_sampler(dataset, strategy="sqrt"):
    counts = dataset.class_counts; total  = sum(counts)
    if strategy == "equal": class_w = [total / max(c, 1) for c in counts]
    elif strategy == "sqrt": class_w = [np.sqrt(total / max(c, 1)) for c in counts]
    else: return None
    sample_weights = [class_w[label] for _, label in dataset.samples]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=0, total_epochs=0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"  Train {epoch:3d}/{total_epochs}", unit="batch", ncols=90)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                outputs = model(imgs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            outputs = model(imgs); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0); correct += (outputs.argmax(1) == labels).sum().item(); total += imgs.size(0)
        pbar.set_postfix({"loss": f"{total_loss/total:.4f}", "acc": f"{correct/total:.4f}"}, refresh=False)
    return total_loss / total, correct / total

def get_white_area_ratio(imgs_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(imgs_tensor.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(imgs_tensor.device)
    unnorm = imgs_tensor * std + mean
    brightness = unnorm.mean(dim=1)
    white_mask = (brightness > 0.7).float()
    return white_mask.mean(dim=(1, 2))

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="  Val  ", heuristic_cfg=None):
    model.eval(); total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    use_filter = gp(heuristic_cfg, "white_area_filter", default=False)
    min_ratio = gp(heuristic_cfg, "min_area_ratio", default=0.10)
    for imgs, labels in tqdm(loader, desc=desc, unit="batch", ncols=90, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs); loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0); preds = outputs.argmax(1)
        if use_filter:
            ratios = get_white_area_ratio(imgs)
            preds[(preds == 1) & (ratios < min_ratio)] = 2
        correct += (preds == labels).sum().item(); total += imgs.size(0)
        all_preds.extend(preds.cpu().tolist()); all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels

# Visualization placeholders (omitted or kept) - keeping for completion
def save_training_plot(history, output_dir):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
        plt.subplot(1,2,2); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title('Acc'); plt.legend()
        plt.savefig(Path(output_dir) / "training_curve.png"); plt.close()
    except: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_windows.yaml")
    parser.add_argument("--no-ram", action="store_true", help="Disable RAM preloading")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    patches_raw = gp(cfg, "data", "patches_dir")
    patches_dirs = [Path(p) for p in patches_raw] if isinstance(patches_raw, list) else [Path(patches_raw)]
    gomrade_dirs = gp(cfg, "data", "gomrade_dir")
    
    output_dir = Path(gp(cfg, "model", "output_dir", default="checkpoints"))
    arch = gp(cfg, "model", "arch", default="StoneCNN")
    patch_size = gp(cfg, "model", "patch_size", default=48)
    epochs = gp(cfg, "train", "epochs", default=60)
    batch_size = gp(cfg, "train", "batch_size", default=128)
    lr = gp(cfg, "train", "lr", default=1e-3)
    num_workers = gp(cfg, "train", "num_workers", default=4)
    sampling_strat = gp(cfg, "train", "sampling_strategy", default="sqrt")
    subset_ratio = gp(cfg, "train", "subset_ratio", default=1.0)

    device = detect_device()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = TeeLogger(Path(gp(cfg, "logging", "log_dir", default="log")) / f"train_ram_{ts}.log")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15), transforms.ColorJitter(0.4, 0.4, 0.2, 0.05),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"\n[RAM MODE] Architecture: {arch} | RAM Preloading: {not args.no_ram}")
    
    # Dataset
    train_ds = PatchDataset(patches_dirs, gomrade_dirs, "train", train_transforms, subset_ratio=subset_ratio, preload_to_ram=not args.no_ram)
    val_ds   = PatchDataset(patches_dirs, gomrade_dirs, "val",   val_transforms, preload_to_ram=not args.no_ram)
    
    sampler = make_weighted_sampler(train_ds, sampling_strat)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(arch, 3, patch_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    start_epoch = 1
    best_val_acc = 0.0
    latest_ckpt = output_dir / "latest_model.pt"
    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"]); start_epoch = ckpt["epoch"] + 1; best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"[RESUME] Loaded E{start_epoch-1}")

    criterion = nn.CrossEntropyLoss()
    T_max = max(1, epochs - start_epoch + 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(start_epoch, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, epochs)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, heuristic_cfg=gp(cfg, "heuristic"))
        scheduler.step()
        history["train_acc"].append(tr_acc); history["val_acc"].append(val_acc)
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        print(f"  [{epoch:3d}/{epochs}] Train {tr_acc:.4f} Val {val_acc:.4f} Best {best_val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_acc": val_acc}, output_dir / "best_model.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_val_acc": best_val_acc}, latest_ckpt)

    print(f"\n[DONE] Best Val Acc: {best_val_acc:.4f}")
    save_training_plot(history, output_dir)

if __name__ == "__main__":
    main()
