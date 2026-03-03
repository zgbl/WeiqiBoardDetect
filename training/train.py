"""
train.py — 围棋棋子分类器训练脚本
================================================================
特性:
  ✅ 路径全部从 config.yaml 读取，可直接拷贝到 Windows 改路径
  ✅ 屏幕输出同步写入日志文件（log/目录）
  ✅ --subset 0.2 只动20%数据快速迭代验证参数
  ✅ sqrt 采样策略，解决空格过多导致误报的问题
  ✅ 自动检测 CUDA / Apple MPS / CPU
  ✅ CUDA AMP 混合精度，GTX1080 加速
  ✅ 断点续训（--resume）

使用方法:
  pip install torch torchvision pyyaml scikit-learn matplotlib tqdm

  # Mac：
  python3 train.py --config config.yaml --subset 0.2 --epochs 100

  # Windows GTX1080：
  python train.py --config config_windows.yaml

  # 快速验证（仅 5 轮）
  python3 train.py --epochs 5 --subset 0.1

  # 断点续训
  python3 train.py --resume
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

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from model import build_model, detect_device, CLASSES


# ──────────────────────────────────────────────────────────────
# TeeLogger — 屏幕 + 文件同步输出
# ──────────────────────────────────────────────────────────────
class TeeLogger:
    """
    同时把输出写到 屏幕 和 日志文件。

    使用：sys.stdout = TeeLogger(log_path)
    自动将所有 print() 同时输出到两个地方。
    """
    def __init__(self, log_path, mode="w", encoding="utf-8"):
        self.terminal = sys.__stdout__   # 原始屏幕
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, mode, encoding=encoding, buffering=1)
        print(f"[LOG] 日志输出: {log_path}")

    def write(self, msg):
        self.terminal.write(msg)
        self.log_file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def close(self):
        sys.stdout = self.terminal   # 恢复原始 stdout
        self.log_file.close()


# ──────────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────────
def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[警告] 配置文件 {config_path} 不存在，使用脚本默认值")
        return {}
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def gp(cfg, *keys, default=None):
    """嵌套字典安全取值。"""
    v = cfg
    for k in keys:
        if not isinstance(v, dict) or k not in v:
            return default
        v = v[k]
    return v if v is not None else default


# ──────────────────────────────────────────────────────────────
# 数据集
# ──────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    """
    从 patches/{B,W,E}/ 目录加载分类数据集。

    目录结构：
      patches/
        B/  ← 黑子 patch
        W/  ← 白子 patch
        E/  ← 空格 patch
    """
    CLASS_TO_IDX = {"B": 0, "W": 1, "E": 2}

    def __init__(self, patches_dir, split="train", transform=None,
                 val_ratio=0.15, test_ratio=0.05, seed=42, subset_ratio=1.0):
        self.transform = transform
        self.samples = []

        patches_dir = Path(patches_dir)
        all_samples = []

        for cls_name, cls_idx in self.CLASS_TO_IDX.items():
            cls_dir = patches_dir / cls_name
            if not cls_dir.exists():
                print(f"  [警告] 类别目录不存在: {cls_dir}")
                continue
            files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            for f in files:
                all_samples.append((str(f), cls_idx))

        if not all_samples:
            raise RuntimeError(f"在 {patches_dir} 中没有找到任何 patch！请先运行 extract_patches.py")

        rng = random.Random(seed)
        rng.shuffle(all_samples)

        n = len(all_samples)
        n_test  = int(n * test_ratio)
        n_val   = int(n * val_ratio)
        n_train = n - n_test - n_val

        if split == "train":
            self.samples = all_samples[:n_train]
            # 子集采样（只对 train 有效）
            if 0.0 < subset_ratio < 1.0:
                k = max(100, int(len(self.samples) * subset_ratio))
                self.samples = self.samples[:k]
                print(f"  [子集] 训练集中取前 {k:,} 条 ({subset_ratio*100:.0f}%)（快速迭代模式）")
        elif split == "val":
            self.samples = all_samples[n_train:n_train + n_val]
        elif split == "test":
            self.samples = all_samples[n_train + n_val:]
        else:
            self.samples = all_samples

        counts = [0, 0, 0]
        for _, idx in self.samples:
            counts[idx] += 1
        print(f"  [{split:5s}] 总={len(self.samples):,}  "
              f"B={counts[0]:,}  W={counts[1]:,}  E={counts[2]:,}")

        self.class_counts = counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if PIL_AVAILABLE:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
        elif CV2_AVAILABLE:
            arr = cv2.imread(path)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
        else:
            raise RuntimeError("需要安装 Pillow 或 opencv-python")

        return img, label


def make_weighted_sampler(dataset, strategy="sqrt"):
    """
    按策略创建加权随机采样器。

    strategy:
      "equal"  — B:W:E 强制1:1:1（理论正确）但导致模型过于乐观，会将太多空格预测为有子
      "sqrt"   — 平方根增益，不那么极端。真实棋盘约 14% 有子格子，此策略更接近真实分布（推荐）
      "none"   — 不加权，按原始分布训练（模型会学到只预测空格）
    """
    counts = dataset.class_counts
    total  = sum(counts)

    if strategy == "equal":
        class_w = [total / max(c, 1) for c in counts]
    elif strategy == "sqrt":
        class_w = [np.sqrt(total / max(c, 1)) for c in counts]
    else:  # "none"
        return None

    sample_weights = [class_w[label] for _, label in dataset.samples]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


# ──────────────────────────────────────────────────────────────
# 训练核心
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=0, total_epochs=0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    desc = f"  Train {epoch:3d}/{total_epochs}"
    pbar = tqdm(loader, desc=desc, unit="batch", ncols=90,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

        # 每批次实时更新进度条右侧显示
        pbar.set_postfix({
            "loss": f"{total_loss/max(total,1):.4f}",
            "acc":  f"{correct/max(total,1):.4f}",
        }, refresh=False)

    pbar.close()
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="  Val  "):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=desc, unit="batch", ncols=90, leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix({"acc": f"{correct/max(total,1):.4f}"}, refresh=False)

    pbar.close()
    return (total_loss / max(total, 1),
            correct / max(total, 1),
            all_preds, all_labels)


# ──────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────
def save_training_plot(history, output_dir):
    """保存 Loss / Accuracy 训练曲线图。"""
    try:
        import matplotlib.pyplot as plt
        epochs = range(1, len(history["train_acc"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, history["train_loss"], label="Train Loss")
        ax1.plot(epochs, history["val_loss"],   label="Val Loss")
        ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)

        ax2.plot(epochs, history["train_acc"], label="Train Acc")
        ax2.plot(epochs, history["val_acc"],   label="Val Acc")
        ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "training_curve.png", dpi=150)
        plt.close()
        print(f"  [保存] 训练曲线 → {output_dir}/training_curve.png")
    except ImportError:
        print("  [提示] 安装 matplotlib 可以保存训练曲线图")


def save_confusion_matrix(all_preds, all_labels, output_dir):
    """保存混淆矩阵图。"""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, classification_report
        import numpy as np

        cm = confusion_matrix(all_labels, all_preds)
        labels = ["B (黑)", "W (白)", "E (空)"]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("预测"); ax.set_ylabel("真实")
        ax.set_title("混淆矩阵（测试集）")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"  [保存] 混淆矩阵 → {output_dir}/confusion_matrix.png")

        report = classification_report(all_labels, all_preds,
                                       target_names=["B", "W", "E"])
        print(f"\n  分类报告:\n{report}")
        with open(Path(output_dir) / "classification_report.txt", "w") as f:
            f.write(report)
    except ImportError:
        print("  [提示] 安装 scikit-learn + matplotlib 可以保存混淆矩阵")


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="围棋棋子 CNN 分类器训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 train.py                               # Mac 标准训练
  python3 train.py --config config_windows.yaml  # Windows 配置
  python3 train.py --subset 0.2 --epochs 100     # 快速迭代模式（20% 数据）
  python3 train.py --resume                      # 断点续训
  python3 train.py --workers 0                   # Windows 密 bug 时用
  python3 train.py --sampling sqrt               # 修改采样策略
        """
    )
    parser.add_argument("--config",     default="config.yaml",  help="配置文件路径")
    parser.add_argument("--patches",    default=None,           help="patch 目录（覆盖配置）")
    parser.add_argument("--output",     default=None,           help="checkpoint 输出目录（覆盖配置）")
    parser.add_argument("--log-dir",    default=None,           help="日志目录（覆盖配置）")
    parser.add_argument("--arch",       default=None,           help="模型架构: StoneCNN | MobileNetV3")
    parser.add_argument("--epochs",     type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")
    parser.add_argument("--lr",         type=float, default=None, help="初始学习率")
    parser.add_argument("--workers",    type=int, default=None,  help="DataLoader 进程数")
    parser.add_argument("--patch-size", type=int, default=None,  help="patch 像素尺寸")
    parser.add_argument("--subset",     type=float, default=None,
                        help="子集训练比例: 0.0~1.0（1.0=全量, 0.2=只动20%快速迭代）")
    parser.add_argument("--sampling",   default=None,
                        choices=["equal", "sqrt", "none"],
                        help="采样策略: sqrt=推荐 equal=强制1:1:1 none=原始分布")
    parser.add_argument("--resume",     action="store_true",    help="从最新 checkpoint 断点续训")
    parser.add_argument("--no-amp",     action="store_true",    help="禁用混合精度训练")
    parser.add_argument("--seed",       type=int, default=42,   help="随机种子")
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 参数优先级：CLI 参数 > config.yaml > 默认值
    patches_dir   = Path(args.patches  or gp(cfg, "data",  "patches_dir",       default="patches"))
    output_dir    = Path(args.output   or gp(cfg, "model", "output_dir",        default="checkpoints"))
    log_dir       = Path(args.log_dir  or gp(cfg, "logging", "log_dir",         default="log"))
    arch          = args.arch          or gp(cfg, "model", "arch",              default="StoneCNN")
    patch_size    = args.patch_size    or gp(cfg, "model", "patch_size",        default=48)
    num_classes   =                       gp(cfg, "model", "num_classes",       default=3)
    epochs        = args.epochs        or gp(cfg, "train", "epochs",            default=60)
    batch_size    = args.batch_size    or gp(cfg, "train", "batch_size",        default=128)
    lr            = args.lr            or gp(cfg, "train", "lr",                default=1e-3)
    weight_dec    =                       gp(cfg, "train", "weight_decay",      default=1e-4)
    num_workers   = args.workers   if args.workers is not None else gp(cfg, "train", "num_workers", default=4)
    use_amp       = (not args.no_amp)  and gp(cfg, "train", "amp",             default=True)
    val_ratio     =                       gp(cfg, "data",  "val_ratio",         default=0.15)
    test_ratio    =                       gp(cfg, "data",  "test_ratio",        default=0.05)
    subset_ratio  = args.subset        or gp(cfg, "train", "subset_ratio",      default=1.0)
    sampling_strat= args.sampling      or gp(cfg, "train", "sampling_strategy", default="sqrt")

    # 固定随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 初始化日志展开（屏幕+文件同步）──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"train_{ts}.log"
    tee = TeeLogger(log_path)
    sys.stdout = tee

    output_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None
    if amp_enabled:
        print("[AMP] 混合精度训练已启用ﾈ节省显存，加快训练）")

    print(f"\n═══════════════════════════════════════════")
    print(f"  架构: {arch}  |  Patch: {patch_size}px  |  Epochs: {epochs}")
    print(f"  Batch: {batch_size}  |  LR: {lr}  |  Workers: {num_workers}")
    print(f"  采样策略: {sampling_strat}  |  子集: {subset_ratio*100:.0f}%")
    print(f"  Patch目录: {patches_dir}")
    print(f"  输出目录:  {output_dir}")
    print(f"═══════════════════════════════════════════\n")

    # ── 数据增强 ──
    aug_cfg    = gp(cfg, "augment") or {}
    jitter_cfg = aug_cfg.get("color_jitter", {})

    train_transforms = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        *([transforms.RandomHorizontalFlip()] if aug_cfg.get("random_flip", True) else []),
        *([transforms.RandomVerticalFlip()]   if aug_cfg.get("random_flip", True) else []),
        transforms.RandomRotation(aug_cfg.get("random_rotation", 15)),
        transforms.ColorJitter(
            brightness = jitter_cfg.get("brightness", 0.4),
            contrast   = jitter_cfg.get("contrast",   0.4),
            saturation = jitter_cfg.get("saturation", 0.2),
            hue        = jitter_cfg.get("hue",        0.05),
        ),
        *([transforms.GaussianBlur(3, sigma=(0.1, 1.5))] if aug_cfg.get("gaussian_blur", True) else []),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── 数据集 ──
    print("[数据] 加载数据集...")
    train_ds = PatchDataset(patches_dir, "train", train_transforms, val_ratio, test_ratio, args.seed, subset_ratio)
    val_ds   = PatchDataset(patches_dir, "val",   val_transforms,   val_ratio, test_ratio, args.seed)
    test_ds  = PatchDataset(patches_dir, "test",  val_transforms,   val_ratio, test_ratio, args.seed)

    sampler = make_weighted_sampler(train_ds, sampling_strat)
    shuffle_train = sampler is None  # 没有采样器时才 shuffle
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # ── 模型 ──
    model = build_model(arch, num_classes, patch_size).to(device)

    # 断点续训
    start_epoch = 1
    best_val_acc = 0.0
    best_ckpt = output_dir / "best_model.pt"
    latest_ckpt = output_dir / "latest_model.pt"

    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"[续训] 从 epoch {start_epoch} 继续，历史最优 val_acc={best_val_acc:.4f}")

    # ── 损失函数（类别加权，处理 E 类过多）──
    counts = train_ds.class_counts
    total = sum(counts)
    class_weights = torch.tensor(
        [total / (num_classes * max(c, 1)) for c in counts],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── 优化器 + 学习率调度 ──
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_dec)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if args.resume and latest_ckpt.exists():
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    # ── 训练循环 ──
    print(f"\n[训练] 开始训练，共 {epochs} 轮...\n")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    log_interval = gp(cfg, "logging", "log_interval", default=5)

    t_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch=epoch, total_epochs=epochs
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device,
            desc=f"  Val   {epoch:3d}/{epochs}"
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]
        elapsed_total = time.time() - t_start
        eta_sec = elapsed_total / epoch * (epochs - epoch)
        eta_str = f"{int(eta_sec//60)}m{int(eta_sec%60):02d}s" if eta_sec > 0 else "--"

        # 每个 epoch 都打印一行结果（不只是每5轮）
        star = "★" if val_acc > best_val_acc else " "
        print(f"  {star} [{epoch:3d}/{epochs}] "
              f"Train {train_acc:.4f}/{train_loss:.4f}  "
              f"Val {val_acc:.4f}/{val_loss:.4f}  "
              f"LR={cur_lr:.2e}  {elapsed:.0f}s  ETA={eta_str}")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "val_acc": val_acc, "arch": arch,
                "patch_size": patch_size, "num_classes": num_classes,
                "class_names": CLASSES,
            }, best_ckpt)
            print(f"  ★ 新最优: val_acc={val_acc:.4f} → 保存到 {best_ckpt.name}")

        # 保存最新 checkpoint（用于续训）
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "arch": arch, "patch_size": patch_size,
        }, latest_ckpt)

    total_time = time.time() - t_start
    print(f"\n[完成] 训练耗时 {total_time/60:.1f} 分钟")
    print(f"  最优验证集准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    # ── 最终测试集评估 ──
    print("\n[评估] 加载最优模型，在测试集上评估...")
    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state["model"])
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"  测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # ── 保存训练曲线 ──
    save_training_plot(history, output_dir)
    if gp(cfg, "logging", "save_confusion_matrix", default=True):
        save_confusion_matrix(preds, labels, output_dir)

    # ── 导出 ONNX ──
    print("\n[导出] 正在导出 ONNX 模型（用于移动端部署）...")
    model.eval().cpu()
    dummy = torch.randn(1, 3, patch_size, patch_size)
    onnx_path = output_dir / "stone_classifier.onnx"
    try:
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"],
            opset_version=17,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
        )
        print(f"  ✅ ONNX 导出成功: {onnx_path}")
    except Exception as e:
        print(f"  ⚠ ONNX 导出失败: {e}")

    # ── 保存训练摘要 ──
    summary = {
        "arch": arch, "patch_size": patch_size,
        "best_val_acc": round(best_val_acc, 6),
        "test_acc":     round(test_acc, 6),
        "total_epochs": epochs,
        "training_time_min": round(total_time / 60, 1),
        "class_names": CLASSES,
        "device": str(device),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n╔══════════════════════════════════════════╗")
    print(f"║  训练完成！输出文件        ║")
    print(f"╚══════════════════════════════════════════╝")
    print(f"  {output_dir}/")
    print(f"    best_model.pt          ← 最优模型权重（PyTorch）")
    print(f"    latest_model.pt        ← 最新 checkpoint（续训用）")
    print(f"    stone_classifier.onnx  ← ONNX 模型（移动端部署用）")
    print(f"    training_curve.png     ← 训练曲线图")
    print(f"    confusion_matrix.png   ← 混淆矩阵（测试集）")
    print(f"    classification_report.txt ← 每类精确率/召回率")
    print(f"    training_summary.json  ← 训练结果摘要")


if __name__ == "__main__":
    main()
