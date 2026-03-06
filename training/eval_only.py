import torch
from pathlib import Path
import sys
import os

# 确保能引用到同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_win import evaluate, PatchDataset, load_config, gp, build_model, detect_device
from torch.utils.data import DataLoader
from torchvision import transforms

def do_test(checkpoint_path, config_path):
    device = detect_device()
    cfg = load_config(config_path)
    
    patch_size = gp(cfg, "model", "patch_size", default=48)
    arch = gp(cfg, "model", "arch", default="StoneCNN")
    num_classes = gp(cfg, "model", "num_classes", default=3)
    
    # 1. 准备数据
    print(f"[DATA] Loading test set...")
    val_transforms = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    patches_raw = gp(cfg, "data", "patches_dir")
    gomrade_dirs = gp(cfg, "data", "gomrade_dir")
    
    # 我们只加载测试集
    test_ds = PatchDataset(patches_raw, gomrade_dirs, split="test", transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    # 2. 加载模型
    print(f"[MODEL] Loading weights from: {checkpoint_path}")
    model = build_model(arch, num_classes, patch_size).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 兼容处理：有些 checkpoint 只有 state_dict，有些是整个字典
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    
    # 3. 运行评估 (不带任何 Grad，不更新任何权重)
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc, preds, labels = evaluate(
        model, test_loader, criterion, device, 
        desc="Testing", 
        heuristic_cfg=gp(cfg, "heuristic")
    )
    
    print(f"\n======================================")
    print(f"  Test Results for: {Path(checkpoint_path).name}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Loss:     {loss:.4f}")
    print(f"======================================\n")

if __name__ == "__main__":
    # 使用刚备份的 V5 权重进行测试
    ckpt = r"d:\Codes\WeiqiBoardDetect\WeiqiBoardDetect-main\training\checkpoints_v5_41ep\best_model.pt"
    cfg = r"d:\Codes\WeiqiBoardDetect\WeiqiBoardDetect-main\training\config_windows.yaml"
    do_test(ckpt, cfg)
