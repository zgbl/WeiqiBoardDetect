"""
model.py — 围棋棋子分类器模型定义
================================================================
提供两种模型：
  StoneCNN      - 自定义轻量 CNN，专为 patch 分类优化（推荐）
  MobileNetV3   - ImageNet 预训练迁移学习（可选）

配置文件中 model.arch 指定使用哪种。
"""

import torch
import torch.nn as nn

CLASSES = ["B", "W", "E"]  # 黑子, 白子, 空


class StoneCNN(nn.Module):
    """
    专为围棋交叉点 patch 设计的轻量 3 层 CNN。

    输入: [N, 3, patch_size, patch_size]
    输出: [N, 3]  （logits，对应 B/W/E）

    参数量约 130K，推理极快，适合移动端。
    """
    def __init__(self, num_classes=3, patch_size=48):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 48→24
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            # Block 2: 24→12
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),

            # Block 3: 12→6
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),

            # Global Average Pooling → 128×1×1（任意输入尺寸均可，MPS 兼容）
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_mobilenet_v3(num_classes=3, pretrained=True):
    """
    MobileNetV3-Small 预训练模型（迁移学习版本）。
    精度比 StoneCNN 高约 0.5-1%，但模型更大（2.5MB vs 0.5MB）。

    需要安装: pip install torchvision
    """
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    # 替换最后分类层
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(arch="StoneCNN", num_classes=3, patch_size=48):
    """
    根据配置构建并返回模型。

    Args:
        arch:        "StoneCNN" 或 "MobileNetV3"
        num_classes: 分类数（默认3）
        patch_size:  输入 patch 的边长像素

    Returns:
        model (nn.Module)
    """
    if arch == "StoneCNN":
        model = StoneCNN(num_classes=num_classes, patch_size=patch_size)
        print(f"[模型] StoneCNN — 参数量: {sum(p.numel() for p in model.parameters()):,}")
    elif arch == "MobileNetV3":
        model = get_mobilenet_v3(num_classes=num_classes, pretrained=True)
        print(f"[模型] MobileNetV3-Small (预训练) — 参数量: {sum(p.numel() for p in model.parameters()):,}")
    else:
        raise ValueError(f"未知模型架构: {arch}，请选择 StoneCNN 或 MobileNetV3")

    return model


def detect_device():
    """
    自动检测最佳计算设备：
      CUDA (GTX1080) > Apple MPS (M系列Mac) > CPU
    """
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"[设备] 使用 GPU: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[设备] 使用 Apple MPS (M系列芯片)")
    else:
        device = torch.device("cpu")
        import multiprocessing
        cores = multiprocessing.cpu_count()
        print(f"[设备] 使用 CPU ({cores} 核)")
    return device
