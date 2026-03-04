"""
model_win.py — Weiqi Stone Classifier Model Definitions (Windows optimized)
================================================================
Provides two models:
  StoneCNN      - Custom lightweight CNN, optimized for patch classification (Recommended)
  MobileNetV3   - ImageNet pretrained transfer learning (Optional)

Specified via model.arch in config file.
"""

import torch
import torch.nn as nn

CLASSES = ["B", "W", "E"]  # Black, White, Empty


class StoneCNN(nn.Module):
    """
    Lightweight 3-layer CNN designed for Weiqi intersection patches.

    Input: [N, 3, patch_size, patch_size]
    Output: [N, 3] (logits, corresponding to B/W/E)

    Approx 130K parameters, extremely fast inference, mobile-ready.
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

            # Global Average Pooling → 128×1×1
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
    MobileNetV3-Small pretrained model (Transfer Learning version).
    Slightly better accuracy than StoneCNN (~0.5-1%), but larger (2.5MB vs 0.5MB).
    """
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    # Replace last classification layer
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(arch="StoneCNN", num_classes=3, patch_size=48):
    """
    Builds model based on architecture name.

    Args:
        arch:        "StoneCNN" or "MobileNetV3"
        num_classes: Number of classes (default 3)
        patch_size:  Input patch pixel dimension

    Returns:
        model (nn.Module)
    """
    if arch == "StoneCNN":
        model = StoneCNN(num_classes=num_classes, patch_size=patch_size)
        print(f"[MODEL] StoneCNN — Params: {sum(p.numel() for p in model.parameters()):,}")
    elif arch == "MobileNetV3":
        model = get_mobilenet_v3(num_classes=num_classes, pretrained=True)
        print(f"[MODEL] MobileNetV3-Small (Pretrained) — Params: {sum(p.numel() for p in model.parameters()):,}")
    else:
        raise ValueError(f"Unknown model architecture: {arch}. Please choose StoneCNN or MobileNetV3")

    return model


def detect_device():
    """
    Automatically detects best computing device.
    """
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"[DEVICE] Using GPU: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[DEVICE] Using Apple MPS (M-series chip)")
    else:
        device = torch.device("cpu")
        import multiprocessing
        cores = multiprocessing.cpu_count()
        print(f"[DEVICE] Using CPU ({cores} cores)")
    return device
