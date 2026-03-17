import torch
import torch.nn as nn
from torchvision import models

class PatchDetector(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(PatchDetector, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # 移除原有的 FC 层

        # 头部 1: 分类 (0:BG, 1:TL, 2:TR, 3:BR, 4:BL)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
        
        # 头部 2: 坐标回归 (x, y)
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        cls_logits = self.classifier(features)
        reg_coords = self.regressor(features)
        return cls_logits, reg_coords
