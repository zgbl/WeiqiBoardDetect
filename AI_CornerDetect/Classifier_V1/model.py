import torch
import torch.nn as nn
from torchvision import models

class PatchClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(PatchClassifier, self).__init__()
        # 使用基础但有效的 ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
