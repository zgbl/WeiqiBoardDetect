import torch
import torch.nn as nn
from torchvision import models

class PatchCornerRegressor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super(PatchCornerRegressor, self).__init__()
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
        elif backbone == 'mobilenet_v3':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.mobilenet_v3_small(weights=weights)
            in_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.Hardswish(),
                nn.Linear(128, 2)
            )
    def forward(self, x):
        return self.model(x)
