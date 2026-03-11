import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class CornerRegressor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super(CornerRegressor, self).__init__()
        
        if backbone == 'resnet18':
            if pretrained:
                # 显式指定 weights 以兼容 PyTorch 最新版本
                self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet18(weights=None)
                
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 8)
            )
        elif backbone == 'mobilenet_v3':
            if pretrained:
                self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            else:
                self.model = models.mobilenet_v3_small(weights=None)
            in_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(256, 8)
            )
        else:
            raise ValueError("Unsupported backbone. Use 'resnet18' or 'mobilenet_v3'")

    def forward(self, x):
        return self.model(x)

# Windows 下如果要用 multiprocessing，建议在脚本外面定义类
