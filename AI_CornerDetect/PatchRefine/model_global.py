import torch
import torch.nn as nn
from torchvision import models

class GlobalKeypointModel(nn.Module):
    def __init__(self, num_points=4):
        super(GlobalKeypointModel, self).__init__()
        # 使用 ResNet18 作为特征提取器
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 移除 FC 层和池化层
        self.features = nn.Sequential(*list(resnet.children())[:-2]) # 输出 7x7

        # 简单的上采样器 (从 7x7 恢复到 56x56)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_points, kernel_size=1) # 输出 4 通道热力图
        )

    def forward(self, x):
        x = self.features(x)
        x = self.deconv(x)
        return x
