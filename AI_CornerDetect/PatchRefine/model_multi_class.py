import torch
import torch.nn as nn
from torchvision import models

class PatchClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(PatchClassifier, self).__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    model = PatchClassifier(4)
    dummy = torch.randn(1, 3, 128, 128)
    out = model(dummy)
    print(out.shape)
