
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

class AttentionCornerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 修正：只取 ResNet 的特征提取部分（去最后两层: avgpool 和 fc）
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 直线特征提取 (假设输入特征通道是 512)
        self.line_detector = nn.Conv2d(512, 2, 1)
        
        # 注意力模块
        self.attention = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # 角点检测头
        self.corner_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 14->28
            nn.ReLU(),
            nn.Conv2d(128, 4, 1) # 28x28 heatmap for 4 corners
        )
    
    def forward(self, x):
        feat = self.backbone(x) # [B, 512, 7, 7] for 224 input
        
        # 生成直线注意力图
        line_feat = self.line_detector(feat) # [B, 2, 7, 7]
        attn_map = self.attention(line_feat) # [B, 1, 7, 7]
        
        # 用注意力加权特征
        feat = feat * attn_map
        
        return self.corner_head(feat), attn_map

def test_on_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionCornerNet().to(device).eval()
    
    # 加载图像
    img_orig = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_orig).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output, attn_map = model(input_tensor)
    
    # 打印形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Attention map shape: {attn_map.shape}")
    print(f"Output (Corner Head) shape: {output.shape}")
    
    # 可视化注意力图
    # 将 7x7 的注意力图放大回 224x224 以便在原图上叠加
    attn_cpu = attn_map.squeeze().cpu().numpy()
    attn_resized = cv2.resize(attn_cpu, (224, 224))
    
    # 归一化到 0-255
    attn_vis = (attn_resized * 255).astype(np.uint8)
    attn_color = cv2.applyColorMap(attn_vis, cv2.COLORMAP_JET)
    
    # 处理原图
    img_224 = np.array(img_orig.resize((224, 224)))
    overlay = cv2.addWeighted(img_224, 0.6, attn_color, 0.4, 0)
    
    # 保存结果
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_224)
    plt.title("Origin (224x224)")
    
    plt.subplot(1, 3, 2)
    plt.imshow(attn_resized, cmap='jet')
    plt.title("Attention Map (7x7 -> 224x224)")
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    
    plt.savefig("output/attention_test_result.png")
    print("Result saved as output/attention_test_result.png")

if __name__ == "__main__":
    # 使用用户数据集中的一张图
    sample_img = r"e:\Data\Gomrade\kaggle-gomrade\dataset1\1\11.png"
    test_on_image(sample_img)
