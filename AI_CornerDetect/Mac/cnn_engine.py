import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import sys
from pathlib import Path

# 配置路径，以便加载原始 Classifier_V1 模型定义
MAC_DIR = Path(__file__).parent
CLASSIFIER_V1_DIR = MAC_DIR.parent / "Classifier_V1"
if str(CLASSIFIER_V1_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_V1_DIR))

try:
    from model import PatchClassifier
except ImportError:
    class PatchClassifier(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.backbone = models.resnet18(weights=None)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        def forward(self, x): return self.backbone(x)

class CNNVerifier:
    """
    模块 2: CNN 角点真实验证器
    完全独立，它不知道也不关心角点是怎么来的（人点的、OpenCV找的、传统算法算出来的）
    只负责验证某个点是不是物理真角！
    """
    def __init__(self, weights_path):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"[*] CNN 加载在: {self.device}")
        
        self.model = PatchClassifier(num_classes=4)
        self.model.to(self.device)
        
        ckpt = torch.load(weights_path, map_location=self.device)
        state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("backbone.") and "backbone." + k in self.model.state_dict():
                new_state_dict["backbone." + k] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['Corner', 'Edge', 'Inner', 'Outer']

    def verify_point(self, img, pt, crop_radius=24):
        """接受一个确切的 x, y 点和原图，返回它的分类及置信度"""
        x_int, y_int = int(pt[0]), int(pt[1])
        x1, y1 = max(0, x_int - crop_radius), max(0, y_int - crop_radius)
        x2, y2 = min(img.shape[1], x_int + crop_radius), min(img.shape[0], y_int + crop_radius)
        
        patch_bgr = img[y1:y2, x1:x2]
        h, w = patch_bgr.shape[:2]
        expected_size = crop_radius * 2
        if h < expected_size or w < expected_size:
            patch_bgr = cv2.copyMakeBorder(patch_bgr, 0, max(0, expected_size - h), 0, max(0, expected_size - w), cv2.BORDER_CONSTANT, value=(128,128,128))
            
        patch_rgb = cv2.cvtColor(cv2.resize(patch_bgr, (48, 48)), cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(Image.fromarray(patch_rgb)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred = torch.max(probs, 0)
            
        return self.classes[pred.item()], conf.item()

    def verify_corners(self, img, corners_list):
        """
        验证外部传入的一系列角点，返回所有判断结果的结构化字典List
        """
        results = []
        for pt in corners_list:
            label, conf = self.verify_point(img, pt)
            results.append({
                "point": pt,
                "is_corner": (label == "Corner"),
                "label": label,
                "confidence": conf
            })
        return results
