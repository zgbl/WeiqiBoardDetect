import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.model import CornerRegressor
from pathlib import Path

def predict(image_path, model_path, device='cpu'):
    # 1. 加载模型
    model = CornerRegressor(backbone='resnet18', pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 2. 预处理图像
    orig_img = cv2.imread(image_path)
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w = orig_img.shape[:2]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.fromarray(orig_img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # 3. 推理
    with torch.no_grad():
        output = model(input_tensor) # shape (1, 8)
        preds = output.cpu().numpy()[0]
    
    # 4. 反归一化坐标
    preds = preds.reshape(4, 2)
    preds[:, 0] *= w
    preds[:, 1] *= h
    
    # 5. 可视化
    plt.figure(figsize=(10, 8))
    plt.imshow(orig_img_rgb)
    
    # 绘制预测的点
    colors = ['red', 'green', 'blue', 'yellow']
    for i, (px, py) in enumerate(preds):
        plt.scatter(px, py, c=colors[i], s=100, label=f'Corner {i}')
    
    # 绘制连接线
    pts = preds.astype(np.int32)
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'cyan', lw=2)
        
    plt.legend()
    plt.title("Model Prediction")
    
    output_vis = Path("data_check") / f"pred_{Path(image_path).name}"
    plt.savefig(output_vis)
    plt.show()
    print(f"Prediction saved to {output_vis}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to test image")
    parser.add_argument("--model", default="checkpoints/best_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    predict(args.img, args.model, device=device)
