import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.model_win import CornerRegressor
import os

def visualize_inference(image_path, model_path, device='cpu'):
    # 1. 加载模型
    print(f"Loading model from {model_path}")
    model = CornerRegressor(backbone='resnet18', pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查checkpoint格式并加载
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # 2. 加载原始图像
    print(f"Loading image from {image_path}")
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Failed to load image {image_path}")
        return
    
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_img.shape[:2]
    print(f"Original image size: {orig_w} x {orig_h}")
    
    # 3. 预处理图像（模型看到的输入）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.fromarray(orig_img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # 4. 创建一个可视化的224x224图像（反标准化后）
    # 用于显示模型实际看到的图像
    img_224 = cv2.resize(orig_img_rgb, (224, 224))
    
    # 5. 推理
    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)
        raw_preds = output.cpu().numpy()[0]
    
    print(f"Raw model output (8 values for 4 corners):")
    for i in range(0, 8, 2):
        print(f"  Corner {i//2 + 1}: ({raw_preds[i]:.4f}, {raw_preds[i+1]:.4f})")
    
    # 检查输出范围
    if np.any(raw_preds < 0) or np.any(raw_preds > 1):
        print(f"⚠️ WARNING: Output values range from {raw_preds.min():.3f} to {raw_preds.max():.3f}")
        print("   Model might need a sigmoid activation at the end!")
    else:
        print(f"✅ Output values in valid range [0, 1]")
    
    # 6. 重塑坐标为4个点
    pred_points_norm = raw_preds.reshape(4, 2)
    
    # 7. 创建多个可视化结果
    
    # 创建输出目录
    os.makedirs("visualization_output", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 7.1 在224x224图像上绘制预测点
    img_224_draw = img_224.copy()
    for i, (x, y) in enumerate(pred_points_norm):
        px, py = int(x * 224), int(y * 224)
        # 绘制点
        cv2.circle(img_224_draw, (px, py), 5, (255, 0, 0), -1)  # 蓝色填充圆
        cv2.circle(img_224_draw, (px, py), 8, (0, 255, 0), 2)   # 绿色边框
        # 添加标签
        cv2.putText(img_224_draw, str(i+1), (px+10, py-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 绘制四边形
    points = (pred_points_norm * 224).astype(np.int32)
    cv2.polylines(img_224_draw, [points], True, (0, 255, 0), 2)
    
    # 保存224x224结果
    cv2.imwrite(f"visualization_output/{base_name}_224_view.jpg", 
                cv2.cvtColor(img_224_draw, cv2.COLOR_RGB2BGR))
    print(f"Saved: visualization_output/{base_name}_224_view.jpg")
    
    # 7.2 在原始图像上绘制预测点
    orig_img_draw = orig_img.copy()
    for i, (x, y) in enumerate(pred_points_norm):
        px, py = int(x * orig_w), int(y * orig_h)
        # 绘制点
        cv2.circle(orig_img_draw, (px, py), 10, (0, 0, 255), -1)  # 红色填充圆
        cv2.circle(orig_img_draw, (px, py), 15, (0, 255, 0), 3)   # 绿色边框
        # 添加标签
        cv2.putText(orig_img_draw, str(i+1), (px+20, py-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    # 绘制四边形
    orig_points = (pred_points_norm * [orig_w, orig_h]).astype(np.int32)
    cv2.polylines(orig_img_draw, [orig_points], True, (0, 255, 0), 3)
    
    # 保存原始图像结果
    cv2.imwrite(f"visualization_output/{base_name}_original_view.jpg", orig_img_draw)
    print(f"Saved: visualization_output/{base_name}_original_view.jpg")
    
    # 7.3 创建对比图（matplotlib）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original Image\n{orig_w}x{orig_h}")
    axes[0].axis('off')
    
    # 224x224模型输入
    axes[1].imshow(img_224)
    axes[1].set_title("Model Input (224x224)\nBefore Normalization")
    axes[1].axis('off')
    
    # 带预测点的图像
    axes[2].imshow(cv2.cvtColor(orig_img_draw, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Predicted Corners\nValues: [{raw_preds[0]:.2f}, {raw_preds[1]:.2f}, ...]")
    axes[2].axis('off')
    
    # 添加预测坐标文本
    text_str = "Predicted normalized coordinates:\n"
    for i, (x, y) in enumerate(pred_points_norm):
        text_str += f"  Corner {i+1}: ({x:.3f}, {y:.3f})\n"
    
    plt.suptitle(text_str, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"visualization_output/{base_name}_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: visualization_output/{base_name}_comparison.png")
    
    # 8. 额外诊断：显示特征图（可选）
    # 这需要修改模型来返回中间特征图
    try:
        # 尝试获取中间特征图（如果模型支持）
        print("\nAttempting to get intermediate features...")
        
        # 创建一个钩子来捕获特征图
        features = []
        def hook_fn(module, input, output):
            features.append(output.detach().cpu())
        
        # 注册钩子到最后一个卷积层
        if hasattr(model.backbone, 'layer4'):
            handle = model.backbone.layer4.register_forward_hook(hook_fn)
            
            # 再次推理
            with torch.no_grad():
                _ = model(input_tensor)
            
            handle.remove()
            
            if features:
                # 可视化特征图
                feat_map = features[0][0]  # [C, H, W]
                print(f"Feature map shape: {feat_map.shape}")
                
                # 显示前16个特征图
                n_feats = min(16, feat_map.shape[0])
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                for i in range(n_feats):
                    ax = axes[i // 4, i % 4]
                    ax.imshow(feat_map[i], cmap='viridis')
                    ax.axis('off')
                    ax.set_title(f'Channel {i}')
                
                plt.suptitle("Feature Maps from Last Conv Layer")
                plt.tight_layout()
                plt.savefig(f"visualization_output/{base_name}_feature_maps.png")
                plt.show()
                print(f"Saved: visualization_output/{base_name}_feature_maps.png")
    except Exception as e:
        print(f"Feature map visualization not available: {e}")
    
    print("\n✅ Visualization complete! Check the 'visualization_output' folder.")
    return pred_points_norm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    visualize_inference(args.img, args.model, device=args.device)