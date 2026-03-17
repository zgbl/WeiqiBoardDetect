import torch
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from model import PatchClassifier

def verify_multiscale():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 路径配置
    BASE_DIR = Path(r"E:\Data\Gomrade\MultiClassPatches")
    
    # 加载已训练模型
    model = PatchClassifier(num_classes=4).to(device)
    model_path = Path("weights/best_classifier.pth")
    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}. Please train the model first.")
        return
    
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_names = {0: 'corner', 1: 'edge', 2: 'inner', 3: 'outer'}
    source_categories = ['corner', 'edge', 'inner', 'outer']

    # 均值和标准差 (ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    for src_cat in source_categories:
        src_dir = BASE_DIR / src_cat
        if not src_dir.exists():
            print(f"Skipping {src_cat}: Folder not found.")
            continue
            
        # 扫描该目录下直接存放的图片文件
        files = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        if not files:
            print(f"No flat files to process in {src_cat}.")
            continue
            
        print(f"\nVerifying folder: {src_cat} ({len(files)} files)")

        # 在当前目录下创建预测类别的子目录
        for name in class_names.values():
            (src_dir / name).mkdir(exist_ok=True)
        (src_dir / 'unsure').mkdir(exist_ok=True)

        for img_path in tqdm(files, desc=f"Processing {src_cat}"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            
            # 预处理
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(device)
            
            # 归一化
            img_tensor = (img_tensor - mean) / std
            
            with torch.no_grad():
                output = model(img_tensor.unsqueeze(0))
                probs = torch.softmax(output, dim=1)
                conf, pred_id = torch.max(probs, 1)
            
            conf_val = conf.item()
            pred_id_val = pred_id.item()
            
            # 分类逻辑：置信度门槛设定为 0.8
            if conf_val > 0.8:
                dest_cat = class_names[pred_id_val]
            else:
                dest_cat = 'unsure'
                
            dest_folder = src_dir / dest_cat
            dest_path = dest_folder / img_path.name
            
            # 移动文件
            try:
                shutil.move(str(img_path), str(dest_path))
            except Exception as e:
                print(f"Error moving {img_path.name}: {e}")

    print("\nMultiscale verification and sorting complete.")

if __name__ == "__main__":
    verify_multiscale()
