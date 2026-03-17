import torch
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from model import PatchClassifier
import os

def refine_dirty():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 路径配置
    DIRTY_DIR = Path(r"E:\Data\Gomrade\MultiClassPatches\dirty")
    
    if not DIRTY_DIR.exists():
        print(f"Directory {DIRTY_DIR} not found. Skipping refinement.")
        return

    # 在 dirty 目录下创建 4 个子目录
    categories = ['corner', 'edge', 'inner', 'outer', 'unsure']
    for cat in categories:
        (DIRTY_DIR / cat).mkdir(parents=True, exist_ok=True)

    # 加载已训练模型
    model = PatchClassifier(num_classes=4).to(device)
    model_path = Path("weights/best_classifier.pth")
    if not model_path.exists():
        print(f"Model weights not found at {model_path}. Please train the model first.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_names = {0: 'corner', 1: 'edge', 2: 'inner', 3: 'outer'}
    
    # 扫描 dirty 目录下直接存放的图片（不包含子目录里的）
    files = [f for f in DIRTY_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg']]
    print(f"Processing {len(files)} files in DIRTY directory...")

    moved_count = {name: 0 for name in categories}

    for img_path in tqdm(files):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        
        # 预处理
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            conf, pred_id = torch.max(probs, 1)
        
        conf_val = conf.item()
        
        # 分类逻辑：置信度大于 0.8 时放入对应子目录，否则放入 unsure 目录
        if conf_val > 0.8:
            dest_cat = class_names[pred_id.item()]
        else:
            dest_cat = 'unsure'
            
        dest_path = DIRTY_DIR / dest_cat / img_path.name
        
        # 避免重名覆盖
        if dest_path.exists():
            dest_path = DIRTY_DIR / dest_cat / f"refined_{img_path.name}"
            
        shutil.move(str(img_path), str(dest_path))
        moved_count[dest_cat] += 1

    print("\nDirty folder classification complete.")
    for cat, count in moved_count.items():
        print(f"  Moved to dirty/{cat}: {count} files")

if __name__ == "__main__":
    refine_dirty()
