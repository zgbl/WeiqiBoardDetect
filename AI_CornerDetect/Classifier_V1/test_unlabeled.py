import torch
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from model import PatchClassifier
import os

def test_unlabeled():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 路径配置
    TEST_DIR = Path(r"E:\Data\WeiqiPics\patches")
    OUTPUT_DIR = Path(r"E:\Data\WeiqiPics\classified_results")
    
    if not TEST_DIR.exists():
        print(f"Directory {TEST_DIR} not found. Please extract patches first.")
        return

    # 初始化四类子文件夹用于保存结果观察
    categories = ['corner', 'edge', 'inner', 'outer', 'unsure']
    for cat in categories:
        (OUTPUT_DIR / cat).mkdir(parents=True, exist_ok=True)

    # 加载已训练模型
    model = PatchClassifier(num_classes=4).to(device)
    model_path = Path("weights/best_classifier.pth")
    if not model_path.exists():
        print(f"Model weights not found at {model_path}. Please train the model first.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_names = {0: 'corner', 1: 'edge', 2: 'inner', 3: 'outer'}
    
    files = list(TEST_DIR.glob("*.png")) + list(TEST_DIR.glob("*.jpg"))
    print(f"Found {len(files)} patches to classify.")

    moved_count = {name: 0 for name in categories}

    for img_path in tqdm(files, desc="Classifying"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        
        # 预处理 (与训练时必须保持绝对一致)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        
        # 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            conf, pred_id = torch.max(probs, 1)
        
        conf_val = conf.item()
        
        # 使用 0.8 的置信度门槛，低于此值的归入 unsure，方便我们观察模型的不确定性
        if conf_val > 0.8:
            dest_cat = class_names[pred_id.item()]
        else:
            dest_cat = 'unsure'
            
        dest_path = OUTPUT_DIR / dest_cat / img_path.name
        
        # 拷贝图片到对应分类结果目录中（不破坏原测试集）
        shutil.copy(str(img_path), str(dest_path))
        moved_count[dest_cat] += 1

    print("\nClassification complete. Results are saved in:")
    print(f"  --> {OUTPUT_DIR}\n")
    for cat, count in moved_count.items():
        print(f"  {cat}: {count} patches")

if __name__ == "__main__":
    test_unlabeled()
