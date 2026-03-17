import cv2
import numpy as np
import torch
from dataset_global import GlobalBoardDataset
from pathlib import Path

def diagnose():
    # 路径要和你训练时的一致
    DATA_DIRS = [
        r"E:\Data\Gomrade\kaggle-gomrade\dataset1",
        r"E:\Data\Gomrade\kaggle-gomrade\dataset2"
    ]
    
    dataset = GlobalBoardDataset(DATA_DIRS, is_train=True)
    out_dir = Path("data_diagnose")
    out_dir.mkdir(exist_ok=True)
    
    print(f"Sampling 5 images to check if labels and rotation are correct...")
    
    for i in range(5):
        idx = np.random.randint(0, len(dataset))
        img_tensor, hm_tensor = dataset[idx]
        
        # 还原图片 (反归一化)
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 还原热力图
        hms = hm_tensor.numpy()
        combined_hm = np.max(hms, axis=0) # 把4个角合在一起
        combined_hm = (combined_hm * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(combined_hm, cv2.COLORMAP_JET)
        
        # 叠加显示
        vis = cv2.addWeighted(img, 0.5, cv2.resize(heatmap_color, (224, 224)), 0.5, 0)
        
        cv2.imwrite(str(out_dir / f"check_{i}.png"), vis)
        print(f"Saved: {out_dir}/check_{i}.png")

if __name__ == "__main__":
    diagnose()
