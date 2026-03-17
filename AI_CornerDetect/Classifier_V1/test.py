import torch
import json
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import MultiClassPatchDataset
from model import PatchClassifier
from sklearn.metrics import classification_report

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载隔离的测试集信息
    if not Path("test_set_isolation.json").exists():
        print("Error: test_set_isolation.json not found.")
        return
    
    with open("test_set_isolation.json", "r") as f:
        iso_info = json.load(f)
    test_paths = set(iso_info['test_set'])
    print(f"Loading test set: {len(test_paths)} samples")

    # 配置模型
    model = PatchClassifier(num_classes=4).to(device)
    state_dict = torch.load("weights/best_classifier.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 加载数据
    full_dataset = MultiClassPatchDataset(r"E:\Data\Gomrade\MultiClassPatches", is_train=False)
    # 过滤，只留下测试集路径
    test_samples = []
    for path, cid in full_dataset.samples:
        if str(path) in test_paths:
            test_samples.append((path, cid))
    
    full_dataset.samples = test_samples
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            outputs = model(imgs.to(device))
            _, p = outputs.max(1)
            preds.extend(p.cpu().numpy())
            labels.extend(lbls.numpy())

    # 输出报告
    target_names = ['Corner', 'Edge', 'Inner', 'Outer']
    print("\nTest Set Evaluation:")
    print(classification_report(labels, preds, target_names=target_names))

    # 可视化结果
    vis_dir = Path("test_vis")
    vis_dir.mkdir(exist_ok=True)
    print(f"Saving samples to {vis_dir}")
    for i in range(min(50, len(test_samples))):
        img_path, cid = test_samples[i]
        img = cv2.imread(str(img_path))
        p_id = preds[i]
        color = (0, 255, 0) if cid == p_id else (0, 0, 255)
        text = f"GT:{target_names[cid]} P:{target_names[p_id]}"
        cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(str(vis_dir / f"test_{i}.png"), img)

if __name__ == "__main__":
    test()
