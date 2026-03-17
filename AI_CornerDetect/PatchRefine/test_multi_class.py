import torch
import json
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dataset_multi_class import MultiClassPatchDataset
from model_multi_class import PatchClassifier
from sklearn.metrics import classification_report, confusion_matrix

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load split info
    if not Path("split_info.json").exists():
        print("Error: split_info.json not found. Run training first.")
        return
    
    with open("split_info.json", "r") as f:
        split_info = json.load(f)
    
    test_paths = split_info['test']
    print(f"Test set size: {len(test_paths)}")

    # Load model
    model = PatchClassifier(num_classes=4).to(device)
    model_path = "checkpoints/best_patch_classifier.pth"
    if not Path(model_path).exists():
        print(f"Error: {model_path} not found.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create test dataset logic
    # We can reuse the dataset class but override the samples
    full_dataset = MultiClassPatchDataset(r"E:\Data\Gomrade\MultiClassPatches", is_train=False)
    
    # Filter only test paths
    test_samples = []
    class_name_map = {0: 'corner', 1: 'edge', 2: 'inner', 3: 'outer'}
    
    # Convert samples to a lookup for speed if needed, but here we just match
    test_path_set = set(test_paths)
    for path, cid in full_dataset.samples:
        if str(path) in test_path_set:
            test_samples.append((path, cid))
    
    full_dataset.samples = test_samples
    test_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    target_names = ['Corner', 'Edge', 'Inner', 'Outer']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Identify corners (class 0)
    corner_indices = [i for i, label in enumerate(all_labels) if label == 0]
    correct_corners = sum(1 for i in corner_indices if all_preds[i] == 0)
    print(f"\nCorner Accuracy: {correct_corners}/{len(corner_indices)} ({100.*correct_corners/len(corner_indices):.2f}%)")

    # Visualization of some predictions
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Saving sample predictions to {output_dir}...")
    for i in range(min(20, len(test_samples))):
        img_path, class_id = test_samples[i]
        pred_id = all_preds[i]
        
        img = cv2.imread(str(img_path))
        label_text = f"GT: {target_names[class_id]} PRED: {target_names[pred_id]}"
        color = (0, 255, 0) if class_id == pred_id else (0, 0, 255)
        
        cv2.putText(img, label_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(str(output_dir / f"test_{i}.png"), img)

if __name__ == "__main__":
    test_model()
