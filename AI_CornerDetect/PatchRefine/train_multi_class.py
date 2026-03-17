import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

from dataset_multi_class import MultiClassPatchDataset
from model_multi_class import PatchClassifier

def train_multi_class():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ROOT_DIR = r"E:\Data\Gomrade\MultiClassPatches"
    full_dataset = MultiClassPatchDataset(ROOT_DIR, is_train=False)
    
    # Split data: 80% Train, 10% Val, 10% Test
    n = len(full_dataset)
    indices = list(range(n))
    np.random.shuffle(indices)
    
    train_split = int(0.8 * n)
    val_split = int(0.9 * n)
    
    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]
    
    # Save split info to ensure test set isolation
    split_info = {
        'train': [str(full_dataset.samples[i][0]) for i in train_idx],
        'val': [str(full_dataset.samples[i][0]) for i in val_idx],
        'test': [str(full_dataset.samples[i][0]) for i in test_idx]
    }
    with open("split_info.json", "w") as f:
        json.dump(split_info, f)
    
    # Create Subsets
    train_dataset = MultiClassPatchDataset(ROOT_DIR, is_train=True)
    train_dataset.samples = [full_dataset.samples[i] for i in train_idx]
    
    val_dataset = MultiClassPatchDataset(ROOT_DIR, is_train=False)
    val_dataset.samples = [full_dataset.samples[i] for i in val_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = PatchClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Start with small LR
    
    best_acc = 0.0
    epochs = 20
    
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}: Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {avg_val_acc:.2f}%")
        
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), "checkpoints/best_patch_classifier.pth")
            print(f"⭐ Saved Best Model: {avg_val_acc:.2f}%")

    print(f"Training finished. Best Val Acc: {best_acc:.2f}%")
    print("Test set remains isolated and recorded in split_info.json")

if __name__ == "__main__":
    train_multi_class()
