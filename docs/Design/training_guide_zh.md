# 围棋棋子分类器 — 训练完整指南

**适用项目**: WeiqiBoardDetect  
**训练数据**: Gomrade 真实对局数据集（主力）+ 合成数据（补充）  
**更新**: 2026-03

---

## 一、文件说明

```
training/
├── config.yaml          ← ★ 所有路径和参数配置（先改这里）
├── extract_patches.py   ← Step 1: 从 Gomrade 提取训练 patch
├── model.py             ← 模型定义（StoneCNN + MobileNetV3）
├── train.py             ← Step 2: 训练模型
├── patches/             ← 自动生成：提取出的 patch 数据集
│   ├── B/  ← 黑子 patch（约 228,000 个）
│   ├── W/  ← 白子 patch
│   └── E/  ← 空格 patch（约 342,000 个）
└── checkpoints/         ← 自动生成：训练结果
    ├── best_model.pt          ← ★ 最优模型（用于部署）
    ├── latest_model.pt        ← 最新 checkpoint（用于续训）
    ├── stone_classifier.onnx  ← ONNX 模型（移动端部署）
    ├── training_curve.png     ← 训练曲线图
    ├── confusion_matrix.png   ← 混淆矩阵图
    ├── classification_report.txt ← 精确率/召回率报告
    └── training_summary.json  ← 训练摘要
```

---

## 二、环境安装

### Mac（CPU / Apple Silicon MPS）
```bash
pip install torch torchvision          # Apple Silicon: mps 自动启用
pip install pyyaml opencv-python tqdm Pillow
pip install scikit-learn matplotlib    # 用于混淆矩阵（可选但推荐）
```

### Windows PC（GTX 1080 CUDA）
```bash
# 先安装 CUDA 版 PyTorch（GTX 1080 支持 CUDA 11.x / 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install pyyaml opencv-python tqdm Pillow scikit-learn matplotlib
```

> **GTX 1080 特别提示**: 支持 AMP 混合精度训练，训练速度可提升 30-50%。
> config.yaml 中 `train.amp: true` 默认开启。

---

## 三、第一步：修改配置文件

打开 `training/config.yaml`，修改以下关键路径：

```yaml
data:
  gomrade_dir: "/Users/tuxy/Codes/AI/Data/Gomrade-dataset1"  # ← 改为你的路径
  patches_dir: "/Users/tuxy/Codes/AI/WeiqiBoardDetect/training/patches"

model:
  output_dir: "/Users/tuxy/Codes/AI/WeiqiBoardDetect/training/checkpoints"
```

**Windows 路径示例**（把文件拷到 Windows 后修改）：
```yaml
data:
  gomrade_dir: "C:/Users/yourname/Data/Gomrade-dataset1"
  patches_dir: "C:/Users/yourname/WeiqiBoardDetect/training/patches"
```

---

## 四、第二步：提取 Patch（一次完成，约 5-10 分钟）

```bash
cd training/

# 提取所有 patch（约 57 万个）
python3 extract_patches.py

# 调试：只处理前3个文件夹，快速验证格式
python3 extract_patches.py --limit 3
```

**提取完成后，会显示**：
```
✅ 提取完成！
   总计: 570,000+ 个 patch
   黑子 (B): 228,xxx
   白子 (W): 228,xxx
   空格 (E): 342,xxx
   输出目录: training/patches/
     B/  ← xxx 个黑子 patch
     W/  ← xxx 个白子 patch
     E/  ← xxx 个空格 patch
     summary.json
```

**提取一次，之后无需再次提取。**

---

## 五、第三步：训练模型

### Mac 上运行（CPU，速度较慢）
```bash
cd training/
python3 train.py --batch-size 32 --workers 2
```

### Windows GTX 1080（推荐这台机器训练）
```bash
cd training/
python3 train.py   # 自动使用 GPU，AMP 混合精度加速
```

### 快速测试（验证代码是否正常）
```bash
python3 train.py --epochs 3 --batch-size 32 --workers 0
```

### 断点续训（如果训练中断了）
```bash
python3 train.py --resume
```

---

## 六、训练过程：看什么指标

训练过程中每 5 个 epoch 打印一行：
```
Epoch  10/60  Train Loss=0.1234 Acc=0.9512  Val Loss=0.1189 Acc=0.9634  LR=0.000891  [12.3s]
★ 新最优: val_acc=0.9634 → 保存到 best_model.pt
```

| 指标 | 含义 | 目标值 |
|---|---|---|
| `Train Acc` | 训练集准确率 | > 0.95 |
| `Val Acc` | 验证集准确率 | > 0.97 |
| `Train Loss` | 训练损失 | 持续下降 |
| `★ 新最优` | 出现时自动保存 best_model.pt | - |

**正常训练曲线特征**：
- 前5轮：Val Acc 从约 0.7 快速提升到 0.9+
- 中间：Val Acc 缓慢提升，Loss 平稳下降  
- 后期：Val Acc 趋于稳定（0.97-0.99+）

---

## 七、训练完成的输出文件详解

### `checkpoints/best_model.pt` ★ 最重要
PyTorch 格式，保存了最优验证集准确率时的模型权重。

**未来的用法**（在检测脚本里加载）：
```python
import torch
from model import build_model

model = build_model("StoneCNN", num_classes=3, patch_size=48)
state = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(state["model"])
model.eval()

# 对一个 patch 分类
patch_tensor = ...  # [1, 3, 48, 48] tensor
with torch.no_grad():
    logits = model(patch_tensor)
    pred = logits.argmax().item()
    # pred: 0=黑子, 1=白子, 2=空
```

### `checkpoints/stone_classifier.onnx`
通用导出格式，可在以下平台部署：
- **iOS**: 转为 CoreML（使用 `coremltools`）
- **Android**: 用 ONNX Runtime Mobile
- **OpenCV DNN**：直接在 Python/C++ 中加载
- **服务器**: 任何支持 ONNX 的推理框架

```python
# 用 OpenCV 加载 ONNX 推理（不需要 PyTorch）
net = cv2.dnn.readNetFromONNX("stone_classifier.onnx")
blob = cv2.dnn.blobFromImage(patch, 1/255.0, (48, 48))
net.setInput(blob)
output = net.forward()
pred = output.argmax()  # 0=B, 1=W, 2=E
```

### `checkpoints/training_curve.png`
两个子图：Loss 和 Accuracy 随 epoch 的变化曲线。
用来判断是否过拟合（train acc 高但 val acc 低）。

### `checkpoints/confusion_matrix.png`
3×3 混淆矩阵，直观看出哪类别分错了：
```
               预测B  预测W  预测E
真实B    [ 9876   12     3  ]
真实W    [   8   9856    5  ]  
真实E    [   2     1  34201 ]
```

### `checkpoints/classification_report.txt`
文字版精确率/召回率报告：
```
              precision    recall  f1-score
B (黑子)         0.9978    0.9985    0.9981
W (白子)         0.9984    0.9976    0.9980
E (空格)         0.9999    0.9998    0.9999
accuracy                              0.9993
```

---

## 八、在 Mac → Windows 迁移步骤

1. 将整个 `WeiqiBoardDetect/` 文件夹和 `Data/` 文件夹复制到 Windows
2. 用 pip 安装 CUDA 版 PyTorch（见上方安装命令）
3. 修改 `training/config.yaml` 里的路径为 Windows 路径
4. 如果 Mac 上已提取好 patches，可以直接复制 `training/patches/` 到 Windows，**跳过 extract_patches.py**
5. 运行 `python3 train.py`（自动检测 GPU，使用 GTX 1080 + AMP 加速）

---

## 九、预期训练时间

| 设备 | 每 epoch 时间 | 60 轮总计 |
|---|---|---|
| Mac M1/M2 (MPS) | ~8-15 分钟 | ~10 小时 |
| Mac Intel (CPU) | ~15-30 分钟 | ~20 小时 |
| GTX 1080 (CUDA+AMP) | ~2-4 分钟 | **~2-4 小时** |

> **建议**: Mac 上先跑 `--epochs 5` 确认代码无误，然后转移到 GTX 1080 完整训练。
