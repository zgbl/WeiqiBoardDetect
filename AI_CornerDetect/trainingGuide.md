# 围棋棋盘角点检测 AI 训练指南 (AI Corner Detection Training Guide)

本指南旨在帮助你在 Mac (MPS) 和 Windows (NVIDIA CUDA) 环境下顺利训练并部署棋盘定位模型。

## 1. 项目结构
```text
AI_CornerDetect/
├── models/
│   ├── model.py        # Mac 优化模型定义
│   └── model_win.py    # Windows 优化模型定义
├── utils/
│   └── dataset.py      # 数据解析 + 坐标自动排序逻辑 (TL, TR, BR, BL)
├── train.py            # Mac (Metal/MPS) 训练脚本
├── train_win.py        # Windows (CUDA/1080) 训练脚本
├── verify_dataset.py   # 数据集坐标验证工具
├── inference.py        # 单张图片预测与可视化脚本
├── requirements.txt    # 依赖库列表
└── trainingGuide.md    # 你正在阅读的指南
```

---

## 2. 环境配置

### Mac 环境 (M1/M2/Intel)
1. 切换到你的 Python 3.10+ 环境。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### Windows 环境 (RTX 1080 / Python 3.12)
1. 创建环境：
   ```bash
   conda create -n board-ai python=3.12
   conda activate board-ai
   ```
2. 安装 **CUDA 版 PyTorch** (RTX 1080 必备)：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. 安装其他依赖：
   ```bash
   pip install opencv-python pyyaml tqdm matplotlib
   ```

---

## 3. 训练流程

### 第一步：验证标注数据 (Validation)
在开始长时间训练前，必须确认数据集坐标是否正确。
```bash
# Mac/Windows 通用
python verify_dataset.py --session 6
```
检查 `data_check/` 下生成的图片。
> **注意**：程序已内置 `sort_points` 逻辑，训练时会自动将标注点统一为顺时针顺序：[左上, 右上, 右下, 左下]。

### 第二步：开始训练 (Training)

#### Mac (使用 MPS 加速)
```bash
python train.py
```

#### Windows (使用 NVIDIA 1080 加速)
默认 `batch_size` 为 32。
```bash
python train_win.py --data "C:/你的数据集路径/Gomrade-dataset1" --batch_size 32
```

### 第三步：推理与效果检查 (Inference)
模型训练完成后，会在 `checkpoints/` 生成 `best_model.pth` (或 `best_model_win.pth`)。
```bash
python inference.py --img "path/to/your/image.jpg" --model checkpoints/best_model.pth
```

---

## 4. 关键算法说明 (V2.0 升级记录)
1. **坐标归一化**：模型预测的是 0.0 到 1.0 之间的相对坐标，不锁定分辨率。
2. **SmoothL1Loss**：相比 MSE，它对异常值（离群点）更稳健，能使回归结果更精细。
3. **ColorJitter 增强**：在训练过程随机调整亮度、对比度，模拟真实室内灯光阴影。
4. **线性回归输出**：移除了最后的 Sigmoid 激活函数，解决了模型在图像边缘位置“学习困难”的问题，大幅提升定位精度。

## 5. 故障排除
* **Windows 下 BrokenPipeError**：如果报多线程错误，尝试在命令后加 `--num_workers 0`。
* **预测点偏移**：若预测不准，请检查数据集中标注顺序是否极其不统一。如果偏移严重，建议增加训练 Epoch 到 100 以上。
