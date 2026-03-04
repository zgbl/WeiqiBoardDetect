# 围棋棋子分类器 — Windows 训练指南

**适用项目**: WeiqiBoardDetect  
**适用环境**: Windows 10/11 + NVIDIA GPU (GTX 1080)
**更新**: 2026-03

---

## 一、Windows 专用文件说明

在 Windows 环境下，为避免控制台乱码和路径转义问题，请使用以下专用文件：

```
training/
├── config_windows.yaml  ← ★ Windows 路径和参数配置 (先改这里)
├── model_win.py         ← 模型定义 (Windows 优化：全英文日志)
├── train-win.py         ← 训练脚本 (Windows 优化：全英文日志)
├── extract_patches.py   ← 提取 patch (与 Mac 通用)
├── patches/             ← 自动生成：提取出的 patch 数据集
└── checkpoints/         ← 自动生成：训练结果
```

---

## 二、环境安装 (Windows)

### 1. 验证 CUDA 环境
我已检测到你的系统环境如下：
- **GPU**: NVIDIA GeForce GTX 1080
- **Driver**: 581.80
- **CUDA Toolkit**: 13.1

### 2. 安装 CUDA 版 PyTorch (非破坏性)
目前系统中安装的默认 Python 是 3.14，但 **PyTorch 目前不支持 Python 3.14**。
我检测到你系统中已安装了 **Python 3.12**，请务必使用 3.12 来运行本项目。

运行以下命令，专门为你的 Python 3.12 环境安装 CUDA 加速版：
```powershell
& "C:\Users\carso\AppData\Local\Python\pythoncore-3.12-64\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

> [!IMPORTANT]
> 安装完成后，之后启动训练也请指定使用 3.12：
> `& "C:\Users\carso\AppData\Local\Python\pythoncore-3.12-64\python.exe" train-win.py --config config_windows.yaml`

### 3. 安装其他依赖
```powershell
pip install pyyaml opencv-python tqdm Pillow scikit-learn matplotlib
```

---

## 三、修改配置 (config_windows.yaml)

打开 `training/config_windows.yaml`，确保路径指向你的实际位置。建议点击编辑器上方的 **Sync** 按钮确保配置同步。

**示例路径配置**:
```yaml
data:
  gomrade_dir: "D:/Codes/Data/Gomrade/dataset1"
  patches_dir: "D:/Codes/Data/Gomrade/dataset1/patches"

model:
  output_dir: "D:/Codes/WeiqiBoardDetect/WeiqiBoardDetect-main/training/checkpoints"
```

---

## 四、启动训练

### Step 1: 提取 Patch (如已从 Mac 拷贝则跳过)
使用 Windows 优化版提取脚本（已修复 UTF-8 编码问题）：
```powershell
cd training
python extract_patches_win.py --config config_windows.yaml
```

### Step 2: 执行训练
使用 Windows 优化版脚本，该脚本已解决 `UnicodeEncodeError`：

```powershell
python train-win.py --config config_windows.yaml
```

**常用参数**:
- `--subset 0.1`: 仅使用 10% 数据（极速迭代，建议调试阶段使用）
- `--epochs 30`: 训练轮数（先看前 30 轮 Loss 是否下降）
- `--workers 0`: Windows 建议设为 0 以避免多进程问题

> [!TIP]
> **快速迭代建议**:
> 在初步调参阶段，建议保持 `subset_ratio: 0.1` 和 `epochs: 30`。这样在 GTX 1080 上大约 **20-30 分钟** 就能看到完整的训练曲线。如果 `Val Acc` 能稳步上升到 90% 以上，说明参数方向正确，此时再切换到全量数据进行长时训练。

---

## 五、预期性能 (GTX 1080)

- **速度**: 开启 AMP (混合精度) 后，每轮训练约 2-4 分钟。
- **显存**: 建议 batch_size 设为 32 或 64。
- **结果**: 训练完成后，在 `checkpoints/` 目录下查看 `training_curve.png`。

---

## 六、常见问题

1. **乱码/报错**: 如果控制台显示 `UnicodeEncodeError`，通常是因为使用了 `train.py`。请务必使用 `train-win.py`。
2. **路径问题**: YAML 配置文件中请统一使用正斜杠 `/`，不要使用反斜杠 `\`。

---

## 七、永久解决 Windows 编码问题 (推荐)

如果你不想每次都使用专用脚本，可以通过以下方式让 Windows 10 的 Python 默认使用 UTF-8：

### 方法：设置环境变量 `PYTHONUTF8`
1. 在搜索框输入 "环境变量"，选择 **"编辑系统环境变量"**。
2. 点击 **环境变量** 按钮。
3. 在 **"用户变量"** 下点击 **新建**：
   - 变量名：`PYTHONUTF8`
   - 变量值：`1`
4. 保存并重启 PowerShell/VS Code。

**或者直接在 PowerShell 中运行以下命令（已定位到 D 盘上下文）**:
```powershell
setx PYTHONUTF8 1
```
*设置后，Python 将全局默认使用 UTF-8，彻底解决 charmap 解码错误。*
