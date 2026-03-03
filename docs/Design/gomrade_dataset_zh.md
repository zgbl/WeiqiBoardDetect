# Gomrade 数据集分析与利用方案

**数据集**: Gomrade: Go (Baduk) images with annotations  
**路径**: `/Users/tuxy/Codes/AI/Data/Gomrade-dataset1/`  
**更新**: 2026-03

---

## 一、数据集结构解析

### 总体规模（上半部分）
- **约 1,579 个标注样本**（39个对局文件夹）
- **来源多样**：真实比赛场景（意大利围棋联赛，多比赛多光线条件）
- **最大子集**: `20_03_28_17_58_40/` → 418帧（单场对局完整追踪）

### 文件格式（每个样本三个文件）

#### `N.png` — 原始图像
- 真实摄像头拍摄，约 1.3-4MB
- 透视未校正的斜视图或俯视图
- 包含手部、环境遮挡等真实干扰

#### `N.txt` — 棋盘状态标注 ⭐
```
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . B . . . . . . . . . . . . .
. . . W . . . . . . . . . . . B . . .
...
```
- 19行×19列，空格分隔
- `B` = 黑子，`W` = 白子，`.` = 空

**这是完整的棋盘状态标注，不是bounding box，可以直接提取patch！**

#### `board_extractor_state.yml` — 透视变换参数
```yaml
M:                        # 3×3 透视矩阵（已标定！）
  - [1.298, 0.198, -605.148]
  - [-0.028, 1.329, -93.702]
  - [-1.5e-05, 2.44e-04, 1.0]
pts_clicks:               # 人工标注的4个角点（像素坐标）
  - [454, 80]
  - [1599, 104]
  - [1754, 1338]
  - [264, 1325]
```

#### `board_state_classifier_state.yml` — 颜色分类器状态
```yaml
black_colors: [...]        # 黑子的 LAB 颜色模板
white_colors: [...]        # 白子的 LAB 颜色模板
board_colors: [...]        # 棋盘木色模板（6个颜色锚点）
x_grid: [0, 69, 139...]   # 校正后图像中各列的 x 坐标（像素）
y_grid: [0, 82, 165...]   # 校正后图像中各行的 y 坐标（像素）
height: 1490, width: 1259  # 校正后图像尺寸
```

**这意味着数据集提供了完整的透视校正矩阵 + 网格坐标！** 不需要计算，直接用。

---

## 二、数据集对 WeiqiBoardDetect 的直接价值

### 价值1：**1579 张真实标注图像，立即可用于测试**

目前 WeiqiBoardDetect 只有 35 张测试图像，准确率无法定量评估。  
用 Gomrade 数据集可以：
- 拿到完整棋盘状态标注（真值）
- 和程序检测结果对比，计算准确率
- **一行命令**批量测试1579张，得到客观准确率数字

### 价值2：**透视矩阵已标定，可直接提取 patch**

每个文件夹的 `board_extractor_state.yml` 里有现成的透视矩阵M和网格坐标，  
用这个矩阵 warpPerspective 图像，然后按 x_grid/y_grid 提取每个交点的 patch，  
对应 .txt 里的 B/W/. → **立刻获得几十万个已标注的交点 patch**！

### 价值3：**多样化光线和场景**

39个对局文件夹 = 不同比赛场地、不同照明、不同相机距离，  
这正是我们需要的真实世界多样性，合成数据无法替代。

---

## 三、使用方案

### 方案A：批量准确率评估（最优先）

写一个评估脚本，对所有1579帧：
1. 用 Gomrade 的透视矩阵校正图像
2. 运行 WeiqiBoardDetect 的棋子分类逻辑
3. 与 .txt 标注对比，计算 per-intersection 准确率

```python
# 伪代码
def evaluate_on_gomrade(dataset_dir, detector):
    total, correct = 0, 0
    for session_dir in dataset_dir.iterdir():
        M = load_yml(session_dir / "board_extractor_state.yml")["M"]
        for txt_file in session_dir.glob("*.txt"):
            img = cv2.imread(str(txt_file.with_suffix(".png")))
            warped = cv2.warpPerspective(img, M, (width, height))
            gt_state = parse_txt(txt_file)   # 真值
            pred_state = detector.classify(warped)  # 预测
            correct += compare(gt_state, pred_state)
            total += 19 * 19
    print(f"准确率: {correct/total:.3%}")
```

### 方案B：提取 patch 训练 CNN（与 train_classifier.py 对接）

```python
def extract_patches_from_gomrade(dataset_dir, output_dir):
    """
    利用数据集内置的透视矩阵和网格坐标，
    提取所有交叉点 patch，存入 B/W/E 分类目录
    """
    for session_dir in dataset_dir.iterdir():
        state_yml = load_yml(session_dir / "board_state_classifier_state.yml")
        ext_yml   = load_yml(session_dir / "board_extractor_state.yml")
        
        M       = np.array(ext_yml["M"])
        x_grid  = state_yml["x_grid"]   # 长度19，每列x坐标
        y_grid  = state_yml["y_grid"]   # 长度19，每行y坐标
        W, H    = ext_yml["max_width"], ext_yml["max_height"]
        
        for txt_file in session_dir.glob("*.txt"):
            img = cv2.imread(str(txt_file.with_suffix(".png")))
            warped = cv2.warpPerspective(img, M, (W, H))
            gt = parse_txt(txt_file)  # 19x19 列表
            
            for r, row in enumerate(gt):
                for c, label in enumerate(row):
                    x, y = x_grid[c], y_grid[r]
                    patch = crop_patch(warped, x, y, radius=30)
                    save_patch(patch, label, output_dir)
```

预期产出：**100,000+ 个真实场景 patch**（1579帧 × 约 60-80 个有子格子）

---

## 四、优先行动步骤

| 优先级 | 任务 | 预计耗时 |
|---|---|---|
| ★★★ | 写 `gomrade_eval.py`：批量评估 WeiqiBoardDetect 准确率 | 2-3小时 |
| ★★★ | 写 `gomrade_patch_extractor.py`：提取100K+ patch | 2小时 |
| ★★☆ | 将 Gomrade patches 加入 `train_classifier.py` 训练 | 1小时 |
| ★☆☆ | 分析不同光线/场地下准确率差异，定向改进 | 1天 |

---

## 五、数据量估算

```
Gomrade 数据集上半部分：
  约 1,579 帧图像
  × 19×19 = 361 个交叉点 / 帧
  = 约 570,000 个带标注的交叉点

其中有子格子（黑/白）按平均密度40%计算：
  ≈ 228,000 个黑/白 patch
  约 342,000 个空格 patch

总计：~570,000 个高质量真实场景 patch ✅
```

这比合成数据（10,000张）质量高，多样性强 **57倍**。  
**应优先用 Gomrade 真实数据训练分类器，合成数据作为补充。**
