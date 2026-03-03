# WeiqiBoardDetect — 项目现状总结（2025年7月停止时的进度）

**更新日期**: 2026-03  
**分析对象**: `/src/Tran-BoardDect-wDrag-5B-6.py` + `Detectpiece.py`

---

## 一、这个项目比 OpenCVTest1 先进得多

重新找到这个项目，认识到去年做到的进展远超预期。以下是对比：

| 功能 | OpenCVTest1 | WeiqiBoardDetect（本项目） |
|---|---|---|
| 棋盘检测 | 无，依赖手动对准 | ✅ 3种自动检测方案(多备选) |
| 透视变换 | 基本实现 | ✅ 完整实现，含 padding 精细控制 |
| 交互式拖拽调整角点 | 无 | ✅ 完整 GUI 支持，鼠标拖拽确认 |
| 图片自动缩放 | 无 | ✅ 三档参数集（小/中/大图） |
| Hough 线检测 + 合并 | 无 | ✅ 水平/垂直线分类 + 相近线合并 |
| 棋格交点提取 | 无 | ✅ 精确交点计算 + 19×19 网格构建 |
| 棋子检测 | HSV mask + 分水岭 | ✅ HoughCircles + 区域检测（三方法合并） |
| 棋子颜色判定 | 固定阈值 | ✅ 均匀性过滤 + 中心ROI强度判定 |
| 棋盘状态输出 | 无 | ✅ 19×19矩阵 + 终端ASCII显示 |
| 虚拟棋盘生成 | 无 | ✅ 生成纯净棋盘+棋子图像 |
| 测试图片数量 | 7张 | ✅ **35张**（多角度、多光线、真实比赛照） |
| 代码规模 | ~200行 | ✅ **878行主脚本 + 323行 Detectpiece** |

---

## 二、当前代码架构

### 主脚本: `src/Tran-BoardDect-wDrag-5B-6.py`

```
[读取图像]
    │
    ▼
[auto_resize_image] → 自动缩放到合适处理尺寸，选参数集
    │
    ▼
[improve_edge_detection] → Canny + 边缘掩码（排除图像边界干扰）
    │
    ▼（3方案级联尝试）
[find_board_contour_improved] → 轮廓检测找四边形，验证合法性
[validate_board_corners]      → 对角线比例/边长比校验
    │ 找到角点
    ▼
[InteractiveCornerAdjuster] → GUI 拖拽确认角点（★已实现）
    │ 用户确认
    ▼
[getPerspectiveTransform + warpPerspective] → 透视变换到正方形
    │
    ▼
[HoughLinesP → 角度/长度过滤 → merge_lines] → 水平/垂直线
    │
    ▼
[line_intersection] → 计算所有交点
    │
    ▼
[19×19 网格构建 → grid_xy 矩阵]
    │
    ├─→ [HoughCircles] → 传统圆形检测（对标准棋子有效）
    └─→ [Detectpiece.detect_pieces_by_regions] → 区域三方法检测
    │
    ▼
[棋盘状态矩阵 + ASCII输出 + 虚拟棋盘图像]
```

### 棋子检测模块: `src/Detectpiece.py` (323行)

三种方法并行，results 合并去重：

| 方法 | 实现 | 优势 |
|---|---|---|
| `detect_by_threshold` | 双阈值二值化 + connectedComponents | 快速，对清晰画面有效 |
| `detect_by_contours` | Canny + findContours + 面积/位置过滤 | 对非圆形棋子鲁棒 |
| `detect_by_template_matching` | 每交叉点ROI统计分析 | **最关键**，不依赖全局检测 |

优先级：template > contour > threshold（同一交叉点有多个结果时取优先级高的）

---

## 三、遗留问题（去年遗留的难点）

从代码注释可以看出：

1. **部分图像无法检测**：
   ```python
   #img = cv2.imread('../data/raw/IMG20161205130156-16.jpg') # 这个图片现在不能检测
   ```
   
2. **参数未系统化**：各参数集还是手工调的，不是数据驱动

3. **棋子颜色判定**：仍然依赖固定阈值（强度 < 100 = 黑，> 180 = 白），对复杂光线不稳定

4. **无棋理校验层**：没有回合顺序/合法性检验，纯视觉结果

5. **无时序处理**：每张图独立处理，不利用视频帧间关联

6. **无自动化测试框架**：无法批量评估准确率

---

## 四、如何运行

```bash
cd src/

# 安装依赖（缺 scipy）
pip install opencv-python numpy scipy

# 运行最新版本
python3 Tran-BoardDect-wDrag-5B-6.py
```

测试图片路径（在脚本第337行修改）：
```python
img = cv2.imread('../data/raw/IMG20160706171004.jpg')  # 默认
# 其他可用图片（注释掉的选项）：
# '../data/raw/IMG20171015161921.jpg'
# '../data/raw/WechatIMG123.jpg'
# '../data/raw/OGS3.jpeg'
```

---

## 五、从这里出发，到达99%+的路线图

### 马上能改进（低垂果实）
1. **CLAHE 预处理** → 改善复杂光线下的阈值稳定性  
2. **自适应颜色阈值** → 用棋盘角点空白格校准黑白子阈值  
3. **添加批量测试脚本** → 一键测试所有35张图，输出准确率

### 中期改进（1-2月）
4. **接入 Gomrade 数据集** → 用真实标注数据定量评估（见 gomrade_dataset_zh.md）  
5. **提取 patch 分类数据集** → 用 `train_classifier.py` 训练 CNN  
6. **棋理逻辑校验层** → 接入 `live_broadcast_architecture_zh.md` 的设计

### 长期目标（3-6月）
7. **移动端移植** → OpenCV Mobile + CoreML  
8. **实时视频流处理** → 背景差分触发 + 帧间状态机
