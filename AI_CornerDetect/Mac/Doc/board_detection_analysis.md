# 棋盘角点检测：各版本问题分析与 AG 版设计

## 各版本核心问题

### [opencv_engine.py](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine.py)（原版）
| 问题 | 具体表现 |
|------|----------|
| [separate_by_angle](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine_qw_V1.py#97-198) 返回 5 个值 | 返回了 `peak1_angle`, `peak2_angle`，但分组逻辑内含 **嵌套条件过滤** (`diff_to_target < 15°`)，导致"该留的线被过滤" |
| DP 是 O(N³·n) | 3D DP 表 `dp[n][N][N]`，当 N > 40 时内存和时间都炸 |
| [select_n_evenly_spaced](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine_qw_V2.py#189-267) 有预过滤 | 70% expected_gap 门槛太激进，透视下近端间距大、远端间距小，容易误删 |
| 硬编码容差 `v_tol=18, h_tol=10` | 不同照片透视程度不同，固定容差必然有一部分图片出问题 |

### [opencv_engine_qwen.py](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine_qwen.py)（Qwen 修复版）
- 基本是原版的微调，核心架构不变
- 加了 `shrink_ratio` 内缩和边界钳制，但**根因未解决**
- 如果线分组就错了，后面的内缩保护毫无意义

### [opencv_engine_qw_V1.py](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine_qw_V1.py)（V1 保守版）
- 容差放到 `v_tol=20, h_tol=15`，太宽松 → **噪声线混入**
- 加了 `perspective_factor` 检测，但阈值 < 0.6 太粗糙

### [opencv_engine_qw_V2.py](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine_qw_V2.py)（V2 轮廓辅助版）
- 引入轮廓检测 [detect_board_contour](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine_qw_V3.py#296-324)，思路可以
- 但轮廓检测要求棋盘有清晰的外边框轮廓——**不一定有**
- 融合逻辑 `blend_ratio=0.4` 太机械，不能自适应

### [opencv_engine_qw_V3.py](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine_qw_V3.py)（V3 透视矫正版）
- 先透视矫正再检测，**思路最正确**
- 但依赖 [detect_board_contour](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine_qw_V3.py#296-324) 找到 4 点四边形→ 照片里棋盘边缘被桌子/其他物体遮挡时就失败
- 内部 [_find_corners_in_warped](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine_qw_V3.py#391-434) 用的分组逻辑和原版一样，没有根本改进

---

## AG 版核心设计差异

### 1. 使用标准 HoughLines 替代 HoughLinesP

```diff
- raw_lines = cv2.HoughLinesP(edge2, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
+ lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
```

**理由**：
- `HoughLinesP` 返回的是**线段**（有起止点），棋盘线在图像边缘可能断裂 → 一条线变多段
- `HoughLines` 返回的是**全局直线参数** [(rho, theta)](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/test_mac.py#15-91)，天然就是"一条棋盘线一个(rho,theta)"
- 减少了后续需要处理的"碎片线段合并"问题

### 2. 多尺度 Canny 融合

```python
edge_strong = cv2.Canny(blur, 50, 150)
edge_weak   = cv2.Canny(blur, 30, 100)
edges = cv2.bitwise_or(edge_strong, edge_weak)
```

远端棋盘线因透视缩小，对比度低 → 单一 Canny 阈值可能漏检

### 3. 自适应角度容差

```python
if sep < 75:    base_tol = 18   # 透视非常严重
elif sep < 85:  base_tol = 12   # 轻度透视
else:           base_tol = 8    # 近似正交
```

根据两组线的实际正交分离度自动调整，不需要人工调参

### 4. O(N²·n) DP 替代 O(N³·n) DP

```diff
- dp = np.full((n, N, N), np.inf)   # 3D: 追踪前两条线
+ dp = np.full((n, N), INF)         # 2D: 只追踪最后一条
```

**关键洞察**：我们不需要同时记住"最后两条线"来惩罚步长变化。在透视视角下，间距本来就**单调变化**（近大远小），没有必要惩罚相邻步长差异。只需惩罚**偏离全局预期间距**即可。

### 5. 角点 = 首末线的 4 个交点

```python
h_first × v_first → TL
h_first × v_last  → TR
h_last  × v_first → BL
h_last  × v_last  → BR
```

不需要先算 361 个交点再从中选 4 个极值点。如果 19 条线选对了，角点就是直接算出来的。

---

## 运行方式

```bash
python /Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/test_mac_ag.py --img /Users/tuxy/Codes/AI/Data/EmptyBoard/EmptyBoard3.jpg
```

测试所有 3 张图：
```bash
for i in 1 2 3; do
  python /Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/test_mac_ag.py \
    --img /Users/tuxy/Codes/AI/Data/EmptyBoard/EmptyBoard${i}.jpg
done
```

## 文件清单

| 文件 | 说明 |
|------|------|
| [opencv_engine_ag.py](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/opencv_engine_ag.py) | AG 版引擎（完全独立重写） |
| [test_mac_ag.py](file:///Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/test_mac_ag.py) | 配套测试脚本 |
