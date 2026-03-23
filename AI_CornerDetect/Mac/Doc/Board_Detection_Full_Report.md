# 棋盘角点检测 (Hybrid AI-CV Corner Detection) 技术分析与 Debug 总结

这是一篇关于解决 **WeiqiBoardDetect** 项目中棋盘四个角点精确检测与拓扑验证的全方位技术文档。本项目结合了深度学习（CNN）与传统计算机视觉（OpenCV + Harris）的各自优势，攻克了透视畸变导致角点定位失效的核心问题。

---

## 核心算法点 (Algorithm Innovations)

### 1. CNN 驱动的中心外推寻边 (CNN-Driven Edge Probing)
**现状**：传统边缘检测在复杂背景（木纹、杂物）下极易产生误导干扰。
**创新**：我们开发了基于棋盘中心点向四个象限外推的 **主动式寻边算法**。利用一个训练好的 4 分类 CNN 模型（Corner, Inner, Edge, Outer），通过采样 Patch 的类别，自动寻找棋盘与背景的分界线。

**技术细节**：从中心出发沿主角度移动。当 CNN 返回 `Outer` 时，通过 **二分法 (Binary Search)** 精确锁定边缘点（`Edge` 类）。

<div align="center">
  <img src="./Image/EdgeDetect.png" width="400">
  <br>
  <i>图 1: CNN 驱动的中心外推寻边轨迹与二分法锁定边缘</i>
</div>

**核心代码实现：**
```python
# 沿主方向步进探测，直到遇到非 Inner 标签
for step in range(max_steps):
    label, conf = self.classify_patch(img, (xi, yi))
    if label == 'Outer' and last_inner_pt:
        # 发现转折，启动二分查找精确边缘
        edge_pt = self._binary_search(img, last_inner_pt, (xi, yi))
        return edge_pt, 'Edge'
```

### 2. 两阶段 CNN 角点搜索与抗震荡机制 (Two-Phase Corner Search)
**问题**：早期逻辑在发现第一个 `Edge`（边缘）时就立即开始单轴探测。由于透视畸变，第一个边缘往往太“浅”，Patch 内缺乏足够的棋盘信息。这导致 AI 容易陷入 X 和 Y 方向都被 `Outer` 阻挡的死循环，并在回缩后反复回到同一个被阻挡的位置（Oscillation）。

**创新**：我们设计并实现了 **两阶段搜索 (Two-Phase Search)** 策略：

- **阶段 1：持续深度回缩 (Deep Retreat)**：从 OpenCV 预测的粗略位置（通常是 `Outer`）开始，持续向棋盘中心移动。关键在于：**将 `Edge` 视同 `Outer` 处理** —— 除非碰到 `Inner`，否则绝不停步。最终锚定在碰到 `Inner` 前的最后一个 `Edge` 位置。
- **阶段 2：单轴锁定滑动 (Edge Sliding)**：从深度锚点出发，沿棋盘边缘寻找真正的 `Corner`。
  - **轴锁定 (Axis Locking)**：当从某个方向回退后，锁定该轴，强制 AI 只能沿另一个轴滑动，彻底杜绝了对角线方向的偏移和死循环。
  - **双 Outer 强制跳跃 (Force-Forward)**：当 X 和 Y 两个方向探测都是 `Outer` 时，不再触发 Block。而是**强制向前跳跃一步**，交给主循环的 `Outer` 处理逻辑（向内推回）来自动重新定向。

**结果**：确保了 CNN 始终能从具备丰富棋盘信息的“深度边缘”出发，绕过复杂的背景干扰，最终精准定位角点。

<div align="center">
  <img src="./Image/4CornerPatchTrace.png" width="500">
  <br>
  <i>图 2: CNN 驱动的两阶段角点定位轨迹</i>
</div>

**核心代码实现：**
```python
# ===== 阶段 1：穿过 Outer 和 Edge 直到碰见 Inner =====
for k in range(30):
    label, conf = self.classify_patch(img, (int(cx), int(cy)))
    if label == 'Edge':
        last_edge_pos = (cx, cy)  # 记录，但不停止后退
    if label == 'Inner':
        cx, cy = last_edge_pos   # 锚定在最深的 Edge 处
        break
    # 无论是 Outer 还是 Edge：继续向中心后退
    dx, dy = center_x - cx, center_y - cy
    dist = (dx**2 + dy**2)**0.5
    cx += (dx / dist) * step; cy += (dy / dist) * step

# ===== 阶段 2：沿边缘滑动寻找 Corner =====
if label == 'Edge':
    lx_label, _ = self.classify_patch(img, (int(cx + v_x[0]), int(cy)))
    ly_label, _ = self.classify_patch(img, (int(cx), int(cy + v_y[1])))
    can_move_x = lx_label in ['Inner', 'Edge']
    can_move_y = ly_label in ['Inner', 'Edge']
    if locked_axis == 'X': can_move_x = False
    elif locked_axis == 'Y': can_move_y = False
    # 强制跳跃：当双向都是 Outer 时不触发 Block，直接跳过去
    if not can_move_x and not can_move_y:
        next_cx, next_cy = cx + v_x[0], cy + v_y[1]
```

### 3. 基于 CNN 定位的混合精调 (Hybrid Finetuning)
**创新**：一旦 CNN 确定了 Corner Patch，立刻切换到 **OpenCV Engine**：
- 利用 ROI 内的 **HoughLinesP** 锁定物理解交点。
- 结合全局主角度约束，排除了木纹干扰线。

**核心代码实现：**
```python
# 筛选最贴合全局夹角的局部线条
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=20)
for seg in lines:
    rt = segment_to_rho_theta(...)
    if abs(circular_angle_diff(rt[1], global_angle)) < np.radians(30):
        candidates.append(rt)
# 定位 L 形状的“第二条线”（避开最外侧的木头边框）
best_h = sorted(h_lines, key=lambda x: x[1], reverse=take_max)[1]
```

### 4. Harris 像素簇离散化建模 (Harris Centroid Clustering)
**问题**：真实 Harris 响应往往是数百个离散像素点，直接计算拓扑关系会导致计算爆炸。
**创新**：使用 `connectedComponentsWithStats` 对像素级响应进行 **重心化处理 (Centroid Extraction)**。
**结果**：将几百个像素杂音还原为离散的几何重心（蓝点候选），物理还原了棋盘网格的几何拓扑。

**核心代码实现：**
```python
# 将 Harris 响应像素聚合成离散点
ret, thresh_img = cv2.threshold(dst, thresh, 255, cv2.THRESH_BINARY)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img)
all_harris_global = [(int(c[0]) + x1, int(c[1]) + y1) for c in centroids[1:]]
```

<div align="center">
  <img src="./Image/FarTR-Return.png" width="500">
  <br>
  <i>图 3: TR 角点调试轨迹：展示了 Patch 成功回缩并锚定的路线</i>
</div>

### 5. 局部间距估算与透视补偿 (Local Gap & Perspective Compensation)
**问题**：全局平均间距（如 25px）无法适配透视拉伸严重的边角（如 BR 角实际间距达到 80px）。
**创新**：新增 `_estimate_local_gap_from_harris` 函数。动态分析当前 ROI 内 Harris 点之间的距离，推导 **局部格子间距** 进行验证。
**结果**：解决了几何校验在畸变区域“卡死”的根本原因。

**核心代码实现：**
```python
# 分析 Harris 邻居距离，推导真实局部 Gap
dists = sorted([np.linalg.norm(p1 - p2) for p1, p2 in pairs])
n_short = max(1, len(dists) // 2)
local_gap = float(np.median(dists[:n_short]))
```

### 6. 步进式 ROI 扩张与严格否决 (Expanding ROI & Strict Veto)
**创新**：
- **ROI 步进扩张**：如果拓扑验证失败，ROI 以 60px 为步长自动向外扩张（100 -> 160 -> 220 -> 280px）。
- **一票否决制**：在 `forbidden_dirs`（棋盘外侧）方向探测到任何点即判定无效，确保选出的点绝对是“边缘角点”。

**核心代码实现：**
```python
# 循环尝试：逐步扩大搜索半径（100 -> 160 -> 220px）
for attempt in range(MAX_EXPAND + 1):
    roi_r = base_roi_r + attempt * 60
    candidates = [p for p in all_harris_global if dist(p, hough_pt) < roi_r]
    # 严格拓扑校验
    if neighbor_in_forbidden_dir: return False # 一票否决
```

<div align="center">
  <img src="./Image/CornFinalProcess.png" width="400">
  <br>
  <i>图 4: 最终角点定位OpenCV + Harris 推理过程分步演示</i>
</div>

### 最终得到了4个角的精确定位
<div align="center">
  <img src="./Image/4cornerAccu.png" width="500">
  <br>
  <i>图 5: 最终4个角的精确定位</i>
</div>

---

## Debug 总结 (The Debug Journey)

在开发 `hybrid_scanner_v4_3.py` 的过程中，我们经历了数次关键转折：

1. **环境与运行状态的可视化**：
   - 在 V4.1 重构期间一度丢失调试信息，通过重构 **轨迹追踪图 (Trajectory Canvas)** 恢复了对 AI 寻路过程的直观把控。

2. **Corner Topo 的极限调优**：
   - 发现 TR 角定位不准是因为它刚好在初始 100px ROI 的边缘。通过 **Expanding ROI (Attempt 1)** 扩充到 160px 后，程序成功捕获并验证了角点。

3. **代码稳定性的反复碰撞**：
   - 解决了 `NameError`（变量拼写）和 `IndentationError`（嵌套注释嵌套错误）等细节问题，保证了工程化落地。

4. **TR 角震荡死循环修复 (V4.3)**：
   - **根本原因**：搜索程序在触碰到极浅的 `Edge`（由于步长 40px，实际上已经跨过了棋盘边界）时就开始单轴探测，导致 `next_pos == curr_pos` 判定为 Block，触发深度后退后再次回到原位，陷入死循环。
   - **修复 1 — 两阶段搜索**：在阶段 1 强制穿过浅层 Edge，直到碰到 Inner 确保搜索深度。
   - **修复 2 — 轴锁定机制**：后退后锁定对应轴，强制单轴滑动寻找转弯点。
   - **修复 3 — 强制跳跃**：探测到双 Outer 不判定为 Block，而是强制按原步长向前跳，交给主循环处理边界推回逻辑。

---

## 截止 3/22/2026 Status 总结 (Final Status)

**Status**: [SUCCESS] - TL, TR, BR, BL 全部精准锁定 (所有测试图均通过)。  
**Date**: 2026-03-22  
**关键结论**：  
1. 成功打造了一套适配极端视角、复杂木纹、具备自我修正能力的两阶段棋盘识别引擎。  
2. 通过“先深入、后滑行”的策略解决了个别角点（如 TR/BR）因透视拉伸导致的寻找失败问题。  
3. 双 Outer 强制跳跃逻辑极大提高了角点附近的搜索鲁棒性，避免了虚假的“死路”判定。  
4. 下一步计划：针对超大规模不同光照、背景环境的图片进行压力测试。

## 流程图 (Detect Process)
<div align="center">
  <img src="./Image/Architect-cornerDetectV1-light.png" >
  <br>
  <i>图 6: 盘角检测流程图</i>
</div>
