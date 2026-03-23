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

### 2. CNN 角点修正与防震荡机制 (Corner Patch Relocation)
**问题**：早期逻辑在寻边后直接交叉，常因畸变导致落在边缘。且 AI 容易在一条直线上反复衰减（Oscillation）却无法找到转弯点。
**创新**：独创了 **多维补丁探测 (Multi-direction Probing)**。当 CNN 判定为 `Edge` 时，程序会分别在横向和纵向进行探测。
- 如果两个方向都被 `Outer` 阻挡，说明冲过头了，强制向中心 **回缩 (Backtrack)**。
**结果**：彻底解决了 AI 在边缘“迷失”的问题，确保 CNN 最终能稳稳地压在 `Corner` 类别上。

<div align="center">
  <img src="./Image/4CornerPatchTrace.png" width="500">
  <br>
  <i>图 2: 角点Patch的定位过程路线</i>
</div>

**核心代码实现：**
```python
# 在边缘状态下探测横纵两步，分析“墙角”位置
if label == 'Edge':
    lx_label, _ = self.classify_patch(img, (int(cx + v_x[0]), int(cy)))
    ly_label, _ = self.classify_patch(img, (int(cx), int(cy + v_y[1])))
    if not can_move_x and not can_move_y:
        # 优先级：如果外推被阻挡，依次回缩 X 或 Y 轴向中心移动
        if backtrack_count % 2 == 1:
            cy -= v_y[1]
        else:
            cx -= v_x[0]
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
  <i>图 3: 最终角点定位OpenCV + Harris 推理过程分步演示</i>
</div>

### 最终得到了4个角的精确定位
<div align="center">
  <img src="./Image/4cornerAccu.png" width="500">
  <br>
  <i>图 4: 最终4个角的精确定位</i>
</div>

---

## Debug 总结 (The Debug Journey)

在开发 `hybrid_scanner_v4_2.py` 的过程中，我们经历了数次关键转折：

1. **环境与运行状态的可视化**：
   - 在 V4.1 重构期间一度丢失调试信息，通过重构 **轨迹追踪图 (Trajectory Canvas)** 恢复了对 AI 寻路过程的直观把控。

2. **Corner Topo 的极限调优**：
   - 发现 TR 角定位不准是因为它刚好在初始 100px ROI 的边缘。通过 **Expanding ROI (Attempt 1)** 扩充到 160px 后，程序成功捕获并验证了角点。

3. **代码稳定性的反复碰撞**：
   - 解决了 `NameError`（变量拼写）和 `IndentationError`（嵌套注释嵌套错误）等细节问题，保证了工程化落地。

---

## 截止3/22/2026 Status 总结 (Final Status)

**Status**: [SUCCESS] - TL, TR, BR, BL ALL PINNED.  
**Date**: 2026-03-22  
**小结**：  
1. 成功打造了一套适配极端视角、复杂木纹、具备自我修正能力的棋盘识别引擎。  
2. 寻角的过程还有待优化，已经想好了新的算法，在下一版实现。  
3. 算法还有待完善，其他图片还寻角失败，需要继续解决。  

## 流程图 (Detect Process)
<div align="center">
  <img src="./Image/Architect-cornerDetectV1-light.png" >
  <br>
  <i>图 5: 盘角检测流程图</i>
</div>
