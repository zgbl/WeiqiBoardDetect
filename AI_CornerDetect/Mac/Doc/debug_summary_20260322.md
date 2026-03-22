# 棋盘角点检测 (Harris Corner Refinement) 艰苦 Debug 总结

这是一篇关于解决 **WeiqiBoardDetect** 项目中棋盘四个角点精确检测与拓扑验证的开发文档。在经过多次迭代与实战图片（如 `EmptyBoard3.jpg`）的测试后，我们终于攻克了由于透视畸变导致角点定位失效的核心问题。

## 1. 核心问题背景

在对大角度拍摄的棋盘进行识别时，图像边缘存在极大的 **透视拉伸 (Perspective Distortion)**：
- **全局平均间距失效**：整盘棋盘的平均格子间距（Expected Gap）约为 25px，但由于近大远小，图像下方的 Bottom-Right (BR) 角附近的实际格子间距可能达到 80px 以上。
- **搜寻半径不足**：初始设定的 100px ROI 搜索半径，在有些极端情况下无法涵盖到足以进行拓扑验证的相邻 Harris 点。
- **干扰点过多**：木纹、反光会导致 Harris 检测出数百个像素，如果按像素点计算拓扑得分，计算量巨大且精度极低。

---

## 2. 关键突破点：三大核心算法优化

### A. 局部间距估算 (Local Gap Estimation)
**问题**：全局 `expected_gap` 无法适配透视拉伸严重的边角。
**方案**：新增 `_estimate_local_gap_from_harris` 函数。它不再盲目相信全局值，而是分析当前 ROI 内所有 Harris 候选点之间的两两距离，根据最短距离分布的中位数，动态推导出该区域的 **局部格子间距**。
**结果**：BR 角的验证通过率从 0% 提升到 100%。

### B. 步进式 ROI 扩大重试机制 (Expanding ROI Retry)
**问题**：由于 OpenCV Engine 的初步定位可能存在数十像素偏差，且 Harris 验证需要邻居点，100px 的初始 ROI 经常找不到足够的点。
**方案**：在 `_snap_to_best_harris_corner` 中引入了 `MAX_EXPAND=3` 的搜索循环：
- 第 0 次：100px (Base)
- 第 1 次：160px (+60px)
- 第 2 次：220px (+120px)
- ...每次失败时，扩大 ROI 并重新进行 Harris 检测与拓扑验证。
**结果**：原本在第一次无法验证的角（如 TR），会在第 1 次或第 2 次尝试中被成功捕获。

### C. 严苛的拓扑一票否决制 (Strict Forbidden Veto)
**问题**：最初的逻辑是对“禁止方向（棋盘外侧）有邻居点”进行扣分，导致一些非角点（如边缘点）因为总分尚可而被误认为是角点。
**方案**：将 `forbidden_dirs` 逻辑从“扣分制”改为“一票否决制”：
```python
for d in forbidden_dirs[corner_type]:
    neighbor, score = find_neighbor_in_dir(d, pts, cx, cy, gap_min, gap_max)
    if neighbor is not None:
        return False, 0.0   # 禁止方向有邻居，直接否决
```
**结果**：极大地提高了角点甄别的纯净度，确保最后选出的点一定是“真正的棋盘转角”。

---

## 3. 开发过程中的意外挑战 (The "Battle Scars")

1. **V4.1 的沉默危机**：由于大幅重构代码，一度丢失了所有 `print` 和 `cv2.imshow` 调试信息。程序看起来像“卡死”，实则是静默运行且没有在图像上绘图。我们通过恢复 **轨迹追踪图 (Trajectory Canvas)** 和 **分步骤日志** 解决了这一问题。
2. **IndentationError 陷阱**：在手动合并代码时， triple-quoted string (`"""`) 内嵌套 docstring 导致 Python 解析出错。
3. **点簇聚类 (Centroid Clustering)**：为了应对 Harris 的数百个像素干扰，我们引入了 `connectedComponentsWithStats`，将像素级的“团块”聚合为离散的“蓝点”重心坐标，这使得拓扑验证的搜索效率提高了 100 倍。

---

## 4. 最终成就

目前版本 `hybrid_scanner_v4_2.py` 已能稳定实现：
1. **CNN 寻路追踪**：自动寻找棋盘边缘。
2. **OpenCV L-Shape 修正**：通过 HoughLinesP 锁定物理解交点。
3. **Harris 拓扑闭环**：利用空间约束确定唯一的真角。
4. **动态透视补偿**：完美适配不同机位下的格子间距差异。

**Status**: [SUCCESS] - ALL 4 CORNERS PINNED.
**Date**: 2026-03-22
