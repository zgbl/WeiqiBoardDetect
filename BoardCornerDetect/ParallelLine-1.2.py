"""
围棋盘线提取器 v3 — 等间距约束
================================
核心思路：
  棋盘线 = 一组等间距平行线（19条，间距固定）
  木纹线 = 随机间距的平行线
  
  → 不靠角度过滤，靠"等间距性"来区分真棋盘线和木纹噪声

流程：
  1. Canny + HoughLinesP 提取线段
  2. 角度直方图找两个主方向（横向族 / 纵向族）
  3. 每族线段投影到垂直轴 → 得到每条线的"位置值"
  4. RANSAC 等间距拟合：找最大子集使得相邻间距接近某个固定值 d
  5. 只保留属于等间距网格的线段，其余全部丢弃

用法：
    python goboard_lines_v3.py --img <图片路径>
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def seg_angle(x1, y1, x2, y2):
    a = np.arctan2(y2 - y1, x2 - x1)
    return a + np.pi if a < 0 else a  # [0, π)

def seg_len(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)

def angle_diff(a, b):
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)

def seg_midpoint(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def project_onto_normal(seg, direction_angle):
    """
    将线段中点投影到 direction_angle 的法线方向上。
    direction_angle 是线段方向角，法线方向 = direction_angle + π/2。
    返回投影值（标量），相当于这条线"在垂直方向上的位置"。
    """
    mx, my = seg_midpoint(*seg)
    # 法线方向单位向量
    nx = np.cos(direction_angle + np.pi / 2)
    ny = np.sin(direction_angle + np.pi / 2)
    return mx * nx + my * ny


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 提取线段
# ─────────────────────────────────────────────────────────────────────────────

def extract_segments(img_gray, canny_lo=30, canny_hi=80,
                     hough_thresh=25, min_len=15, max_gap=12):
    edges = cv2.Canny(img_gray, canny_lo, canny_hi)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 360,
                          threshold=hough_thresh,
                          minLineLength=min_len,
                          maxLineGap=max_gap)
    segs = [tuple(r[0]) for r in raw] if raw is not None else []
    return segs, edges


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 找两个主方向（角度直方图）
# ─────────────────────────────────────────────────────────────────────────────

def find_two_main_directions(segs, bins=360, suppress_deg=25):
    angles  = np.array([seg_angle(*s) for s in segs])
    weights = np.array([seg_len(*s)   for s in segs])
    hist, edges = np.histogram(angles, bins=bins, range=(0, np.pi), weights=weights)
    hist = np.convolve(hist, np.ones(9)/9, mode='same')

    peaks = []
    h = hist.copy()
    sup = int(suppress_deg / 180 * bins)
    for _ in range(2):
        idx = int(np.argmax(h))
        peaks.append((edges[idx] + edges[idx+1]) / 2)
        h[max(0, idx-sup):min(bins, idx+sup)] = 0

    return peaks[0], peaks[1]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 按角度容差粗分族
# ─────────────────────────────────────────────────────────────────────────────

def split_into_clusters(segs, a0, a1, angle_tol_deg=12.0):
    tol = np.radians(angle_tol_deg)
    c0, c1 = [], []
    for s in segs:
        ang = seg_angle(*s)
        d0 = angle_diff(ang, a0)
        d1 = angle_diff(ang, a1)
        best_d = min(d0, d1)
        if best_d > tol:
            continue
        if d0 <= d1:
            c0.append(s)
        else:
            c1.append(s)
    return c0, c1


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 核心 — 等间距 RANSAC
# ─────────────────────────────────────────────────────────────────────────────

def equidistant_ransac(segs, direction_angle,
                       n_lines=19,
                       n_iter=600,
                       pos_tol=0.15,      # 允许偏离理想网格的比例（相对间距 d）
                       min_lines=5):
    """
    在一族线段中，找出最大的等间距子集。
    
    参数：
        segs            : 同一方向族的线段列表
        direction_angle : 这族线的平均方向角
        n_lines         : 棋盘线数量（围棋 19 条）
        pos_tol         : 允许偏离理想位置的比例，0.15 表示 ±15% 间距
        min_lines       : 最少等间距线条数才算有效
    
    返回：
        best_segs    : 等间距子集中的线段
        best_grid    : (origin, spacing) 最佳网格参数
    """
    if len(segs) < min_lines:
        return segs, None

    # 计算每条线段的"垂直位置"
    positions = np.array([project_onto_normal(s, direction_angle) for s in segs])
    seg_arr = np.array(segs)

    best_mask = np.zeros(len(segs), dtype=bool)
    best_count = 0
    best_grid = None

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        # 随机取 2 条线，用它们的间距作为候选 spacing
        i, j = rng.choice(len(segs), 2, replace=False)
        p_i, p_j = positions[i], positions[j]
        if abs(p_i - p_j) < 1.0:
            continue

        spacing = abs(p_i - p_j)
        # 候选：这个 spacing 对应几格？（1格到 n_lines-1 格都试）
        for k in range(1, min(n_lines, 8)):
            d = spacing / k           # 单格间距候选
            if d < 3:                 # 间距太小，不可能是棋盘线
                continue

            tol_abs = pos_tol * d     # 绝对容差

            # 以 p_i 为原点，用 d 建立候选网格，统计所有线段有多少落在格点上
            origin = p_i % d          # 网格原点（取模使其在 [0, d) 内）
            
            # 每条线段到最近格点的距离
            shifted = (positions - origin) % d
            dist_to_grid = np.minimum(shifted, d - shifted)
            
            mask = dist_to_grid < tol_abs
            count = mask.sum()

            if count > best_count:
                best_count = count
                best_mask = mask.copy()
                best_grid = (origin, d)

    if best_count < min_lines:
        # 退回到全部线段
        return segs, None

    best_segs = [s for s, m in zip(segs, best_mask) if m]
    return best_segs, best_grid


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 精化方向角（用内点做加权平均）
# ─────────────────────────────────────────────────────────────────────────────

def refine_direction(segs):
    if not segs:
        return 0.0
    angles  = np.array([seg_angle(*s) for s in segs])
    weights = np.array([seg_len(*s)   for s in segs])
    s = np.average(np.sin(2 * angles), weights=weights)
    c = np.average(np.cos(2 * angles), weights=weights)
    a = np.arctan2(s, c) / 2
    return a if a >= 0 else a + np.pi


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def visualize(img_bgr, segs_all, c0_final, c1_final,
              a0, a1, grid0, grid1, edges):
    h, w = img_bgr.shape[:2]
    kept = set(map(tuple, c0_final)) | set(map(tuple, c1_final))
    outliers = [s for s in segs_all if tuple(s) not in kept]

    col0 = (0, 220, 255)   # 黄
    col1 = (80, 255, 80)   # 绿

    # ── 画布 A：只显示棋盘线
    dark = (img_bgr * 0.2).astype(np.uint8)
    A = dark.copy()
    for segs, col in [(c0_final, col0), (c1_final, col1)]:
        for s in segs:
            cv2.line(A, (s[0],s[1]), (s[2],s[3]), col, 2, cv2.LINE_AA)

    def grid_info(grid, angle):
        if grid:
            return f"spacing={grid[1]:.1f}px  angle={np.degrees(angle):.1f}°"
        return f"angle={np.degrees(angle):.1f}°"

    cv2.putText(A, f"Cluster0: {grid_info(grid0, a0)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col0, 2)
    cv2.putText(A, f"Cluster1: {grid_info(grid1, a1)}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col1, 2)
    cv2.putText(A, f"Lines: {len(c0_final)} + {len(c1_final)}", (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
    cv2.putText(A, "Equidistant Board Lines Only", (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # ── 画布 B：原图叠加
    B = img_bgr.copy()
    for s in outliers:
        cv2.line(B, (s[0],s[1]), (s[2],s[3]), (50,50,50), 1)
    for segs, col in [(c0_final, col0), (c1_final, col1)]:
        for s in segs:
            cv2.line(B, (s[0],s[1]), (s[2],s[3]), col, 2, cv2.LINE_AA)
    cv2.putText(B, "Board Lines (equidistant)", (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # ── 画布 C：Canny
    C = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(C, "Canny", (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    return np.hstack([A, B, C])


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def process(img_path,
            canny_lo=30, canny_hi=80,
            hough_thresh=25, min_len=15, max_gap=12,
            angle_tol=12.0,
            pos_tol=0.15,
            show=True, save=True):

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[-] 无法读取: {img_path}"); return

    h, w = img_bgr.shape[:2]
    print(f"[*] 图片: {w}x{h}")

    # 预处理
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 1
    segs, edges = extract_segments(gray, canny_lo, canny_hi,
                                   hough_thresh, min_len, max_gap)
    print(f"[*] 原始线段: {len(segs)}")
    if len(segs) < 10:
        print("[!] 线段太少"); return

    # Step 2
    a0, a1 = find_two_main_directions(segs)
    print(f"[*] 主方向: {np.degrees(a0):.1f}°  {np.degrees(a1):.1f}°")

    # Step 3: 粗分族（角度容差放宽到 12°，先多收集一些）
    c0_raw, c1_raw = split_into_clusters(segs, a0, a1, angle_tol)
    print(f"[*] 粗分族: 族0={len(c0_raw)}, 族1={len(c1_raw)}")

    # Step 4: 精化方向（用粗分族结果）
    a0 = refine_direction(c0_raw) if c0_raw else a0
    a1 = refine_direction(c1_raw) if c1_raw else a1

    # Step 5: 等间距 RANSAC（关键步骤）
    print(f"[*] 等间距 RANSAC...")
    c0_final, grid0 = equidistant_ransac(c0_raw, a0, pos_tol=pos_tol)
    c1_final, grid1 = equidistant_ransac(c1_raw, a1, pos_tol=pos_tol)

    print(f"[*] 最终: 族0={len(c0_final)} 条  grid={grid0}")
    print(f"[*]       族1={len(c1_final)} 条  grid={grid1}")

    # 可视化
    combined = visualize(img_bgr, segs, c0_final, c1_final,
                         a0, a1, grid0, grid1, edges)

    if save:
        out = str(Path(img_path).with_suffix('')) + "_v3_equidist.jpg"
        cv2.imwrite(out, combined)
        print(f"[+] 已保存: {out}")

    if show:
        dw = min(combined.shape[1], 1800)
        sc = dw / combined.shape[1]
        disp = cv2.resize(combined, (dw, int(combined.shape[0]*sc)))
        cv2.imshow("Go Board Lines v3 - Equidistant", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return c0_final, c1_final, grid0, grid1


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img",           type=str,   required=True)
    p.add_argument("--canny_lo",      type=int,   default=30)
    p.add_argument("--canny_hi",      type=int,   default=80)
    p.add_argument("--hough_thresh",  type=int,   default=25)
    p.add_argument("--min_len",       type=int,   default=15)
    p.add_argument("--max_gap",       type=int,   default=12)
    p.add_argument("--angle_tol",     type=float, default=12.0,
                   help="角度容差（度），默认 12°")
    p.add_argument("--pos_tol",       type=float, default=0.15,
                   help="等间距容差（相对间距比例），默认 0.15")
    p.add_argument("--no_show",       action="store_true")
    a = p.parse_args()
    process(a.img, a.canny_lo, a.canny_hi, a.hough_thresh,
            a.min_len, a.max_gap, a.angle_tol, a.pos_tol,
            show=not a.no_show)