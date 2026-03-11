"""
围棋盘平行线提取器 v2
====================
用"角度一致性"代替消失点约束：
- 棋盘两族线在图像内看起来大致平行（消失点往往在图外很远）
- 主约束：角度偏差 < angle_tol 度
- 辅助约束：线段本身要足够长（短碎线大概率是噪声）
- 两族之间角度差要接近 90°（棋盘格是正交的）

用法:
    python goboard_parallel_lines_v2.py --img <图片路径>
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def seg_angle(x1, y1, x2, y2):
    """返回线段方向角 [0, π)"""
    a = np.arctan2(y2 - y1, x2 - x1)
    if a < 0:
        a += np.pi
    return a

def seg_len(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)

def angle_diff(a, b):
    """两角度之差，结果在 [0, π/2]"""
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Canny + HoughLinesP
# ─────────────────────────────────────────────────────────────────────────────

def extract_segments(img_gray, canny_lo=40, canny_hi=100,
                     hough_thresh=30, min_len=20, max_gap=10):
    edges = cv2.Canny(img_gray, canny_lo, canny_hi)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 360,          # 角度分辨率 0.5°
                          threshold=hough_thresh,
                          minLineLength=min_len,
                          maxLineGap=max_gap)
    if raw is None:
        return [], edges
    return [tuple(r[0]) for r in raw], edges


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 加权角度直方图 → 找两个主方向
# ─────────────────────────────────────────────────────────────────────────────

def find_two_main_directions(segs, bins=360, suppress_deg=20):
    """
    用线段长度加权的角度直方图，找出两个最主要的方向角。
    返回 (angle0, angle1) 单位：弧度
    """
    angles  = np.array([seg_angle(*s) for s in segs])
    weights = np.array([seg_len(*s)   for s in segs])

    hist, edges = np.histogram(angles, bins=bins,
                               range=(0, np.pi), weights=weights)
    # 平滑
    hist = np.convolve(hist, np.ones(7)/7, mode='same')

    peaks = []
    h = hist.copy()
    suppress_bins = int(suppress_deg / 180 * bins)

    for _ in range(2):
        idx = int(np.argmax(h))
        peak_angle = (edges[idx] + edges[idx+1]) / 2
        peaks.append(peak_angle)
        lo = max(0, idx - suppress_bins)
        hi = min(bins, idx + suppress_bins)
        h[lo:hi] = 0

    return peaks[0], peaks[1]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 按角度容差分配线段 + RANSAC 去噪
# ─────────────────────────────────────────────────────────────────────────────

def assign_to_clusters(segs, a0, a1, angle_tol_deg=8.0):
    """
    将线段分配到两个主方向族，不满足任何一族的丢弃。
    angle_tol_deg: 偏离主方向的最大容忍角度（度）
    """
    tol = np.radians(angle_tol_deg)
    c0, c1, outliers = [], [], []
    for s in segs:
        ang = seg_angle(*s)
        d0 = angle_diff(ang, a0)
        d1 = angle_diff(ang, a1)
        if d0 <= tol and d0 <= d1:
            c0.append(s)
        elif d1 <= tol:
            c1.append(s)
        else:
            outliers.append(s)
    return c0, c1, outliers


def ransac_angle_filter(segs, angle_tol_deg=5.0, n_iter=300, min_support=5):
    """
    对已分族的线段再做一次 RANSAC：
    随机取 1 条线段作为"基准角度"，统计与之角度偏差 < tol 的内点数量。
    取内点最多的角度作为精化后的主方向，返回内点列表。

    这一步主要用于剔除分族时容差边界附近混入的噪声线。
    """
    if len(segs) < 2:
        return segs, seg_angle(*segs[0]) if segs else 0.0

    tol = np.radians(angle_tol_deg)
    angles = np.array([seg_angle(*s) for s in segs])
    weights = np.array([seg_len(*s) for s in segs])

    best_mask = None
    best_score = -1
    best_center = 0.0

    rng = np.random.default_rng(0)
    for _ in range(n_iter):
        idx = rng.integers(len(segs))
        ref = angles[idx]
        diffs = np.array([angle_diff(a, ref) for a in angles])
        mask = diffs < tol
        score = weights[mask].sum()
        if score > best_score:
            best_score = score
            best_mask = mask
            best_center = ref

    # 用内点做加权均值精化主方向
    inlier_angles = angles[best_mask]
    inlier_weights = weights[best_mask]
    # 对角度做向量平均（sin/cos），避免 0/π 边界问题
    mean_sin = np.average(np.sin(2 * inlier_angles), weights=inlier_weights)
    mean_cos = np.average(np.cos(2 * inlier_angles), weights=inlier_weights)
    refined_angle = np.arctan2(mean_sin, mean_cos) / 2
    if refined_angle < 0:
        refined_angle += np.pi

    inliers = [s for s, m in zip(segs, best_mask) if m]
    return inliers, refined_angle


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 长度过滤（短碎线大多是噪声）
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_length(segs, img_w, img_h, min_ratio=0.03):
    """
    保留长度 >= min_ratio * min(img_w, img_h) 的线段
    """
    threshold = min_ratio * min(img_w, img_h)
    return [s for s in segs if seg_len(*s) >= threshold]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 可视化
# ─────────────────────────────────────────────────────────────────────────────

def make_result_canvas(img_bgr, segs_all, c0, c1, outliers,
                       a0, a1, edges):
    h, w = img_bgr.shape[:2]

    # ── 画布 A: 只显示两族平行线（背景压暗）
    dark = (img_bgr * 0.25).astype(np.uint8)
    canvas_A = dark.copy()
    colors = [(0, 220, 255), (80, 255, 80)]   # 族0 黄色, 族1 绿色
    for segs, color in zip([c0, c1], colors):
        for seg in segs:
            cv2.line(canvas_A, (seg[0], seg[1]), (seg[2], seg[3]),
                     color, 2, cv2.LINE_AA)

    # 在右上角标注两个主方向角度
    cv2.putText(canvas_A, f"Dir0: {np.degrees(a0):.1f} deg",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[0], 2)
    cv2.putText(canvas_A, f"Dir1: {np.degrees(a1):.1f} deg",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[1], 2)
    cv2.putText(canvas_A, "Parallel Lines Only",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── 画布 B: 原图上显示所有线，平行线高亮
    canvas_B = img_bgr.copy()
    for seg in outliers:
        cv2.line(canvas_B, (seg[0], seg[1]), (seg[2], seg[3]),
                 (60, 60, 60), 1)
    for segs, color in zip([c0, c1], colors):
        for seg in segs:
            cv2.line(canvas_B, (seg[0], seg[1]), (seg[2], seg[3]),
                     color, 2, cv2.LINE_AA)
    cv2.putText(canvas_B, "All Lines (parallel highlighted)",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── 画布 C: Canny
    canvas_C = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas_C, "Canny Edges",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return np.hstack([canvas_A, canvas_B, canvas_C])


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def process(img_path,
            canny_lo=40, canny_hi=100,
            hough_thresh=30, min_len=20, max_gap=10,
            angle_tol=8.0,
            min_len_ratio=0.03,
            show=True,
            save=True):

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[-] 无法读取: {img_path}"); return

    h, w = img_bgr.shape[:2]
    print(f"[*] 图片: {w}x{h}")

    # 预处理
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 1: 提取线段
    segs, edges = extract_segments(gray, canny_lo, canny_hi,
                                   hough_thresh, min_len, max_gap)
    print(f"[*] 原始线段: {len(segs)}")
    if len(segs) < 10:
        print("[!] 线段太少，尝试降低 canny_lo 或 hough_thresh"); return

    # Step 2: 找两个主方向
    a0, a1 = find_two_main_directions(segs)
    print(f"[*] 主方向: {np.degrees(a0):.1f}°, {np.degrees(a1):.1f}°  "
          f"(夹角 {np.degrees(angle_diff(a0,a1)):.1f}°)")

    # Step 3: 按角度容差分族
    c0_raw, c1_raw, outliers_raw = assign_to_clusters(segs, a0, a1, angle_tol)
    print(f"[*] 初步分族: 族0={len(c0_raw)}, 族1={len(c1_raw)}, 丢弃={len(outliers_raw)}")

    # Step 4: 长度过滤
    c0_len = filter_by_length(c0_raw, w, h, min_len_ratio)
    c1_len = filter_by_length(c1_raw, w, h, min_len_ratio)
    print(f"[*] 长度过滤后: 族0={len(c0_len)}, 族1={len(c1_len)}")

    # Step 5: RANSAC 角度精化（二次过滤角度离群点）
    c0_final, a0_refined = ransac_angle_filter(c0_len, angle_tol_deg=angle_tol * 0.7)
    c1_final, a1_refined = ransac_angle_filter(c1_len, angle_tol_deg=angle_tol * 0.7)
    print(f"[*] RANSAC后: 族0={len(c0_final)} ({np.degrees(a0_refined):.1f}°), "
          f"族1={len(c1_final)} ({np.degrees(a1_refined):.1f}°)")

    # 计算最终 outliers（用于可视化）
    kept = set(map(tuple, c0_final)) | set(map(tuple, c1_final))
    outliers_final = [s for s in segs if tuple(s) not in kept]

    # 可视化
    combined = make_result_canvas(img_bgr,
                                  segs, c0_final, c1_final, outliers_final,
                                  a0_refined, a1_refined, edges)

    if save:
        out_path = str(Path(img_path).with_suffix('')) + "_v2_parallel.jpg"
        cv2.imwrite(out_path, combined)
        print(f"[+] 已保存: {out_path}")

    if show:
        # 限制显示宽度
        disp_w = min(combined.shape[1], 1800)
        scale  = disp_w / combined.shape[1]
        disp   = cv2.resize(combined,
                            (disp_w, int(combined.shape[0] * scale)))
        cv2.imshow("Go Board Parallel Lines v2", disp)
        print("    按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return c0_final, c1_final, a0_refined, a1_refined


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",          type=str,   required=True)
    parser.add_argument("--canny_lo",     type=int,   default=40)
    parser.add_argument("--canny_hi",     type=int,   default=100)
    parser.add_argument("--hough_thresh", type=int,   default=30)
    parser.add_argument("--min_len",      type=int,   default=20)
    parser.add_argument("--max_gap",      type=int,   default=10)
    parser.add_argument("--angle_tol",    type=float, default=8.0,
                        help="线段角度偏离主方向的最大容忍度（度），默认 8°")
    parser.add_argument("--min_len_ratio",type=float, default=0.03,
                        help="线段最短长度（相对图片短边的比例），默认 0.03")
    parser.add_argument("--no_show",      action="store_true")
    a = parser.parse_args()

    process(a.img, a.canny_lo, a.canny_hi, a.hough_thresh,
            a.min_len, a.max_gap, a.angle_tol, a.min_len_ratio,
            show=not a.no_show)