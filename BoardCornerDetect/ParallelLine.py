"""
围棋盘平行线提取器
================
利用消失点约束，从 Canny 图中提取棋盘的两族平行线（在透视下汇聚到消失点）。

用法:
    python goboard_parallel_lines.py --img <图片路径>
    python goboard_parallel_lines.py --img <图片路径> --canny_lo 30 --canny_hi 80
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def line_to_homogeneous(x1, y1, x2, y2):
    """两点 → 齐次直线 (a, b, c)，满足 ax+by+c=0"""
    p1 = np.array([x1, y1, 1.0])
    p2 = np.array([x2, y2, 1.0])
    return np.cross(p1, p2)


def line_intersection(l1, l2):
    """两条齐次直线的交点（齐次坐标）"""
    pt = np.cross(l1, l2)
    if abs(pt[2]) < 1e-10:
        return None  # 平行线，交点在无穷远
    return pt[:2] / pt[2]


def angle_of_segment(x1, y1, x2, y2):
    """线段方向角 [0, π)"""
    angle = np.arctan2(y2 - y1, x2 - x1)
    if angle < 0:
        angle += np.pi
    return angle


def segment_length(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)


def point_to_line_dist(pt, line_hom):
    """点到齐次直线的距离"""
    a, b, c = line_hom
    x, y = pt
    denom = np.sqrt(a*a + b*b)
    if denom < 1e-10:
        return 1e9
    return abs(a*x + b*y + c) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Canny + HoughLinesP
# ─────────────────────────────────────────────────────────────────────────────

def extract_raw_lines(img_gray, canny_lo=40, canny_hi=100,
                      hough_thresh=40, min_len=30, max_gap=8):
    edges = cv2.Canny(img_gray, canny_lo, canny_hi)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180,
                          threshold=hough_thresh,
                          minLineLength=min_len,
                          maxLineGap=max_gap)
    if raw is None:
        return [], edges
    segs = [tuple(r[0]) for r in raw]
    return segs, edges


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 角度聚类 → 找两个主方向族
# ─────────────────────────────────────────────────────────────────────────────

def cluster_by_angle(segs, n_clusters=2, angle_bins=180):
    """
    用角度直方图 + 峰值寻找两个主方向，然后将每条线段分配到最近的方向族。
    返回 (cluster0_segs, cluster1_segs, angle0, angle1)
    """
    angles = np.array([angle_of_segment(*s) for s in segs])
    weights = np.array([segment_length(*s) for s in segs])

    # 加权直方图（用线段长度作为权重，长线更可信）
    hist, bin_edges = np.histogram(angles, bins=angle_bins,
                                   range=(0, np.pi), weights=weights)
    # 平滑直方图
    hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')

    # 找前两个峰值（间距 > 15°）
    peaks = []
    visited = set()
    for _ in range(n_clusters):
        idx = int(np.argmax(hist_smooth))
        # 压制已找到峰值附近 ±15° 的区域
        suppress_r = int(15 / 180 * angle_bins)
        for j in range(max(0, idx - suppress_r), min(angle_bins, idx + suppress_r)):
            visited.add(j)
            hist_smooth[j] = 0
        peak_angle = (bin_edges[idx] + bin_edges[idx+1]) / 2
        peaks.append(peak_angle)

    if len(peaks) < 2:
        return segs, [], peaks[0] if peaks else 0, 0

    a0, a1 = peaks[0], peaks[1]

    # 分配每条线段到最近的主方向
    cluster0, cluster1 = [], []
    for seg, ang in zip(segs, angles):
        d0 = min(abs(ang - a0), np.pi - abs(ang - a0))
        d1 = min(abs(ang - a1), np.pi - abs(ang - a1))
        if d0 < d1:
            cluster0.append(seg)
        else:
            cluster1.append(seg)

    return cluster0, cluster1, a0, a1


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 消失点 RANSAC — 过滤每族中不共享消失点的噪声线
# ─────────────────────────────────────────────────────────────────────────────

def estimate_vanishing_point_ransac(segs, n_iter=500, inlier_dist=12.0):
    """
    用 RANSAC 估计一族线段的消失点，返回 (vp, inlier_mask)。
    
    原理：
    - 随机取 2 条线段，计算它们延伸直线的交点作为候选消失点
    - 统计所有线段到该候选点的"方向一致性"得票
    - 选得票最多的候选消失点
    
    判断线段是否指向消失点：
    - 将线段延伸为无限直线（齐次形式）
    - 计算消失点到该直线的距离，小于阈值则为内点
    """
    if len(segs) < 2:
        return None, [True] * len(segs)

    lines_hom = [line_to_homogeneous(*s) for s in segs]
    best_vp = None
    best_count = 0
    best_mask = [False] * len(segs)

    n = len(segs)
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        vp = line_intersection(lines_hom[i], lines_hom[j])
        if vp is None:
            continue
        # 排除消失点距图像太近（通常 VP 在图像外或远处）
        # 不做这个限制，允许各种透视

        mask = []
        count = 0
        for lh in lines_hom:
            dist = point_to_line_dist(vp, lh)
            inlier = dist < inlier_dist
            mask.append(inlier)
            if inlier:
                count += 1

        if count > best_count:
            best_count = count
            best_vp = vp
            best_mask = mask

    return best_vp, best_mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 精化 — 用内点重新估计消失点（最小二乘）
# ─────────────────────────────────────────────────────────────────────────────

def refine_vanishing_point(segs, mask):
    """
    用内点线段做最小二乘消失点估计：
    最小化 sum_i (a_i*x + b_i*y + c_i)^2 / (a_i^2+b_i^2)
    等价于求矩阵 A^T W A 的最小特征向量（齐次系统）。
    """
    inlier_segs = [s for s, m in zip(segs, mask) if m]
    if len(inlier_segs) < 2:
        return None

    A = []
    W = []
    for seg in inlier_segs:
        l = line_to_homogeneous(*seg)
        norm = np.sqrt(l[0]**2 + l[1]**2)
        if norm < 1e-10: continue
        A.append(l / norm)
        W.append(segment_length(*seg))  # 用线段长度加权

    A = np.array(A[:, :2] if False else A)  # (N, 3)
    W = np.diag(W)

    # 最小化 || A @ vp_hom ||^2  subject to ||vp_hom|| = 1
    M = A.T @ W @ A
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    vp_hom = eigenvectors[:, 0]  # 最小特征值对应的特征向量

    if abs(vp_hom[2]) < 1e-10:
        return None
    vp = vp_hom[:2] / vp_hom[2]
    return vp


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 可视化
# ─────────────────────────────────────────────────────────────────────────────

def draw_lines_on_image(img, segs, color, thickness=1):
    out = img.copy()
    for x1, y1, x2, y2 in segs:
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def draw_vanishing_point(img, vp, color, radius=15):
    if vp is None:
        return img
    out = img.copy()
    x, y = int(vp[0]), int(vp[1])
    cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, radius * 2, 2)
    cv2.circle(out, (x, y), radius, color, 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def process(img_path, canny_lo=40, canny_hi=100,
            hough_thresh=35, min_len=25, max_gap=10,
            ransac_dist=10.0, show=True):

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[-] 无法读取图片: {img_path}")
        return
    
    cv2.imshow("img_bgr-original", img_bgr)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    h, w = img_bgr.shape[:2]
    print(f"[*] 图片尺寸: {w}x{h}")

    # 预处理
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE 增强对比度（对侧光、阴影条件下的棋盘格线更友好）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)
    # 轻微去噪
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Step 1: 提取原始线段
    segs, edges = extract_raw_lines(img_gray, canny_lo, canny_hi,
                                     hough_thresh, min_len, max_gap)
    print(f"[*] 原始 HoughLinesP 线段数: {len(segs)}")

    if len(segs) < 10:
        print("[!] 线段太少，尝试降低 --canny_lo 或 --hough_thresh")
        return

    # Step 2: 角度聚类
    cluster0, cluster1, a0, a1 = cluster_by_angle(segs, n_clusters=2)
    print(f"[*] 角度族 0: {np.degrees(a0):.1f}°，{len(cluster0)} 条线")
    print(f"[*] 角度族 1: {np.degrees(a1):.1f}°，{len(cluster1)} 条线")

    # Step 3 & 4: 消失点 RANSAC + 精化
    results = []
    colors = [(0, 200, 255), (0, 255, 100)]  # 族0: 黄色, 族1: 绿色

    for idx, (cluster, color) in enumerate(zip([cluster0, cluster1], colors)):
        if len(cluster) < 3:
            results.append(([], None))
            continue

        vp_rough, mask = estimate_vanishing_point_ransac(
            cluster, n_iter=800, inlier_dist=ransac_dist
        )
        vp_refined = refine_vanishing_point(cluster, mask)
        vp = vp_refined if vp_refined is not None else vp_rough

        # 用精化后的 VP 重新计算内点
        if vp is not None:
            lines_hom = [line_to_homogeneous(*s) for s in cluster]
            mask = [point_to_line_dist(vp, lh) < ransac_dist for lh in lines_hom]

        inliers = [s for s, m in zip(cluster, mask) if m]
        outliers = [s for s, m in zip(cluster, mask) if not m]
        inlier_ratio = len(inliers) / max(1, len(cluster))

        print(f"[*] 族 {idx}: VP=({vp[0]:.0f},{vp[1]:.0f}) | "
              f"内点 {len(inliers)}/{len(cluster)} ({inlier_ratio:.0%})")
        results.append((inliers, vp))

    # ── 可视化 ──
    # 画布1: 只显示两族平行线（隐去所有其他线）
    canvas_final = img_bgr.copy()
    
    # 先画暗化背景（隐去非棋盘线）
    overlay = np.zeros_like(img_bgr)
    canvas_final = cv2.addWeighted(img_bgr, 0.35, overlay, 0.65, 0)

    for (inliers, vp), color in zip(results, colors):
        for seg in inliers:
            x1, y1, x2, y2 = seg
            cv2.line(canvas_final, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        if vp is not None:
            canvas_final = draw_vanishing_point(canvas_final, vp, color)

    # 画布2: 原图叠加全部线段（供参考）
    canvas_all = img_bgr.copy()
    for seg in segs:
        cv2.line(canvas_all, (seg[0], seg[1]), (seg[2], seg[3]), (80, 80, 80), 1)
    for (inliers, vp), color in zip(results, colors):
        for seg in inliers:
            cv2.line(canvas_all, (seg[0], seg[1]), (seg[2], seg[3]), color, 2, cv2.LINE_AA)

    # 画布3: Canny 图
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 标注说明
    for canvas, title in [(canvas_final, "Parallel Lines Only"),
                           (canvas_all,   "All vs Parallel"),
                           (edges_bgr,    "Canny Edges")]:
        cv2.putText(canvas, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 拼接输出
    if h > 800:
        scale = 800 / h
        def rs(im): return cv2.resize(im, (int(w*scale), int(h*scale)))
        canvas_final = rs(canvas_final)
        canvas_all   = rs(canvas_all)
        edges_bgr    = rs(edges_bgr)

    combined = np.hstack([canvas_final, canvas_all, edges_bgr])

    # 保存
    out_path = str(Path(img_path).with_suffix('')) + "_parallel_lines.jpg"
    cv2.imwrite(out_path, combined)
    print(f"[+] 结果已保存: {out_path}")

    if show:
        cv2.imshow("Go Board - Parallel Line Extraction", combined)
        print("    按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return canvas_final, results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="围棋盘平行线提取（透视不变）")
    parser.add_argument("--img",         type=str,   required=True,   help="输入图片路径")
    parser.add_argument("--canny_lo",    type=int,   default=40,      help="Canny 低阈值")
    parser.add_argument("--canny_hi",    type=int,   default=100,     help="Canny 高阈值")
    parser.add_argument("--hough_thresh",type=int,   default=35,      help="Hough 投票阈值")
    parser.add_argument("--min_len",     type=int,   default=25,      help="最短线段长度(px)")
    parser.add_argument("--max_gap",     type=int,   default=10,      help="最大线段间隔(px)")
    parser.add_argument("--ransac_dist", type=float, default=10.0,    help="RANSAC 内点阈值(px)")
    parser.add_argument("--no_show",     action="store_true",         help="不弹窗，只保存文件")
    args = parser.parse_args()

    process(
        img_path     = args.img,
        canny_lo     = args.canny_lo,
        canny_hi     = args.canny_hi,
        hough_thresh = args.hough_thresh,
        min_len      = args.min_len,
        max_gap      = args.max_gap,
        ransac_dist  = args.ransac_dist,
        show         = not args.no_show,
    )