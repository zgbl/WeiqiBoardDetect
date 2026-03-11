"""
围棋盘角点检测 v1 — Harris + 网格 RANSAC
==========================================
流程：
  1. Harris Corner 提取候选角点（已知效果不错）
  2. 对角点云做 两轴等间距 RANSAC：
       - 找一个仿射网格 (origin_x, origin_y, dx, dy, angle)
       - 使得尽可能多的角点落在格点附近
  3. 内点 = 真正的棋盘交叉点；外点（木纹噪声等）丢弃
  4. 从内点推算棋盘四角

用法：
    python goboard_grid_v1.py --img <图片路径>
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial import KDTree


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Harris Corner 提取
# ─────────────────────────────────────────────────────────────────────────────

def extract_harris_corners(img_bgr,
                           block_size=4,
                           ksize=3,
                           k=0.04,
                           thresh_ratio=0.15,
                           nms_radius=6):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    harris = cv2.cornerHarris(gray, block_size, ksize, k)
    harris = cv2.dilate(harris, None)  # 局部最大值

    threshold = thresh_ratio * harris.max()
    ys, xs = np.where(harris > threshold)
    scores = harris[ys, xs]

    # NMS：按得分排序，贪心保留
    order = np.argsort(-scores)
    xs, ys, scores = xs[order], ys[order], scores[order]

    kept = []
    used = np.zeros(len(xs), dtype=bool)
    pts_arr = np.stack([xs, ys], axis=1).astype(np.float32)
    tree = KDTree(pts_arr)

    for i in range(len(xs)):
        if used[i]: continue
        kept.append((xs[i], ys[i], scores[i]))
        # 压制邻居
        neighbors = tree.query_ball_point([xs[i], ys[i]], nms_radius)
        for nb in neighbors:
            used[nb] = True

    pts = np.array([(x, y) for x, y, _ in kept], dtype=np.float32)
    print(f"[*] Harris NMS 后角点数: {len(pts)}")
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 主方向估计（用 Hough 线段或角点对方向直方图）
# ─────────────────────────────────────────────────────────────────────────────

def estimate_grid_directions(img_gray, canny_lo=30, canny_hi=80):
    """
    从 Hough 线段估计棋盘两个主方向角。
    返回 (angle0, angle1) 单位弧度，[0, π)
    """
    edges = cv2.Canny(img_gray, canny_lo, canny_hi)
    raw = cv2.HoughLinesP(edges, 1, np.pi/360,
                          threshold=30, minLineLength=20, maxLineGap=12)
    if raw is None or len(raw) < 4:
        return 0.0, np.pi/2

    segs = [r[0] for r in raw]
    angles  = []
    weights = []
    for x1,y1,x2,y2 in segs:
        a = np.arctan2(y2-y1, x2-x1)
        if a < 0: a += np.pi
        angles.append(a)
        weights.append(np.hypot(x2-x1, y2-y1))

    angles  = np.array(angles)
    weights = np.array(weights)

    bins = 360
    hist, edges_h = np.histogram(angles, bins=bins,
                                 range=(0, np.pi), weights=weights)
    hist = np.convolve(hist, np.ones(9)/9, mode='same')

    peaks = []
    h = hist.copy()
    sup = int(20 / 180 * bins)
    for _ in range(2):
        idx = int(np.argmax(h))
        peaks.append((edges_h[idx] + edges_h[idx+1]) / 2)
        h[max(0,idx-sup):min(bins,idx+sup)] = 0

    return peaks[0], peaks[1]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 核心 — 透视网格 RANSAC
# ─────────────────────────────────────────────────────────────────────────────

def project_points(pts, angle):
    """将点集投影到 angle 方向（平行轴）和 angle+90° 方向（法线轴）"""
    ca, sa = np.cos(angle), np.sin(angle)
    # 平行方向
    proj_parallel = pts[:,0]*ca + pts[:,1]*sa
    # 法线方向
    proj_normal   = pts[:,0]*(-sa) + pts[:,1]*ca
    return proj_parallel, proj_normal


def find_equidistant_inliers_1d(positions, n_iter=400,
                                 tol_ratio=0.18, min_count=5):
    """
    在 1D 位置数组中，用 RANSAC 找最大等间距子集。
    返回 (mask, spacing, origin)
    """
    n = len(positions)
    if n < min_count:
        return np.ones(n, dtype=bool), 0, 0

    best_mask  = np.zeros(n, dtype=bool)
    best_count = 0
    best_d     = 0
    best_orig  = 0

    rng = np.random.default_rng(7)
    pos = np.array(positions)

    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        gap = abs(pos[i] - pos[j])
        if gap < 2:
            continue

        for k in range(1, 10):          # gap 可能是 k 个间距
            d = gap / k
            if d < 3:
                continue
            tol = tol_ratio * d
            orig = pos[i] % d
            shifted = (pos - orig) % d
            dist = np.minimum(shifted, d - shifted)
            mask = dist < tol
            count = mask.sum()
            if count > best_count:
                best_count = count
                best_mask  = mask.copy()
                best_d     = d
                best_orig  = orig

    return best_mask, best_d, best_orig


def grid_ransac(pts, a0, a1,
                n_iter_global=300,
                tol_ratio=0.18,
                min_grid_pts=12):
    """
    对角点集做两轴等间距 RANSAC：
      - 在方向 a0 的法线上找等间距（对应"横线族"间距）
      - 在方向 a1 的法线上找等间距（对应"竖线族"间距）
      - 两轴同时满足的点 = 棋盘交叉点

    透视下两个方向的间距不是全局固定的，但在整图范围内变化有限，
    用 tol_ratio=0.18 可以容纳轻微的透视拉伸。
    """
    best_mask   = np.zeros(len(pts), dtype=bool)
    best_count  = 0
    best_params = None

    # 尝试两种方向组合（哪个是"行"哪个是"列"不确定）
    for row_angle, col_angle in [(a0, a1), (a1, a0)]:
        # 行方向：法线投影给出"行号"
        _, row_proj = project_points(pts, row_angle)
        # 列方向：法线投影给出"列号"
        _, col_proj = project_points(pts, col_angle)

        row_mask, row_d, row_orig = find_equidistant_inliers_1d(
            row_proj, n_iter=n_iter_global, tol_ratio=tol_ratio)
        col_mask, col_d, col_orig = find_equidistant_inliers_1d(
            col_proj, n_iter=n_iter_global, tol_ratio=tol_ratio)

        combined_mask = row_mask & col_mask
        count = combined_mask.sum()

        if count > best_count:
            best_count  = count
            best_mask   = combined_mask
            best_params = {
                'row_angle': row_angle, 'col_angle': col_angle,
                'row_d': row_d, 'row_orig': row_orig,
                'col_d': col_d, 'col_orig': col_orig,
            }

    print(f"[*] 网格 RANSAC 内点: {best_count}/{len(pts)}")
    return best_mask, best_params


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 从内点推算棋盘四角
# ─────────────────────────────────────────────────────────────────────────────

def assign_grid_indices(pts_inlier, params, tol_ratio=0.25):
    """
    给每个内点分配 (row_idx, col_idx) 网格坐标。
    """
    row_angle = params['row_angle']
    col_angle = params['col_angle']
    row_d     = params['row_d']
    row_orig  = params['row_orig']
    col_d     = params['col_d']
    col_orig  = params['col_orig']

    _, row_proj = project_points(pts_inlier, row_angle)
    _, col_proj = project_points(pts_inlier, col_angle)

    row_idx = np.round((row_proj - row_orig) / row_d).astype(int)
    col_idx = np.round((col_proj - col_orig) / col_d).astype(int)

    # 归一化为从 0 开始
    row_idx -= row_idx.min()
    col_idx -= col_idx.min()

    return row_idx, col_idx


def find_board_corners_from_grid(pts_inlier, row_idx, col_idx):
    """
    从网格坐标找四个角：
    (min_row, min_col), (min_row, max_col),
    (max_row, min_col), (max_row, max_col)
    对每个角找最近的内点作为代表。
    """
    r_min, r_max = row_idx.min(), row_idx.max()
    c_min, c_max = col_idx.min(), col_idx.max()

    corners_rc = [
        (r_min, c_min), (r_min, c_max),
        (r_max, c_min), (r_max, c_max)
    ]
    corner_pts = []
    for (tr, tc) in corners_rc:
        # 找 row_idx==tr 且 col_idx==tc 的点
        mask = (row_idx == tr) & (col_idx == tc)
        if mask.sum() > 0:
            corner_pts.append(pts_inlier[mask][0])
        else:
            # 找最近的
            dist = (row_idx - tr)**2 + (col_idx - tc)**2
            corner_pts.append(pts_inlier[np.argmin(dist)])

    return np.array(corner_pts)


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def visualize(img_bgr, all_pts, inlier_mask, grid_corners, params):
    out = img_bgr.copy()

    # 所有 Harris 角点（灰色小点）
    for pt in all_pts:
        cv2.circle(out, (int(pt[0]), int(pt[1])), 3, (100,100,100), -1)

    # 外点（橙色）
    for pt, m in zip(all_pts, inlier_mask):
        if not m:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 4, (0,140,255), 1)

    # 内点（绿色）
    for pt, m in zip(all_pts, inlier_mask):
        if m:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 5, (0,255,80), -1)

    # 棋盘四角（红色大圆 + 标注）
    if grid_corners is not None and len(grid_corners) == 4:
        labels = ["TL", "TR", "BL", "BR"]
        for pt, lbl in zip(grid_corners, labels):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(out, (x,y), 14, (0,0,255), 3)
            cv2.circle(out, (x,y),  4, (0,0,255), -1)
            cv2.putText(out, lbl, (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        # 连线
        order = [0, 1, 3, 2, 0]
        for i in range(4):
            p1 = tuple(grid_corners[order[i]].astype(int))
            p2 = tuple(grid_corners[order[i+1]].astype(int))
            cv2.line(out, p1, p2, (0,0,255), 2)

    if params:
        info = (f"row_d={params['row_d']:.1f}px  "
                f"col_d={params['col_d']:.1f}px  "
                f"inliers={inlier_mask.sum()}/{len(inlier_mask)}")
        cv2.putText(out, info, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def process(img_path,
            harris_thresh=0.12,
            nms_radius=7,
            canny_lo=30, canny_hi=80,
            tol_ratio=0.18,
            show=True, save=True):

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[-] 无法读取: {img_path}"); return

    h, w = img_bgr.shape[:2]
    print(f"[*] 图片: {w}x{h}")

    # Step 1: Harris
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    all_pts = extract_harris_corners(img_bgr,
                                     thresh_ratio=harris_thresh,
                                     nms_radius=nms_radius)
    if len(all_pts) < 20:
        print("[!] Harris 角点太少，降低 harris_thresh"); return

    # Step 2: 主方向
    gray_blur = cv2.GaussianBlur(gray_eq, (3,3), 0)
    a0, a1 = estimate_grid_directions(gray_blur, canny_lo, canny_hi)
    print(f"[*] 主方向: {np.degrees(a0):.1f}°  {np.degrees(a1):.1f}°")

    # Step 3: 网格 RANSAC
    inlier_mask, params = grid_ransac(all_pts, a0, a1,
                                      tol_ratio=tol_ratio)
    inlier_pts = all_pts[inlier_mask]

    # Step 4: 分配格坐标 + 找四角
    grid_corners = None
    if params and len(inlier_pts) >= 4:
        row_idx, col_idx = assign_grid_indices(inlier_pts, params)
        grid_corners = find_board_corners_from_grid(inlier_pts, row_idx, col_idx)
        n_rows = row_idx.max() - row_idx.min() + 1
        n_cols = col_idx.max() - col_idx.min() + 1
        print(f"[*] 网格估计大小: {n_rows} 行 x {n_cols} 列")
        print(f"[*] 棋盘四角:\n{grid_corners}")

    # 可视化
    result = visualize(img_bgr, all_pts, inlier_mask, grid_corners, params)

    if save:
        out_path = str(Path(img_path).with_suffix('')) + "_grid_v1.jpg"
        cv2.imwrite(out_path, result)
        print(f"[+] 已保存: {out_path}")

    if show:
        dw = min(result.shape[1], 1400)
        sc = dw / result.shape[1]
        disp = cv2.resize(result, (dw, int(result.shape[0]*sc)))
        cv2.imshow("Go Board Grid v1", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return inlier_pts, grid_corners, params


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img",           type=str,   required=True)
    p.add_argument("--harris_thresh", type=float, default=0.12,
                   help="Harris 响应阈值比例 (0~1)，越小角点越多，默认 0.12")
    p.add_argument("--nms_radius",    type=int,   default=7,
                   help="Harris NMS 半径(px)，默认 7")
    p.add_argument("--canny_lo",      type=int,   default=30)
    p.add_argument("--canny_hi",      type=int,   default=80)
    p.add_argument("--tol_ratio",     type=float, default=0.18,
                   help="等间距容差（相对间距比例），默认 0.18")
    p.add_argument("--no_show",       action="store_true")
    a = p.parse_args()
    process(a.img, a.harris_thresh, a.nms_radius,
            a.canny_lo, a.canny_hi, a.tol_ratio,
            show=not a.no_show)