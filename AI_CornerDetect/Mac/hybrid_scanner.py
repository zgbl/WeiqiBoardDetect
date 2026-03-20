"""
hybrid_scanner.py — CNN 引导搜索 + OpenCV 精确拟合 棋盘角点检测
===================================================================
全新 approach：不再先找全部线再选，而是让 CNN 引导空间搜索。

流程:
  1. OpenCV 粗检测：在整图上检测线段，找到一个"有网格"的种子区域，
     同时获取两组线的方向角度。
  2. CNN 探索：从种子点出发，沿四个方向（两组线 × 正反）逐步移动 patch，
     由 CNN 判断 Inner→Edge→Corner，找到 4 条边的位置。
  3. 角点推算：从 4 条边交叉推算 4 个角点，再用 CNN 验证。
  4. 精确拟合：用已知的棋盘区域和精确间距参数，重新筛选 OpenCV 线段，
     拟合 19×19 网格线。

依赖: cnn_engine.py 中的 CNNVerifier（verify_point 方法）

Work with .../AI_CornerDetect/Mac/test_hybrid.py
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import sys
from pathlib import Path

# 确保能导入同目录模块
sys.path.append(str(Path(__file__).parent))

# =====================================================================
# 底层几何工具（从 release/Mac0.1/opencv_engine.py 保留）
# =====================================================================

def segment_to_rho_theta(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx*dx + dy*dy)
    if length < 1e-8: return None
    nx, ny = -dy/length, dx/length
    rho = nx*x1 + ny*y1
    if rho < 0: rho, nx, ny = -rho, -nx, -ny
    theta = np.arctan2(ny, nx)
    if theta < 0: theta += 2*np.pi
    return rho, theta

def normalize_line(rho, theta):
    if rho < 0: rho, theta = -rho, theta + np.pi
    return rho, theta % (2*np.pi)

def circular_angle_diff(a1, a2, period=np.pi):
    d = abs(a1 - a2) % period
    return min(d, period - d)

def intersect_lines(rho1, theta1, rho2, theta2):
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    det = a1*b2 - a2*b1
    if abs(det) < 1e-8: return None
    return ((b2*rho1 - b1*rho2)/det, (a1*rho2 - a2*rho1)/det)


# =====================================================================
# 阶段 1: OpenCV 粗检测 — 找种子区域和方向
# =====================================================================

def find_seed_and_directions(img, debug_show=False):
    """
    在图像上检测线段，找到线段最密集的区域作为种子点，
    同时返回两组线的主方向角度和估计间距。

    返回:
        seed_point: (x, y) 种子点坐标
        dir1: (dx, dy) 第一组线的方向向量（归一化）
        dir2: (dx, dy) 第二组线的方向向量（归一化）
        angle1: float 第一组线的 theta 角（弧度）
        angle2: float 第二组线的 theta 角（弧度）
        est_gap: float 估计的棋盘线间距（像素）
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)

    raw_segs = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                                minLineLength=50, maxLineGap=10)
    if raw_segs is None or len(raw_segs) < 10:
        print("[Seed] Not enough line segments detected")
        return None, None, None, 0, 0, 0

    # 转换为 (rho, theta) + 原始端点
    lines_rt = []
    segments = []
    for seg in raw_segs:
        x1, y1, x2, y2 = seg[0]
        rt = segment_to_rho_theta(x1, y1, x2, y2)
        if rt:
            lines_rt.append(rt)
            segments.append((x1, y1, x2, y2))

    # --- 角度直方图找两个主方向 ---
    mapped = [normalize_line(r, t) for r, t in lines_rt]
    thetas_mod = np.array([t % np.pi for _, t in mapped])

    n_bins = 180; bin_sz = np.pi / n_bins
    hist = np.zeros(n_bins)
    for t in thetas_mod:
        hist[int(t / bin_sz) % n_bins] += 1

    k = 7; ker = np.ones(k)/k
    pad = np.concatenate([hist[-k:], hist, hist[:k]])
    sm = np.convolve(pad, ker, mode='same')[k:-k]

    # 水平优先
    p1_bin = -1; best = -1
    for i in range(n_bins):
        a = (i+0.5)*bin_sz
        if circular_angle_diff(a, np.pi/2) < np.radians(10) and sm[i] > best:
            best = sm[i]; p1_bin = i
    if p1_bin == -1: p1_bin = int(np.argmax(sm))
    angle1 = (p1_bin + 0.5) * bin_sz

    # 正交方向
    target = (angle1 + np.pi/2) % np.pi
    p2_bin = -1; best2 = -1
    for i in range(n_bins):
        a = (i+0.5)*bin_sz
        if circular_angle_diff(a, target) < np.radians(15) and sm[i] > best2:
            best2 = sm[i]; p2_bin = i
    if p2_bin == -1:
        sup = sm.copy()
        for i in range(n_bins):
            if circular_angle_diff((i+0.5)*bin_sz, angle1) < np.radians(30): sup[i] = 0
        p2_bin = int(np.argmax(sup))
    angle2 = (p2_bin + 0.5) * bin_sz

    print(f"[Seed] Directions: {np.degrees(angle1):.1f}° and {np.degrees(angle2):.1f}°")

    # --- 方向向量：theta 是法线角，线的走向 = 法线旋转90° ---
    # 对 theta 方向的线，沿线方向是 (-sin(theta), cos(theta))
    dir1 = np.array([-np.sin(angle1), np.cos(angle1)])  # 组1线的走向
    dir2 = np.array([-np.sin(angle2), np.cos(angle2)])  # 组2线的走向

    # --- 找线段最密集的区域作为种子点 ---
    # 用密度热图：把每条线段的中点累加到网格上
    cell = 64
    grid_h, grid_w = h // cell + 1, w // cell + 1
    density = np.zeros((grid_h, grid_w))
    for x1, y1, x2, y2 in segments:
        mx, my = (x1+x2)//2, (y1+y2)//2
        gi, gj = min(my//cell, grid_h-1), min(mx//cell, grid_w-1)
        density[gi][gj] += 1

    # 用 3x3 窗口平滑密度
    from scipy.ndimage import uniform_filter
    try:
        density_smooth = uniform_filter(density, size=3)
    except:
        density_smooth = density

    peak = np.unravel_index(np.argmax(density_smooth), density_smooth.shape)
    seed_y = int((peak[0] + 0.5) * cell)
    seed_x = int((peak[1] + 0.5) * cell)
    seed_y = max(cell, min(h - cell, seed_y))
    seed_x = max(cell, min(w - cell, seed_x))

    print(f"[Seed] Seed point: ({seed_x}, {seed_y}), density peak: {density_smooth[peak]:.0f}")

    # --- 估计间距 ---
    # 分组后，在种子点附近找同方向线的 rho 间距中位数
    g1_rhos = []
    g2_rhos = []
    for (rho, theta), (x1, y1, x2, y2) in zip(lines_rt, segments):
        mx, my = (x1+x2)/2, (y1+y2)/2
        # 只看种子点附近 300px 的线
        if abs(mx - seed_x) > 300 or abs(my - seed_y) > 300:
            continue
        mt = theta % np.pi
        if circular_angle_diff(mt, angle1 % np.pi) < np.radians(15):
            g1_rhos.append(rho)
        elif circular_angle_diff(mt, angle2 % np.pi) < np.radians(15):
            g2_rhos.append(rho)

    def median_gap(rhos):
        if len(rhos) < 3: return 60  # fallback
        rhos = sorted(rhos)
        diffs = np.diff(rhos)
        valid = diffs[diffs > 5]
        return np.median(valid) if len(valid) > 3 else 60

    gap1 = median_gap(g1_rhos)
    gap2 = median_gap(g2_rhos)
    est_gap = (gap1 + gap2) / 2

    print(f"[Seed] Estimated gaps: dir1={gap1:.1f}, dir2={gap2:.1f}, avg={est_gap:.1f}")

    if debug_show:
        vis = img.copy()
        cv2.circle(vis, (seed_x, seed_y), 20, (0, 0, 255), 3)
        # 画方向箭头
        arrow_len = 100
        for d, color in [(dir1, (0,255,0)), (dir2, (255,200,0))]:
            pt2 = (int(seed_x + d[0]*arrow_len), int(seed_y + d[1]*arrow_len))
            cv2.arrowedLine(vis, (seed_x, seed_y), pt2, color, 2, tipLength=0.3)
        cv2.imshow("[Hybrid] Seed + Directions", vis)

    return (seed_x, seed_y), dir1, dir2, angle1, angle2, est_gap


# =====================================================================
# 阶段 2 & 3: CNN 引导搜索 — 沿线方向探测边界
# =====================================================================

def search_edge_along_direction(img, cnn, start_pt, direction, step_size,
                                 max_steps=30, crop_radius=64):
    """
    从 start_pt 出发，沿 direction 逐步移动，每步让 CNN 判断。
    当 CNN 判断从 Inner 变成 Edge 或 Corner 或 Outer 时停止。

    返回:
        edge_pt: (x, y) 边界点坐标，或 None
        edge_type: 'Edge' / 'Corner' / 'Outer' / None
        path: 所有探测点和分类结果的列表
    """
    h, w = img.shape[:2]
    x, y = float(start_pt[0]), float(start_pt[1])
    dx, dy = direction[0] * step_size, direction[1] * step_size

    path = []
    last_inner_pt = None

    for step in range(max_steps):
        xi, yi = int(round(x)), int(round(y))

        # 边界检查
        if xi < crop_radius or xi >= w - crop_radius or yi < crop_radius or yi >= h - crop_radius:
            print(f"  [Search] Out of bounds at step {step}: ({xi}, {yi})")
            if last_inner_pt:
                return last_inner_pt, 'Edge', path
            return None, None, path

        # CNN 判断
        label, conf = cnn.verify_point(img, (xi, yi), crop_radius=crop_radius)
        path.append({'x': xi, 'y': yi, 'label': label, 'conf': conf, 'step': step})
        print(f"    step {step}: ({xi}, {yi}) -> {label} ({conf:.2f})")

        if label == 'Inner':
            last_inner_pt = (xi, yi)
        elif label in ('Edge', 'Corner'):
            print(f"  [Search] Found {label} at step {step}: ({xi}, {yi}), conf={conf:.2f}")
            return (xi, yi), label, path
        elif label == 'Outer':
            # 已出界，退回一步
            if last_inner_pt:
                # 在 last_inner_pt 和当前点之间做二分查找
                edge_pt = binary_search_boundary(img, cnn, last_inner_pt,
                                                  (xi, yi), crop_radius)
                return edge_pt, 'Edge', path
            else:
                # 一步都没走就出界，种子点可能本身就在边缘
                print(f"  [Search] Immediately Outer at step {step}")
                return None, 'Outer', path

        x += dx
        y += dy

    # 走完了还没找到边界
    print(f"  [Search] Max steps reached without finding edge")
    if last_inner_pt:
        return last_inner_pt, 'Edge', path
    return None, None, path


def binary_search_boundary(img, cnn, inner_pt, outer_pt, crop_radius=64, max_iter=6):
    """
    在 inner_pt (Inner) 和 outer_pt (Outer/Edge) 之间二分查找精确边界位置。
    """
    ix, iy = float(inner_pt[0]), float(inner_pt[1])
    ox, oy = float(outer_pt[0]), float(outer_pt[1])

    for _ in range(max_iter):
        mx, my = (ix + ox) / 2, (iy + oy) / 2
        mi, mj = int(round(mx)), int(round(my))

        label, conf = cnn.verify_point(img, (mi, mj), crop_radius=crop_radius)

        if label == 'Inner':
            ix, iy = mx, my
        else:
            ox, oy = mx, my

    # 返回最后的中点
    return (int(round((ix+ox)/2)), int(round((iy+oy)/2)))


def cnn_guided_search(img, cnn, seed_pt, dir1, dir2, est_gap,
                       debug_show=False):
    """
    从种子点出发，沿 4 个方向（dir1 正/反, dir2 正/反）搜索边界。
    dir1 方向的搜索找到的是「与 dir1 垂直的线」的边界，
    即 dir2 方向线组的首/末线位置。

    返回:
        edges: dict 包含 4 个方向的边界点
            'dir1_pos': 沿 dir1 正方向搜到的边界
            'dir1_neg': 沿 dir1 反方向搜到的边界
            'dir2_pos': 沿 dir2 正方向搜到的边界
            'dir2_neg': 沿 dir2 反方向搜到的边界
    """
    step = max(50, est_gap * 0.8)  # 步长略小于间距，最小50px防止间距估错时步太小

    print(f"\n[CNN Search] Seed: {seed_pt}, step_size: {step:.1f}")

    edges = {}
    vis = img.copy() if debug_show else None

    for name, direction in [('dir1_pos', dir1), ('dir1_neg', -dir1),
                            ('dir2_pos', dir2), ('dir2_neg', -dir2)]:
        print(f"\n  === Searching {name} ===")
        edge_pt, edge_type, path = search_edge_along_direction(
            img, cnn, seed_pt, direction, step, max_steps=25
        )
        edges[name] = {'point': edge_pt, 'type': edge_type, 'path': path}

        if debug_show and vis is not None:
            # 画探测路径
            colors = {'Inner': (0,255,0), 'Edge': (0,165,255),
                      'Corner': (0,0,255), 'Outer': (128,128,128)}
            for p in path:
                c = colors.get(p['label'], (255,255,255))
                cv2.circle(vis, (p['x'], p['y']), 5, c, -1)
            if edge_pt:
                cv2.circle(vis, edge_pt, 12, (0,0,255), 3)
                cv2.putText(vis, name, (edge_pt[0]+15, edge_pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    if debug_show and vis is not None:
        cv2.circle(vis, seed_pt, 10, (255,0,255), -1)
        cv2.imshow("[Hybrid] CNN Search Paths", vis)

    return edges


# =====================================================================
# 阶段 4: 从边界推算角点 + CNN 验证
# =====================================================================

def compute_corners_from_edges(edges, dir1, dir2, angle1, angle2):
    """
    从 4 个方向的边界点推算 4 个角点。

    逻辑：
    - dir1_pos 和 dir1_neg 给出了沿 dir1 方向的两个边界
    - dir2_pos 和 dir2_neg 给出了沿 dir2 方向的两个边界
    - 4 个角 = 4 条边界线的交点

    每条边界线垂直于搜索方向，过边界点。
    """
    pts = {}
    for key in ['dir1_pos', 'dir1_neg', 'dir2_pos', 'dir2_neg']:
        if edges[key]['point'] is None:
            print(f"[Corners] Missing edge: {key}")
            return []
        pts[key] = edges[key]['point']

    # 边界线的表示：
    # 沿 dir1 搜索到的点，对应一条垂直于 dir1 的线（即 dir2 方向的线）
    # 沿 dir2 搜索到的点，对应一条垂直于 dir2 的线（即 dir1 方向的线）

    # dir1_pos 边界：过 pts['dir1_pos']，方向为 dir2（即垂直于 dir1）
    # dir1_neg 边界：过 pts['dir1_neg']，方向为 dir2
    # dir2_pos 边界：过 pts['dir2_pos']，方向为 dir1
    # dir2_neg 边界：过 pts['dir2_neg']，方向为 dir1

    def line_through_point_with_direction(pt, direction):
        """返回过 pt、方向为 direction 的直线的 (rho, theta)"""
        # 法向量 = direction 旋转 90°
        nx, ny = -direction[1], direction[0]
        rho = nx * pt[0] + ny * pt[1]
        theta = np.arctan2(ny, nx)
        if rho < 0:
            rho = -rho
            theta += np.pi
        return rho, theta % (2*np.pi)

    # 4 条边界线
    line_d1_pos = line_through_point_with_direction(pts['dir1_pos'], dir2)  # 棋盘远端横线
    line_d1_neg = line_through_point_with_direction(pts['dir1_neg'], dir2)  # 棋盘近端横线
    line_d2_pos = line_through_point_with_direction(pts['dir2_pos'], dir1)  # 棋盘右侧纵线
    line_d2_neg = line_through_point_with_direction(pts['dir2_neg'], dir1)  # 棋盘左侧纵线

    # 4 个角 = 4 条边界线的 4 个交点
    corner_pairs = [
        ('TL', line_d1_neg, line_d2_neg),
        ('TR', line_d1_neg, line_d2_pos),
        ('BR', line_d1_pos, line_d2_pos),
        ('BL', line_d1_pos, line_d2_neg),
    ]

    corners = []
    for name, l1, l2 in corner_pairs:
        pt = intersect_lines(l1[0], l1[1], l2[0], l2[1])
        if pt:
            corners.append(pt)
            print(f"  [Corners] {name}: ({int(pt[0])}, {int(pt[1])})")
        else:
            print(f"  [Corners] {name}: intersection failed!")
            return []

    return corners  # [TL, TR, BR, BL]


def verify_corners_with_cnn(img, cnn, corners, crop_radius=64):
    """用 CNN 验证每个角点是否真的是 Corner"""
    results = []
    for i, (x, y) in enumerate(corners):
        labels = ['TL', 'TR', 'BR', 'BL']
        label, conf = cnn.verify_point(img, (int(x), int(y)), crop_radius=crop_radius)
        is_ok = label == 'Corner'
        tag = labels[i] if i < 4 else f'C{i}'
        print(f"  [{tag}] ({int(x)},{int(y)}) -> {label} ({conf:.2f}) {'✓' if is_ok else '✗'}")
        results.append({'point': (x, y), 'label': label, 'conf': conf, 'ok': is_ok})
    return results


# =====================================================================
# 阶段 5: 精确拟合 19×19 网格线
# =====================================================================

def refine_grid_with_known_bounds(img, corners, angle1, angle2,
                                   debug_show=False):
    """
    已知棋盘的 4 个角点后，用 OpenCV 线检测 + 严格空间约束拟合 19×19 网格。

    策略：
    1. 只保留落在棋盘区域内的线段
    2. 用角点信息精确计算期望间距
    3. 用 DP 选出 19 条等间距线
    """
    h, w = img.shape[:2]

    if len(corners) < 4:
        print("[Refine] Not enough corners")
        return [], []

    # 棋盘四角
    tl, tr, br, bl = [np.array(c) for c in corners[:4]]

    # 计算棋盘区域的 mask（用于空间约束）
    board_poly = np.array([tl, tr, br, bl], dtype=np.int32)
    board_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(board_mask, board_poly, 255)
    # 稍微扩大 mask（允许边缘线略超出）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    board_mask = cv2.dilate(board_mask, kernel, iterations=1)

    # 检测线段
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)
    # 只保留棋盘区域内的边缘
    edges_masked = cv2.bitwise_and(edges, board_mask)

    raw_segs = cv2.HoughLinesP(edges_masked, 1, np.pi/180, 80,
                                minLineLength=40, maxLineGap=10)
    if raw_segs is None:
        print("[Refine] No segments in board area")
        return [], []

    # 转 (rho, theta)
    lines_rt = []
    for seg in raw_segs:
        rt = segment_to_rho_theta(*seg[0])
        if rt: lines_rt.append(rt)

    # 分组
    group1, group2 = [], []
    for rho, theta in lines_rt:
        mt = theta % np.pi
        d1 = circular_angle_diff(mt, angle1 % np.pi)
        d2 = circular_angle_diff(mt, angle2 % np.pi)
        if d1 < np.radians(10) and d1 <= d2:
            group1.append((rho, theta))
        elif d2 < np.radians(15):
            group2.append((rho, theta))

    print(f"[Refine] Board lines: G1={len(group1)}, G2={len(group2)}")

    # 聚类
    def cluster(lines, rho_thresh=10):
        if not lines: return []
        lines = sorted(lines, key=lambda l: l[0])
        clusters = [[lines[0]]]
        for r, t in lines[1:]:
            ar = np.mean([c[0] for c in clusters[-1]])
            if abs(r - ar) < rho_thresh:
                clusters[-1].append((r, t))
            else:
                clusters.append([(r, t)])
        return [(np.mean([c[0] for c in cl]),
                 np.arctan2(sum(np.sin(c[1]) for c in cl),
                            sum(np.cos(c[1]) for c in cl)) % (2*np.pi))
                for cl in clusters]

    c1 = cluster(group1, rho_thresh=12)
    c2 = cluster(group2, rho_thresh=12)
    print(f"[Refine] After clustering: G1={len(c1)}, G2={len(c2)}")

    # 用 DP 选 19 条（从 release/Mac0.1 的稳定版本，参数调整为精确间距）
    def select_19(lines, n=19):
        if len(lines) <= n: return lines
        lines = sorted(lines, key=lambda l: l[0])
        rhos = np.array([l[0] for l in lines])
        N = len(rhos)
        diffs = np.diff(rhos)
        valid = diffs[diffs > 5]
        if len(valid) == 0: return lines[:n]
        med_gap = np.median(valid)

        dp = np.full((n, N, N), np.inf)
        parent = np.full((n, N, N), -1, dtype=int)

        for i in range(N):
            for j in range(i+1, N):
                g = rhos[j] - rhos[i]
                if g >= 0.2 * med_gap:
                    dp[1][j][i] = abs(g - med_gap) * 0.5

        for k in range(2, n):
            for j in range(k, N):
                for i in range(k-1, j):
                    gij = rhos[j] - rhos[i]
                    if gij > 3*med_gap or gij < 0.2*med_gap: continue
                    for p in range(max(0, k-2), i):
                        if dp[k-1][i][p] == np.inf: continue
                        gpi = rhos[i] - rhos[p]
                        cost = dp[k-1][i][p] + abs(gij-gpi) + abs(gij-med_gap)*0.3
                        if cost < dp[k][j][i]:
                            dp[k][j][i] = cost
                            parent[k][j][i] = p

        bc = np.inf; bj, bi = -1, -1
        for i in range(n-2, N):
            for j in range(i+1, N):
                if dp[n-1][j][i] < bc:
                    bc = dp[n-1][j][i]; bj, bi = j, i
        if bi == -1:
            return [lines[i] for i in np.linspace(0, N-1, n, dtype=int)]

        path = [bj, bi]
        ck, cj, ci = n-1, bj, bi
        while ck > 1:
            p = parent[ck][cj][ci]; path.append(p)
            cj, ci = ci, p; ck -= 1
        path.reverse()
        return [lines[i] for i in path]

    sel1 = select_19(c1)
    sel2 = select_19(c2)
    print(f"[Refine] Selected: G1={len(sel1)}, G2={len(sel2)}")

    # 可视化
    if debug_show:
        vis = img.copy()
        if len(sel1) >= 2 and len(sel2) >= 2:
            for r, t in sel1:
                a = intersect_lines(r, t, sel2[0][0], sel2[0][1])
                b = intersect_lines(r, t, sel2[-1][0], sel2[-1][1])
                if a and b:
                    cv2.line(vis, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), (0,255,0), 1)
            for r, t in sel2:
                a = intersect_lines(r, t, sel1[0][0], sel1[0][1])
                b = intersect_lines(r, t, sel1[-1][0], sel1[-1][1])
                if a and b:
                    cv2.line(vis, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), (255,200,0), 1)

        # 画角点
        labels = ["TL", "TR", "BR", "BL"]
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        for i, (cx, cy) in enumerate(corners):
            c = colors[i] if i < 4 else (255,255,255)
            cv2.circle(vis, (int(cx),int(cy)), 15, c, -1)
            cv2.circle(vis, (int(cx),int(cy)), 17, (255,255,255), 2)
            if i < 4:
                cv2.putText(vis, labels[i], (int(cx)+20, int(cy)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("[Hybrid] Final Grid", vis)
        cv2.waitKey(0) # Added waitKey

    return sel1, sel2


# =====================================================================
# 主入口
# =====================================================================

class HybridScanner:
    """CNN 引导搜索 + OpenCV 精确拟合的混合棋盘检测器"""

    # ========== 可调参数（方便手工调试）==========
    SCAN_RADIUS = 64       # 切 patch 的半径（128x128）
    CNN_INPUT_SIZE = 128    # CNN 训练时的输入尺寸
    STEP_SIZE_MIN = 40      # 最小步长（像素）
    # ============================================

    def _relocate_precise_corners(self, img, edges, angle1, angle2):
        """
        利用 CNN 边缘点进行局部锁定。彻底对齐 board 的全局主方向（angle1, angle2），
        防止在 3.5 度的偏转下因为 hardcode 0/90 度导致的交叉点离谱。
        """
        h, w = img.shape[:2]
        pts_cnn = {k: v['point'] for k, v in edges.items() if v['point'] is not None}
        if len(pts_cnn) < 4: return []

        def get_refined_line(center_pt, target_angle):
            roi_r = 40
            x1, y1 = max(0, int(center_pt[0])-roi_r), max(0, int(center_pt[1])-roi_r)
            x2, y2 = min(w-1, int(center_pt[0])+roi_r), min(h-1, int(center_pt[1])+roi_r)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0: return None
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            edges_roi = cv2.Canny(gray, 50, 150)
            
            lines = cv2.HoughLines(edges_roi, 1, np.pi/180, 25)
            if lines is None: return None
            
            best_l = None
            min_err = 999
            for entry in lines:
                rho_roi, theta = entry[0]
                if circular_angle_diff(theta, target_angle) < np.radians(10):
                    rho_g = x1 * np.cos(theta) + y1 * np.sin(theta) + rho_roi
                    dist_to_cnn = abs(center_pt[0]*np.cos(theta) + center_pt[1]*np.sin(theta) - rho_g)
                    if dist_to_cnn < 30:
                        if dist_to_cnn < min_err:
                            min_err = dist_to_cnn
                            best_l = (rho_g, theta)
            return best_l

        # angle1 为水平组法线角，angle2 为垂直组法线角
        # dir1 探测的是 L/R 端（垂直边界），dir2 探测的是 T/B 端（水平边界）
        line_l = get_refined_line(pts_cnn['dir1_pos'], angle2)
        line_r = get_refined_line(pts_cnn['dir1_neg'], angle2)
        line_t = get_refined_line(pts_cnn['dir2_neg'], angle1)
        line_b = get_refined_line(pts_cnn['dir2_pos'], angle1)

        # 兜底：如果局部 OpenCV 没反应，严格使用 CNN 点 + 棋盘全局偏移角
        def fallback_l(pt, ang): return (pt[0]*np.cos(ang) + pt[1]*np.sin(ang), ang)

        if not line_l: line_l = fallback_l(pts_cnn['dir1_pos'], angle2)
        if not line_r: line_r = fallback_l(pts_cnn['dir1_neg'], angle2)
        if not line_t: line_t = fallback_l(pts_cnn['dir2_neg'], angle1)
        if not line_b: line_b = fallback_l(pts_cnn['dir2_pos'], angle1)

        # 四条边界线求精确交点
        c_tl = intersect_lines(line_t[0], line_t[1], line_l[0], line_l[1])
        c_tr = intersect_lines(line_t[0], line_t[1], line_r[0], line_r[1])
        c_br = intersect_lines(line_b[0], line_b[1], line_r[0], line_r[1])
        c_bl = intersect_lines(line_b[0], line_b[1], line_l[0], line_l[1])
        
        final_pts = [c_tl, c_tr, c_br, c_bl]
        if any(p is None for p in final_pts):
            final_pts = [pts_cnn['dir1_pos'], pts_cnn['dir1_neg'], 
                         pts_cnn['dir2_pos'], pts_cnn['dir2_neg']]

        return self._sort_corners_geometrically(final_pts)

    def _sort_corners_geometrically(self, pts):
        """精准排序：左上(TL)、右上(TR)、右下(BR)、左下(BL)"""
        pts = np.array(pts)
        tl = pts[np.argmin(pts[:,0] + pts[:,1])]
        br = pts[np.argmax(pts[:,0] + pts[:,1])]
        tr = pts[np.argmin(pts[:,1] - pts[:,0])]
        bl = pts[np.argmax(pts[:,1] - pts[:,0])]
        return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]

    def __init__(self, weights_path, debug_show=True):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from PIL import Image
        import sys
        from pathlib import Path

        self.debug_show = debug_show
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"[*] CNN 加载在: {self.device}")

        MAC_DIR = Path(__file__).parent
        CLASSIFIER_V1_DIR = MAC_DIR.parent / "Classifier_V1"
        if str(CLASSIFIER_V1_DIR) not in sys.path:
            sys.path.insert(0, str(CLASSIFIER_V1_DIR))
        from model import PatchClassifier

        self.model = PatchClassifier(num_classes=4)
        self.model.to(self.device)
        ckpt = torch.load(weights_path, map_location=self.device)
        state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
        
        new_sd = {}
        model_keys = self.model.state_dict().keys()
        for k, v in state_dict.items():
            if k in model_keys:
                new_sd[k] = v
            elif "backbone." + k in model_keys:
                new_sd["backbone." + k] = v
            elif k.replace("backbone.", "") in model_keys:
                new_sd[k.replace("backbone.", "")] = v
        
        self.model.load_state_dict(new_sd, strict=False)
        w_fc = self.model.backbone.fc.weight.abs().mean().item()
        print(f"[*] Model loaded: {len(new_sd)} keys matched, FC weights mean={w_fc:.6f}")
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.CNN_INPUT_SIZE, self.CNN_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes = ['Corner', 'Inner', 'Edge', 'Outer']
        self._torch = torch
        self._Image = Image

        self._patch_vis_boxes = None
        self._patch_vis_canvas = None
        self._captured_patches = []

    def classify_patch(self, img, center_xy, scan_radius=None):
        if scan_radius is None:
            scan_radius = self.SCAN_RADIUS

        h, w = img.shape[:2]
        cx, cy = int(center_xy[0]), int(center_xy[1])

        x1 = max(0, cx - scan_radius)
        y1 = max(0, cy - scan_radius)
        x2 = min(w, cx + scan_radius)
        y2 = min(h, cy + scan_radius)

        patch_bgr = img[y1:y2, x1:x2]
        ph, pw = patch_bgr.shape[:2]
        target = scan_radius * 2
        if ph < target or pw < target:
            patch_bgr = cv2.copyMakeBorder(
                patch_bgr, 0, max(0, target-ph), 0, max(0, target-pw),
                cv2.BORDER_CONSTANT, value=(128,128,128))

        patch_resized = cv2.resize(patch_bgr, (self.CNN_INPUT_SIZE, self.CNN_INPUT_SIZE))
        patch_rgb = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)

        input_tensor = self.transform(self._Image.fromarray(patch_rgb)).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            outputs = self.model(input_tensor)
            probs = self._torch.softmax(outputs, dim=1)[0]
            conf, pred = self._torch.max(probs, 0)

        label = self.classes[pred.item()]
        confidence = conf.item()

        p_canvas = patch_resized.copy()
        cv2.putText(p_canvas, f"{label} {confidence:.2f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        self._captured_patches.append(p_canvas)

        if self._patch_vis_boxes is not None:
            colors = {'Inner': (0,255,0), 'Edge': (0,165,255),
                      'Corner': (0,0,255), 'Outer': (128,128,128)}
            c = colors.get(label, (255,255,255))
            cv2.rectangle(self._patch_vis_boxes, (x1, y1), (x2, y2), c, 2)

        if self._patch_vis_canvas is not None:
            ph, pw = (y2-y1), (x2-x1)
            self._patch_vis_canvas[y1:y2, x1:x2] = patch_bgr[:ph, :pw]
            cv2.rectangle(self._patch_vis_canvas, (x1, y1), (x2, y2), (200,200,200), 1)

        return label, confidence

    def detect(self, img):
        print("\n" + "="*50)
        print("  Hybrid Scanner: CNN-Guided Board Detection")
        print("="*50)
        h, w = img.shape[:2]
        self._patch_vis_boxes = img.copy()
        self._patch_vis_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        self._captured_patches = []

        result = find_seed_and_directions(img, debug_show=self.debug_show)
        _, dir1, dir2, angle1, angle2, est_gap = result
        if dir1 is None: return [], [], [], {}, 0.0, 0.0

        seed = (w // 2, h // 2)
        step = max(self.STEP_SIZE_MIN, est_gap * 1.5)
        self.classify_patch(img, seed)
        edges = self._cnn_search(img, seed, dir1, dir2, step)

        dist_dir1 = 0.0
        dist_dir2 = 0.0
        if edges.get('dir1_pos') and edges.get('dir1_neg') and edges['dir1_pos']['point'] and edges['dir1_neg']['point']:
            p1, p2 = edges['dir1_pos']['point'], edges['dir1_neg']['point']
            dist_dir1 = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
            print(f"  [Edges] Distance between dir1_pos and dir1_neg: {dist_dir1:.2f}")
            
        if edges.get('dir2_pos') and edges.get('dir2_neg') and edges['dir2_pos']['point'] and edges['dir2_neg']['point']:
            p1, p2 = edges['dir2_pos']['point'], edges['dir2_neg']['point']
            dist_dir2 = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
            print(f"  [Edges] Distance between dir2_pos and dir2_neg: {dist_dir2:.2f}")

        if self.debug_show and len(self._captured_patches) > 0:
            n_patches = len(self._captured_patches)
            cols = 8
            rows = (n_patches + cols - 1) // cols
            gallery = np.full((rows * 128, cols * 128, 3), 255, dtype=np.uint8)
            for i, p in enumerate(self._captured_patches):
                r, c = i // cols, i % cols
                gallery[r*128:(r+1)*128, c*128:(c+1)*128] = p
            cv2.imshow("[Hybrid] Patch Gallery (CNN Inputs)", gallery)
            cv2.imshow("[Hybrid] Search Path (Original)", self._patch_vis_boxes)
            cv2.imshow("[Hybrid] Search Path (White Canvas)", self._patch_vis_canvas)

        corners = self._relocate_precise_corners(img, edges, angle1, angle2)
        if len(corners) < 4: return [], [], [], edges, dist_dir1, dist_dir2

        labels_name = ['TL', 'TR', 'BR', 'BL']
        for i, (cx, cy) in enumerate(corners):
            label, conf = self.classify_patch(img, (int(cx), int(cy)))
            tag = labels_name[i] if i < 4 else f'C{i}'
            print(f"  [{tag}] ({int(cx)},{int(cy)}) -> {label} ({conf:.2f})")

        sel_h, sel_v = refine_grid_with_known_bounds(img, corners, angle1, angle2, debug_show=self.debug_show)
        cv2.waitKey(0)
        return corners, sel_h, sel_v, edges, dist_dir1, dist_dir2

    def _cnn_search(self, img, seed_pt, dir1, dir2, step_size):
        edges = {}
        for name, direction in [('dir1_pos', dir1), ('dir1_neg', -dir1),
                                ('dir2_pos', dir2), ('dir2_neg', -dir2)]:
            edge_pt, edge_type, path = self._search_one_direction(img, seed_pt, direction, step_size)
            edges[name] = {'point': edge_pt, 'type': edge_type, 'path': path}
        return edges

    def _search_one_direction(self, img, start_pt, direction, step_size,
                               max_steps=25):
        """沿一个方向搜索，返回 (edge_point, edge_type, path)"""
        h, w = img.shape[:2]
        x, y = float(start_pt[0]), float(start_pt[1])
        dx = direction[0] * step_size
        dy = direction[1] * step_size
        sr = self.SCAN_RADIUS

        path = []
        last_inner_pt = None

        for step in range(max_steps):
            xi, yi = int(round(x)), int(round(y))

            # 边界检查
            if xi < sr or xi >= w - sr or yi < sr or yi >= h - sr:
                print(f"    step {step}: ({xi},{yi}) -> OUT OF BOUNDS")
                if last_inner_pt:
                    return last_inner_pt, 'Edge', path
                return None, None, path

            label, conf = self.classify_patch(img, (xi, yi))
            path.append({'x': xi, 'y': yi, 'label': label, 'conf': conf})
            print(f"    step {step}: ({xi},{yi}) -> {label} ({conf:.2f})")

            if label == 'Inner':
                last_inner_pt = (xi, yi)
            elif label in ('Edge', 'Corner'):
                # 寻找边界的过程：我们期望从 Inner 开始，遇到第一个 Edge/Corner 停止
                # 如果第一步就是 Edge/Corner，通常意味着起点选得太靠近边界，或者分类器误判。
                # 这种情况下，我们应该继续走，直到找到真正的 Inner 或是多次确认边界。
                if step == 0:
                    print(f"    (起点就是 {label}，我们将尝试继续寻找 Inner 或确认边界...)")
                    # 如果起点就是 Edge，由于我们要找边界，这里不能直接返回，否则搜索瞬间结束。
                    # 我们暂时记下这个可能，但继续移动以观察变化。
                else:
                    print(f"  >> Found {label} at step {step}: ({xi},{yi})")
                    return (xi, yi), label, path
            elif label == 'Outer':
                if last_inner_pt:
                    edge_pt = self._binary_search(img, last_inner_pt, (xi, yi))
                    return edge_pt, 'Edge', path
                else:
                    print(f"    step {step}: Immediately Outer, no inner found")
                    return None, 'Outer', path

            x += dx
            y += dy

        print(f"  [Search] Max steps reached")
        if last_inner_pt:
            return last_inner_pt, 'Edge', path
        return None, None, path

    def _binary_search(self, img, inner_pt, outer_pt, max_iter=6):
        """在 inner 和 outer 之间二分查找边界"""
        ix, iy = float(inner_pt[0]), float(inner_pt[1])
        ox, oy = float(outer_pt[0]), float(outer_pt[1])
        for _ in range(max_iter):
            mx, my = (ix+ox)/2, (iy+oy)/2
            label, _ = self.classify_patch(img, (int(mx), int(my)))
            if label == 'Inner':
                ix, iy = mx, my
            else:
                ox, oy = mx, my
        return (int(round((ix+ox)/2)), int(round((iy+oy)/2)))

    def _search_for_inner(self, img, start, step):
        """在 start 附近螺旋搜索一个 Inner 点"""
        h, w = img.shape[:2]
        sr = self.SCAN_RADIUS
        # 螺旋探测更多点，探测间距缩小
        dist_steps = [step*0.2, step*0.5, step, step*1.5, step*2, step*3]
        for radius in dist_steps:
            for angle in range(0, 360, 30):
                rad = np.radians(angle)
                x = int(start[0] + radius * np.cos(rad))
                y = int(start[1] + radius * np.sin(rad))
                if sr <= x < w-sr and sr <= y < h-sr:
                    label, _ = self.classify_patch(img, (x, y))
                    if label == 'Inner':
                        print(f"  [Search] Found Inner at ({x},{y})")
                        return (x, y)
        return None

