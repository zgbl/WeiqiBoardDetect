import cv2
import numpy as np
import sys
from pathlib import Path

# =====================================================================
# 底层几何工具
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
# 核心算法：角度分组 (从 opencv_engine.py 移植)
# =====================================================================
def separate_by_angle(lines, v_tol=18, h_tol=10):
    if len(lines) == 0:
        return [], [], [], 0, 0

    normalized = []
    for rho, theta in lines:
        r, t = normalize_line(rho, theta)
        normalized.append((r, t))

    mapped_thetas = np.array([t % np.pi for _, t in normalized])

    n_bins = 180
    bin_size = np.pi / n_bins 
    hist = np.zeros(n_bins)
    for theta in mapped_thetas:
        bin_idx = int(theta / bin_size) % n_bins
        hist[bin_idx] += 1

    kernel_size = 7
    kernel = np.ones(kernel_size) / kernel_size
    padded = np.concatenate([hist[-kernel_size:], hist, hist[:kernel_size]])
    smoothed = np.convolve(padded, kernel, mode='same')[kernel_size:-kernel_size]

    best_h_score = -1
    peak1_bin = -1
    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        if circular_angle_diff(bin_angle, np.pi/2) < np.radians(5):
            if smoothed[i] > best_h_score:
                best_h_score = smoothed[i]
                peak1_bin = i

    if peak1_bin == -1 or best_h_score < 1e-3:
        peak1_bin = int(np.argmax(smoothed))

    peak1_angle = (peak1_bin + 0.5) * bin_size 

    best_v_score = -1
    peak2_bin = -1
    target_peak2 = (peak1_angle + np.pi/2) % np.pi
    
    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        if circular_angle_diff(bin_angle, target_peak2) < np.radians(15):
            if smoothed[i] > best_v_score:
                best_v_score = smoothed[i]
                peak2_bin = i

    if peak2_bin == -1 or best_v_score < 0.05 * smoothed[peak1_bin]:
        print("[Angle] No valid orthogonal peak found or it's too weak. Fallback to basic angle search.")
        return [], [], [], peak1_angle, (peak1_angle + np.pi/2) % np.pi

    peak2_angle = (peak2_bin + 0.5) * bin_size

    def get_tol(angle):
        diff_to_90 = circular_angle_diff(angle, np.pi/2)
        if diff_to_90 < np.radians(45):
            return np.radians(h_tol)
        else:
            return np.radians(v_tol)

    def dynamic_perspective_filter(group, peak):
        if len(group) < 3: return group
        group = sorted(group, key=lambda x: x[0]) # 从左到右或从上到下
        
        # 寻找最接近中间且角度比较正常的一根基准线
        mid_idx = len(group) // 2
        best_center_idx = mid_idx
        for offset in range(len(group)):
            idx = mid_idx + offset
            if idx < len(group) and circular_angle_diff(group[idx][1], peak) < np.radians(10):
                best_center_idx = idx
                break
            idx = mid_idx - offset
            if idx >= 0 and circular_angle_diff(group[idx][1], peak) < np.radians(10):
                best_center_idx = idx
                break
                
        valid_lines = [group[best_center_idx]]
        
        # 从中间向右/下逐步追踪
        curr_angle = group[best_center_idx][1]
        for i in range(best_center_idx + 1, len(group)):
            r, t = group[i]
            # 动态容差：当前线必须和相近的一根线保持近似的角度（最多偏转12度），包容极限透视畸变！
            if circular_angle_diff(t, curr_angle) < np.radians(12):
                valid_lines.append((r, t))
                curr_angle = t
                
        # 从中间向左/上逐步追踪
        curr_angle = group[best_center_idx][1]
        for i in range(best_center_idx - 1, -1, -1):
            r, t = group[i]
            if circular_angle_diff(t, curr_angle) < np.radians(12):
                valid_lines.append((r, t))
                curr_angle = t
                
        return sorted(valid_lines, key=lambda x: x[0])

    group1, group2, ignored = [], [], []
    for i, (rho, theta) in enumerate(normalized):
        mt = mapped_thetas[i]
        d1 = circular_angle_diff(mt, peak1_angle)
        d2 = circular_angle_diff(mt, peak2_angle)
        
        # 先做粗糙的角度二分类 (宽松45度)
        if d1 < d2:
            if d1 < np.radians(45):
                group1.append((rho, theta))
            else:
                ignored.append((rho, theta))
        else:
            if d2 < np.radians(45):
                group2.append((rho, theta))
            else:
                ignored.append((rho, theta))

    # 完全符合用户的指导：“纵向角度必须从中间算，向两边寻找”
    group1 = dynamic_perspective_filter(group1, peak1_angle)
    group2 = dynamic_perspective_filter(group2, peak2_angle)

    print(f"[Angle] Peak1={np.degrees(peak1_angle):.1f}° | Peak2={np.degrees(peak2_angle):.1f}°")
    print(f"[Angle] Final Tracking filter assignment: G1={len(group1)}, G2={len(group2)}, Total initial input={len(normalized)}")
    return group1, group2, ignored, peak1_angle, peak2_angle


# =====================================================================
# 核心算法：聚类 (从 opencv_engine.py 移植)
# =====================================================================
def cluster_lines(lines, rho_threshold=6, theta_threshold_deg=2):
    if not lines:
        return []

    lines = sorted(lines, key=lambda l: l[0])
    clusters = []

    for rho, theta in lines:
        found_cluster = False
        for cluster in clusters:
            avg_rho = np.mean([item[0] for item in cluster])
            sum_sin = np.sum([np.sin(item[1]) for item in cluster])
            sum_cos = np.sum([np.cos(item[1]) for item in cluster])
            avg_theta = np.arctan2(sum_sin, sum_cos)

            d_rho = abs(rho - avg_rho)
            d_theta = circular_angle_diff(theta, avg_theta)

            if d_rho < rho_threshold and d_theta < np.radians(theta_threshold_deg):
                cluster.append((rho, theta))
                found_cluster = True
                break
        
        if not found_cluster:
            clusters.append([(rho, theta)])

    result = []
    for cluster in clusters:
        final_rho = np.mean([l[0] for l in cluster])
        sum_sin = np.sum([np.sin(l[1]) for l in cluster])
        sum_cos = np.sum([np.cos(l[1]) for l in cluster])
        final_theta = np.arctan2(sum_sin, sum_cos)
        if final_theta < 0: final_theta += 2 * np.pi
        result.append((final_rho, final_theta))

    print(f"[Cluster] Aggregated {len(lines)} raw lines into {len(result)} clustered lines")
    return sorted(result, key=lambda x: x[0])


# =====================================================================
# 核心算法：DP 筛选 (增加 external_expected_gap 约束)
# =====================================================================
def select_n_evenly_spaced(lines, n=19, group_peak_angle=0, external_expected_gap=None):
    if len(lines) <= n:
        return lines

    lines = sorted(lines, key=lambda l: l[0])
    rhos = np.array([l[0] for l in lines])
    
    if external_expected_gap and external_expected_gap > 10:
        expected_gap = external_expected_gap
        print(f"[select_n_evenly_spaced] Using external_expected_gap: {expected_gap:.2f}")
    else:
        span = rhos[-1] - rhos[0]
        expected_gap = span / (n - 1)
        print(f"[select_n_evenly_spaced] Estimated expected_gap from span: {expected_gap:.2f}")
    
    cleaned_lines = []
    if len(lines) > 0:
        cleaned_lines.append(lines[0])
        for i in range(1, len(lines)):
            curr_rho, curr_theta = lines[i]
            prev_rho, prev_theta = cleaned_lines[-1]
            # 宽容透视形变，只合并极度靠近的干扰线 (从 0.7 降到 0.2 或 0.3)
            min_allowed_gap = 10
            if (curr_rho - prev_rho) < min_allowed_gap:
                pass
            else:
                cleaned_lines.append(lines[i])
    
    print(f"[select_n_evenly_spaced] Pre-filtering lines based on 0.7*expected_gap: {len(lines)} drops to {len(cleaned_lines)}")
    
    lines = cleaned_lines
    if len(lines) <= n: return lines

    rhos = np.array([l[0] for l in lines])
    thetas = [l[1] for l in lines]
    N = len(rhos)
    
    diffs = np.diff(rhos)
    valid_diffs = diffs[diffs > 0.4 * expected_gap]
    typical_gap = np.median(valid_diffs) if len(valid_diffs) >= 5 else expected_gap
    typical_gap = max(typical_gap, 0.4 * expected_gap)

    dp = np.full((n, N, N), np.inf)
    parent = np.full((n, N, N), -1, dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            gap = rhos[j] - rhos[i]
            if gap < 0.2 * typical_gap or gap > 3.5 * expected_gap: continue
            
            ang_diff = circular_angle_diff(thetas[j], thetas[i])
            angle_penalty = (np.degrees(ang_diff) * 10.0) ** 2
            
            dp[1][j][i] = abs(gap - typical_gap) * 1.5 + angle_penalty

    for k in range(2, n):
        for j in range(k, N):
            for i in range(k - 1, j):
                if dp[k-1][i].min() == np.inf: continue
                gap_ij = rhos[j] - rhos[i]
                # 宽容透视，严格遵循要求：跨度 > expected_gap*3.5 的直接一票否决
                if gap_ij < 0.2 * typical_gap or gap_ij > 3.5 * expected_gap: continue
                
                ang_diff = circular_angle_diff(thetas[j], thetas[i])
                angle_penalty = (np.degrees(ang_diff) * 10.0) ** 2
                    
                best_cost = np.inf
                best_p = -1
                for p in range(k - 2, i):
                    if dp[k-1][i][p] == np.inf: continue
                    gap_pi = rhos[i] - rhos[p]
                    # 惩罚项：如果跨度超过 1.5 倍 expected_gap，说明有跳格子（丢失线）
                    skip_penalty = 0
                    if gap_ij > 1.5 * expected_gap: skip_penalty += abs(gap_ij - expected_gap) * 2.0
                    cost = dp[k-1][i][p] + abs(gap_ij - gap_pi) + abs(gap_ij - typical_gap) * 0.5 + angle_penalty + skip_penalty
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p
                if best_p != -1:
                    dp[k][j][i] = best_cost
                    parent[k][j][i] = best_p

    final_best_cost = np.inf
    best_j, best_i = -1, -1
    for i in range(n - 2, N):
        for j in range(i + 1, N):
            if dp[n-1][j][i] == np.inf: continue
            
            curr_k, curr_j, curr_i = n - 1, j, i
            start_p = curr_i
            while curr_k > 1:
                p = parent[curr_k][curr_j][curr_i]
                start_p = p
                curr_j, curr_i = curr_i, p
                curr_k -= 1
                
            span = rhos[j] - rhos[start_p]
            expected_span = expected_gap * (n - 1)
            span_penalty = abs(span - expected_span) * 2.0
            
            # 使用包含：原本的累积代价 + 整个跨度与CNN预测跨度的差值惩罚
            cost = dp[n-1][j][i] + span_penalty
            if cost < final_best_cost:
                final_best_cost = cost
                best_j, best_i = j, i
                
    if best_i == -1:
        print("[select_n_evenly_spaced] Warning: Falling back to linear space, no valid path!")
        return [lines[idx] for idx in np.linspace(0, N-1, n, dtype=int)]

    path = [best_j, best_i]
    curr_k, curr_j, curr_i = n - 1, best_j, best_i
    while curr_k > 1:
        p = parent[curr_k][curr_j][curr_i]
        path.append(p)
        curr_j, curr_i = curr_i, p
        curr_k -= 1
        
    path.reverse()
    return [lines[idx] for idx in path]


# =====================================================================
# 阶段 1: OpenCV 粗检测 — 找种子区域和方向 (增强版)
# =====================================================================
def find_seed_and_directions(img, debug_show=False):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    edges = cv2.Canny(enhanced, 50, 200)

    raw_segs = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    if raw_segs is None or len(raw_segs) < 10:
        print("[Seed] Not enough line segments detected")
        return None, None, None, 0, 0, 0

    lines_rt = []
    segments = []
    for seg in raw_segs:
        x1, y1, x2, y2 = seg[0]
        rt = segment_to_rho_theta(x1, y1, x2, y2)
        if rt:
            lines_rt.append(rt)
            segments.append((x1, y1, x2, y2))

    # 使用 opencv_engine.py 中稳健的分离逻辑 (加大纵横角度包容度，应对极致透视)
    group1, group2, ignored, angle1, angle2 = separate_by_angle(lines_rt, v_tol=35, h_tol=15)
    
    if len(group1) == 0 or len(group2) == 0:
        print("[Seed] separate_by_angle failed to form two robust groups.")
        return None, None, None, 0, 0, 0

    print(f"[Seed] Initial separating by angle: G1={len(group1)}, G2={len(group2)}")
    dir1 = np.array([-np.sin(angle1), np.cos(angle1)])
    dir2 = np.array([-np.sin(angle2), np.cos(angle2)])

    # 禁用密度搜索，强行使用图像硬中心作为初始起步点，防止被背景木纹等误导！
    seed_x, seed_y = w // 2, h // 2
    print(f"[Seed] Identified seed point (Image Center): x={seed_x}, y={seed_y}")

    g1_rhos = [r for r, t in group1]
    g2_rhos = [r for r, t in group2]

    def median_gap(rhos):
        if len(rhos) < 3: return 60
        rhos = sorted(rhos)
        diffs = np.diff(rhos)
        valid = diffs[diffs > 5]
        return np.median(valid) if len(valid) > 3 else 60

    gap1 = median_gap(g1_rhos)
    gap2 = median_gap(g2_rhos)
    est_gap = (gap1 + gap2) / 2
    
    print(f"[Seed] Est gap estimation: For group1={gap1:.2f}, group2={gap2:.2f} -> Average={est_gap:.2f}")

    return (seed_x, seed_y), dir1, dir2, angle1, angle2, est_gap


# =====================================================================
# 阶段 5: 精确拟合
# =====================================================================
def refine_grid_with_known_bounds(img, corners, angle1, angle2, dist_dir1, dist_dir2, debug_show=False):
    h, w = img.shape[:2]
    if len(corners) < 4:
        return [], []

    tl, tr, br, bl = [np.array(c) for c in corners[:4]]

    board_poly = np.array([tl, tr, br, bl], dtype=np.int32)
    board_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(board_mask, board_poly, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    board_mask = cv2.dilate(board_mask, kernel, iterations=1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    edges = cv2.Canny(enhanced, 50, 200)
    edges_masked = cv2.bitwise_and(edges, board_mask)

    raw_segs = cv2.HoughLinesP(edges_masked, 1, np.pi/180, 80, minLineLength=40, maxLineGap=10)
    print(f"\n[Refine Grid] Masked board region. Raw HoughLinesP count: {len(raw_segs) if raw_segs is not None else 0}")
    
    if raw_segs is None:
        return [], []

    lines_rt = []
    for seg in raw_segs:
        rt = segment_to_rho_theta(*seg[0])
        if rt: lines_rt.append(rt)

    # 极大放宽垂直容差，因为棋盘右侧由于消失点原理，纵线角度偏斜极其严重！
    group1, group2, ignored, p1, p2 = separate_by_angle(lines_rt, v_tol=35, h_tol=15)
    
    print(f"[Refine Grid] separate_by_angle result: G1={len(group1)}, G2={len(group2)}, ignored={len(ignored)}")
    
    # 严格的聚类
    c1 = cluster_lines(group1, rho_threshold=6)
    c2 = cluster_lines(group2, rho_threshold=6)
    
    print(f"[Refine Grid] cluster_lines result: C1={len(c1)}, C2={len(c2)}")
    
    if debug_show:
        debug_img = img.copy()
        def draw_inf_line(canvas, rho, theta, color, thickness):
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 3000 * (-b)), int(y0 + 3000 * a))
            pt2 = (int(x0 - 3000 * (-b)), int(y0 - 3000 * a))
            cv2.line(canvas, pt1, pt2, color, thickness)
        
        for r, t in c1: draw_inf_line(debug_img, r, t, (0, 255, 0), 1)
        for r, t in c2: draw_inf_line(debug_img, r, t, (255, 200, 0), 1)
        cv2.imshow("[Refine Grid] Clustered Lines (C1=Green, C2=Yellow)", debug_img)

    # 用取得的棋盘两对边的距离除以 18 计算外部期望间距
    # dir1 是 group1 (水平或垂直播动一组)
    # 因为 dir1 方向上的长度被 dir2 方向的边截断，所以 dist_dir1 就是针对 group2 线组的最佳参考尺度
    # 同理 dist_dir2 是对 group1 线的参考。
    gap_g2 = dist_dir1 / 18.0 if dist_dir1 > 0 else None
    gap_g1 = dist_dir2 / 18.0 if dist_dir2 > 0 else None

    sel1 = select_n_evenly_spaced(c1, 19, group_peak_angle=p1, external_expected_gap=gap_g1)
    sel2 = select_n_evenly_spaced(c2, 19, group_peak_angle=p2, external_expected_gap=gap_g2)
    
    print(f"[Refine Grid] select_n_evenly_spaced total picked: Sel1={len(sel1)} / 19, Sel2={len(sel2)} / 19")

    if debug_show:
        debug_img2 = img.copy()
        def draw_inf_line(canvas, rho, theta, color, thickness):
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 3000 * (-b)), int(y0 + 3000 * a))
            pt2 = (int(x0 - 3000 * (-b)), int(y0 - 3000 * a))
            cv2.line(canvas, pt1, pt2, color, thickness)
        for r, t in sel1: draw_inf_line(debug_img2, r, t, (0, 255, 0), 2)
        for r, t in sel2: draw_inf_line(debug_img2, r, t, (255, 200, 0), 2)
        cv2.imshow("[Refine Grid] Final Selected 19x19 Lines", debug_img2)

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

        labels = ["TL", "TR", "BR", "BL"]
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        for i, (cx, cy) in enumerate(corners):
            c = colors[i] if i < 4 else (255,255,255)
            cv2.circle(vis, (int(cx),int(cy)), 15, c, -1)
            cv2.circle(vis, (int(cx),int(cy)), 17, (255,255,255), 2)
            if i < 4:
                cv2.putText(vis, labels[i], (int(cx)+20, int(cy)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # [DEBUG] Coordinate System Direction clarification (All 4 Corners)
        font_sz, thick = 1.0, 3
        cv2.putText(vis, f"(0,0) Origin", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0,0,255), thick)
        pos_y = min(1400, h - 100)
        cv2.putText(vis, f"(0,{pos_y}) Y-Axis", (30, pos_y), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0,0,255), thick)
        cv2.putText(vis, f"({w},0) X-Axis", (w-350, 100), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0,0,255), thick)
        cv2.putText(vis, f"({w},{pos_y}) Corner", (w-450, pos_y), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0,0,255), thick)

        cv2.imshow("[Hybrid V2] Final Grid", vis)

    return sel1, sel2


# =====================================================================
# Main Class V2
# =====================================================================
class HybridScannerV5:
    """结合强大的 OpenCV Engine 聚类与 DP 筛选 + CNN 寻边的全新整合版"""
    SCAN_RADIUS = 64
    CNN_INPUT_SIZE = 128
    STEP_SIZE_MIN = 40

    def _relocate_precise_corners(self, img, edges, angle1, angle2):
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

        line_l = get_refined_line(pts_cnn['dir1_pos'], angle2)
        line_r = get_refined_line(pts_cnn['dir1_neg'], angle2)
        line_t = get_refined_line(pts_cnn['dir2_neg'], angle1)
        line_b = get_refined_line(pts_cnn['dir2_pos'], angle1)

        def fallback_l(pt, ang): return (pt[0]*np.cos(ang) + pt[1]*np.sin(ang), ang)

        if not line_l: line_l = fallback_l(pts_cnn['dir1_pos'], angle2)
        if not line_r: line_r = fallback_l(pts_cnn['dir1_neg'], angle2)
        if not line_t: line_t = fallback_l(pts_cnn['dir2_neg'], angle1)
        if not line_b: line_b = fallback_l(pts_cnn['dir2_pos'], angle1)

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
        pts = np.array(pts)
        tl = pts[np.argmin(pts[:,0] + pts[:,1])]
        br = pts[np.argmax(pts[:,0] + pts[:,1])]
        tr = pts[np.argmin(pts[:,1] - pts[:,0])]
        bl = pts[np.argmax(pts[:,1] - pts[:,0])]
        return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]

    def __init__(self, weights_path, debug_show=True):
        import torch
        from torchvision import transforms
        from PIL import Image

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
        self._corner_debug_patches = {}  # Store [roi, edges, enhanced, harris, final] for each corner


    def classify_patch(self, img, center_xy, scan_radius=None):
        if scan_radius is None:
            scan_radius = self.SCAN_RADIUS

        h, w = img.shape[:2]
        cx, cy = int(center_xy[0]), int(center_xy[1])

        x1 = max(0, cx - scan_radius)
        y1 = max(0, cy - scan_radius)
        x2 = min(w, cx + scan_radius)
        y2 = min(h, cy + scan_radius)
        
        if x1 >= x2 or y1 >= y2:
            return 'Outer', 1.0

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
        print("  Hybrid Scanner V5 (Enhanced OpenCV Engine)")
        print("="*50)
        h, w = img.shape[:2]
        self._patch_vis_boxes = img.copy()
        # [DEBUG] Coordinate System Direction clarification (All 4 Corners)
        font_sz, thick = 1.0, 3
        cv2.putText(self._patch_vis_boxes, f"(0,0) Origin", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0,0,255), thick)
        pos_y = min(1400, h - 100)
        cv2.putText(self._patch_vis_boxes, f"(0,{pos_y}) Y-Axis", (30, pos_y), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0,0,255), thick)
        cv2.putText(self._patch_vis_boxes, f"({w},0) X-Axis", (w-350, 100), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0,0,255), thick)
        cv2.putText(self._patch_vis_boxes, f"({w},{pos_y}) Corner", (w-450, pos_y), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0,0,255), thick)

        self._patch_vis_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        self._captured_patches = []
        self._corner_debug_patches = {}


        result = find_seed_and_directions(img, debug_show=self.debug_show)
        _, dir1, dir2, angle1, angle2, est_gap = result
        if dir1 is None: return [], [], [], {}, 0.0, 0.0

        seed = (w // 2, h // 2)
        # 调大步长（从30增加到60），加快推边速度，减少对局部噪点的敏感度
        step = 45
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
            cv2.imshow("[Hybrid V2] Patch Gallery", gallery)
            cv2.imshow("[Hybrid V2] Search Path", self._patch_vis_boxes)

        corners = self._relocate_precise_corners(img, edges, angle1, angle2)
        if len(corners) < 4: return [], [], [], edges, dist_dir1, dist_dir2

        # 4> 接入新增的流程，依次精确定位 4 个盘角
        labels_name = ['TL', 'TR', 'BR', 'BL']
        new_corners = []
        
        debug_canvas = img.copy() if self.debug_show else None
        
        for i, pt in enumerate(corners):
            tag = labels_name[i] if i < 4 else f'C{i}'
            exact_pt = self.exact_recognize_corner(img, pt, tag, angle1, angle2, est_gap, debug_canvas)
            new_corners.append(exact_pt)
            
        if self.debug_show and debug_canvas is not None:
            cv2.imshow("[Exact Corner] Tracking Trajectory", debug_canvas)
            
        # 替换！现在corners将包含极度物理贴合的边缘角点了
        corners = new_corners

        # [DEBUG] Show 4x5 Corner Patch Grid
        if self.debug_show and self._corner_debug_patches:
            pw, ph = 160, 160 # Reduced from 200 to avoid 'magnified' feeing on high-DPI screens
            max_cols = max([len(patches) for patches in self._corner_debug_patches.values()] + [5])
            grid = np.full((ph * 4, pw * max_cols, 3), 255, dtype=np.uint8)
            for r, tag in enumerate(['TL', 'TR', 'BR', 'BL']):
                patches = self._corner_debug_patches.get(tag, [])
                for c, p in enumerate(patches):
                    if p is not None:
                        # Use INTER_AREA for higher quality downscaling
                        p_resized = cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA)
                        grid[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = p_resized
            cv2.imshow("[Debug] Corner Grid (Orig, Canny, Enh, Harris, Final, ...)", grid)






        for i, (cx, cy) in enumerate(corners):

            label, conf = self.classify_patch(img, (int(cx), int(cy)))
            tag = labels_name[i] if i < 4 else f'C{i}'
            print(f"  [{tag}] ({int(cx)},{int(cy)}) -> {label} ({conf:.2f})")

        sel_h, sel_v = refine_grid_with_known_bounds(
            img, corners, angle1, angle2, dist_dir1, dist_dir2, debug_show=self.debug_show)
        
        cv2.waitKey(0)
        return corners, sel_h, sel_v, edges, dist_dir1, dist_dir2

    def _cnn_search(self, img, seed_pt, dir1, dir2, step_size):
        print(f"  [CNN Search] Starting search from seed {seed_pt} with step size {step_size}")
        edges = {}
        for name, direction in [('dir1_pos', dir1), ('dir1_neg', -dir1),
                                ('dir2_pos', dir2), ('dir2_neg', -dir2)]:
            edge_pt, edge_type, path = self._search_one_direction(img, seed_pt, direction, step_size, name=name)
            edges[name] = {'point': edge_pt, 'type': edge_type, 'path': path}
            print(f"  [CNN Search] Direction {name}: Found {edge_type} at {edge_pt}")
        return edges

    def _search_one_direction(self, img, start_pt, direction, step_size, max_steps=100, name=""):
        h, w = img.shape[:2]
        x, y = float(start_pt[0]), float(start_pt[1])
        dx = direction[0] * step_size
        dy = direction[1] * step_size
        sr = self.SCAN_RADIUS

        path = []
        last_inner_pt = None

        for step_idx in range(max_steps):
            xi, yi = int(round(x)), int(round(y))

            if xi < sr or xi >= w - sr or yi < sr or yi >= h - sr:
                print(f"    [Search {name}] Triggered boundary at ({xi}, {yi}) after {step_idx} steps")
                if last_inner_pt:
                    return last_inner_pt, 'Edge', path
                return None, None, path

            label, conf = self.classify_patch(img, (xi, yi))
            print(f"    [Search {name} Step {step_idx}]: ({xi}, {yi}) -> {label} (conf: {conf:.2f})")
            path.append({'x': xi, 'y': yi, 'label': label, 'conf': conf})

            if label == 'Inner':
                last_inner_pt = (xi, yi)
            elif label in ('Edge', 'Corner'):
                if step_idx != 0:
                    print(f"    [Search {name}] Found {label} at step {step_idx}, returning result.")
                    return (xi, yi), label, path
            elif label == 'Outer':
                if last_inner_pt:
                    edge_pt = self._binary_search(img, last_inner_pt, (xi, yi))
                    print(f"    [Search {name}] Hit Outer at step {step_idx}, binary search snapped to Edge: {edge_pt}")
                    return edge_pt, 'Edge', path
                else:
                    print(f"    [Search {name}] Hit Outer at step {step_idx} with no Inner background, returning None")
                    return None, 'Outer', path

            x += dx
            y += dy

        if last_inner_pt:
            print(f"    [Search {name}] Max steps reached ({max_steps}), returning last Inner as Edge: {last_inner_pt}")
            return last_inner_pt, 'Edge', path
        print(f"    [Search {name}] Max steps reached ({max_steps}), no Edge found.")
        return None, None, path

    def _binary_search(self, img, inner_pt, outer_pt, max_iter=6):
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

    def _validate_corner_by_harris_neighbors(self, harris_pts, candidate_pt, corner_type,
                                              expected_gap, angle1, angle2,
                                              gap_tol=0.25, excluded_pts=None):
        """
        验证一个候选点是否是合法的棋盘角。
        逻辑：在沿棋盘网格方向，距离约一个格子的位置，必须各有一个Harris点；
        而在反方向，不应该有（或距离明显不对）。
        excluded_pts: 已确认为噪点的集合，验证时会跳过这些点。
        返回: (is_valid, score, forbidden_suspects)
          forbidden_suspects = 被 forbidden 方向挡住的疑似噪点列表
        """
        if len(harris_pts) == 0:
            return False, 0.0, []

        excluded = excluded_pts or set()
        # 过滤掉已确认的噪点
        filtered_pts = [p for p in harris_pts if p not in excluded]
        pts = np.array(filtered_pts, dtype=np.float32) if filtered_pts else np.array(harris_pts, dtype=np.float32)
        cx, cy = candidate_pt

        dir_along1  = np.array([ np.cos(angle1),  np.sin(angle1)])
        dir_along2  = np.array([ np.cos(angle2),  np.sin(angle2)])

        dir_right = dir_along2
        dir_left  = -dir_along2
        dir_down  = dir_along1
        dir_up    = -dir_along1

        required_dirs = {
            'TL': [dir_right, dir_down],
            'TR': [dir_left,  dir_down],
            'BR': [dir_left,  dir_up  ],
            'BL': [dir_right, dir_up  ],
        }
        forbidden_dirs = {
            'TL': [dir_left,  dir_up  ],
            'TR': [dir_right, dir_up  ],
            'BR': [dir_right, dir_down],
            'BL': [dir_left,  dir_down],
        }

        gap_min = expected_gap * (1 - gap_tol)
        gap_max = expected_gap * (1 + gap_tol)

        print(f"    [Topo Debug] candidate={candidate_pt}, corner={corner_type}")
        print(f"    [Topo Debug] total harris pts in ROI: {len(harris_pts)}, excluded: {len(excluded)}")

        def find_neighbor_in_dir(direction, pts, cx, cy, min_g, max_g):
            best_s = -1
            bp = None
            for p in pts:
                dx, dy = p[0] - cx, p[1] - cy
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_g or dist > max_g:
                    continue
                dot = (dx * direction[0] + dy * direction[1]) / (dist + 1e-8)
                if dot > 0.866:
                    if dot > best_s:
                        best_s = dot
                        bp = p
            return bp, best_s

        required_found = []
        score_sum = 0.0
        for d in required_dirs[corner_type]:
            neighbor, score = find_neighbor_in_dir(d, pts, cx, cy, gap_min, gap_max)
            if neighbor is not None:
                required_found.append(neighbor)
                score_sum += score
            else:
                return False, 0.0, []

        # 检测 forbidden 方向是否有邻居，收集疑似噪点而不是直接否决
        forbidden_suspects = []
        for d in forbidden_dirs[corner_type]:
            neighbor, score = find_neighbor_in_dir(d, pts, cx, cy, gap_min, gap_max)
            if neighbor is not None:
                forbidden_suspects.append(tuple(int(x) for x in neighbor))

        if forbidden_suspects:
            return False, 0.0, forbidden_suspects

        final_score = score_sum / len(required_dirs[corner_type])
        return True, final_score, []

    def _is_noise_point(self, img, suspect_pt, all_harris_global, expected_gap, 
                        angle1, angle2, patch_center, gap_tol=0.25, verify_roi_r=280,
                        excluded_pts=None):
        """
        验证一个疑似噪点是否真的是噪点。
        方法：检查这个点在四个方向是否都有邻居。如果在当前ROI内没有，
        就扩大ROI再检测一次。如果扩大后仍然没有，则确认是噪点。
        
        真正的棋盘格子点至少应该在2个正交方向上各有一个邻居。
        噪点通常是孤立的，只有0~1个邻居。
        excluded_pts: 已确认的噪点，计算邻居时要排除。
        """
        h_img, w_img = img.shape[:2]
        sx, sy = int(suspect_pt[0]), int(suspect_pt[1])
        
        dir_along1 = np.array([np.cos(angle1), np.sin(angle1)])
        dir_along2 = np.array([np.cos(angle2), np.sin(angle2)])
        all_dirs = [dir_along2, -dir_along2, dir_along1, -dir_along1]  # right, left, down, up
        dir_names = ['right', 'left', 'down', 'up']
        
        gap_min = expected_gap * (1 - gap_tol)
        gap_max = expected_gap * (1 + gap_tol)
        excl = excluded_pts or set()
        
        def has_neighbor_in_dir(pts_list, direction):
            """检查某个方向上是否有邻居"""
            for p in pts_list:
                if tuple(int(x) for x in p) in excl:
                    continue
                dx, dy = p[0] - sx, p[1] - sy
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < gap_min or dist > gap_max:
                    continue
                dot = (dx * direction[0] + dy * direction[1]) / (dist + 1e-8)
                if dot > 0.866:
                    return True
            return False

        def check_opposing_pairs(pts_list):
            """
            检查是否有至少一个"对穿"对：
            - right+left 轴上有对穿 (dir_along2 和 -dir_along2 方向都有邻居)
            - down+up 轴上有对穿 (dir_along1 和 -dir_along1 方向都有邻居)
            真正的网格交叉点至少在一个轴上有对穿邻居。
            噪点（如棋盘边缘外的点）只在一侧有邻居，没有对穿。
            """
            has_right = has_neighbor_in_dir(pts_list, dir_along2)
            has_left  = has_neighbor_in_dir(pts_list, -dir_along2)
            has_down  = has_neighbor_in_dir(pts_list, dir_along1)
            has_up    = has_neighbor_in_dir(pts_list, -dir_along1)
            
            rl_pair = has_right and has_left
            ud_pair = has_up and has_down
            n_total = sum([has_right, has_left, has_down, has_up])
            
            return rl_pair or ud_pair, n_total, (has_right, has_left, has_down, has_up)
        
        # 第一次：用当前已有的 harris 点集合检查对穿对
        has_pair, n1, dirs1 = check_opposing_pairs(all_harris_global)
        dir_labels = ['R', 'L', 'D', 'U']
        dir_str = ''.join(l for l, v in zip(dir_labels, dirs1) if v)
        if has_pair:
            print(f"    [Noise Check] {suspect_pt} has opposing pair in current ROI (n={n1}, dirs={dir_str}) -> NOT noise")
            return False
        
        # 对穿对不存在 → 可能是角点，也可能是噪点
        # 用更严格的间距容差做"网格一致性"验证：
        # 真正的角点的邻居间距与网格高度一致，噪点的间距偏离较大
        tight_gap_min = expected_gap * 0.75   # 25% 容差
        tight_gap_max = expected_gap * 1.25
        
        def count_tight_neighbors(pts_list):
            """用更严格的间距容差计数邻居"""
            n = 0
            for d in all_dirs:
                for p in pts_list:
                    if tuple(int(x) for x in p) in excl:
                        continue
                    dx, dy = p[0] - sx, p[1] - sy
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < tight_gap_min or dist > tight_gap_max:
                        continue
                    dot = (dx * d[0] + dy * d[1]) / (dist + 1e-8)
                    if dot > 0.866:
                        n += 1
                        break
            return n
        
        # 第二次：扩大 ROI 重新检测 Harris，获取更多上下文
        pcx, pcy = int(patch_center[0]), int(patch_center[1])
        x1 = max(0, pcx - verify_roi_r)
        y1 = max(0, pcy - verify_roi_r)
        x2 = min(w_img - 1, pcx + verify_roi_r)
        y2 = min(h_img - 1, pcy + verify_roi_r)
        roi = img[y1:y2, x1:x2]
        
        if roi.size == 0:
            print(f"    [Noise Check] {suspect_pt} expanded ROI empty -> treating as noise")
            return True
        
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            dst = cv2.cornerHarris(gray, 3, 3, 0.04)
            dst = cv2.dilate(dst, None)
            thresh = 0.05 * dst.max() if dst.max() > 0 else 1.0
            ret, thresh_img = cv2.threshold(dst, thresh, 255, cv2.THRESH_BINARY)
            thresh_img = np.uint8(thresh_img)
            _, _, _, centroids = cv2.connectedComponentsWithStats(thresh_img)
            expanded_pts = [(int(c[0]) + x1, int(c[1]) + y1) for c in centroids[1:]]
        except:
            print(f"    [Noise Check] {suspect_pt} Harris failed on expanded ROI -> treating as noise")
            return True
        
        # 检查扩大ROI后是否有对穿对
        has_pair2, n2, dirs2 = check_opposing_pairs(expanded_pts)
        dir_str2 = ''.join(l for l, v in zip(dir_labels, dirs2) if v)
        if has_pair2:
            print(f"    [Noise Check] {suspect_pt} expanded ROI (r={verify_roi_r}): opposing_pair found (dirs={dir_str2}) -> NOT noise")
            return False
        
        # 仍然没有对穿对 → 最后一关：用紧容差检查网格一致性
        # 真正的角点在紧容差下仍有 ≥2 个邻居（与网格间距高度吻合）
        # 噪点在紧容差下通常只有 0~1 个邻居（间距不规则）
        n_tight = count_tight_neighbors(expanded_pts)
        if n_tight >= 2:
            print(f"    [Noise Check] {suspect_pt} no opposing pair but {n_tight} tight-gap neighbors (gap=[{tight_gap_min:.0f},{tight_gap_max:.0f}]) -> NOT noise (grid-consistent)")
            return False
        
        print(f"    [Noise Check] {suspect_pt} no opposing pair, only {n_tight} tight-gap neighbor(s) (gap=[{tight_gap_min:.0f},{tight_gap_max:.0f}], dirs={dir_str2}) -> NOISE")
        return True

    def _snap_to_best_harris_corner(self, img, hough_pt, patch_center, corner_type,
                                    angle1, angle2, expected_gap, base_roi_r=100):
        """
        在Hough交点附近，找所有Harris候选点，用网格拓扑验证，选出得分最高的。
        如果失败，逐步扩大ROI（最多3次），直到找到合法候选或彻底放弃。

        注意：函数签名增加了 img 和 patch_center 参数，用于扩大ROI时重新抠图。
        hough_pt 是全局坐标的Hough交点，patch_center 是CNN定位的patch中心（全局坐标）。
        """
        h_img, w_img = img.shape[:2]

        MAX_EXPAND = 3
        roi_r = base_roi_r
        extra_harris_plots = []

        for attempt in range(MAX_EXPAND + 1):  # 第0次是正常尝试，第1~3次是扩大重试
            if attempt > 0:
                roi_r = base_roi_r + attempt * 60  # 每次扩大60px: 100->160->220->280
                print(f"  [Harris Topo] Expanding ROI to {roi_r}px (attempt {attempt}/{MAX_EXPAND})")

            # 以 patch_center 为中心重新抠图（不是以hough_pt，保持与原逻辑一致）
            cx, cy = int(patch_center[0]), int(patch_center[1])
            x1 = max(0, cx - roi_r)
            y1 = max(0, cy - roi_r)
            x2 = min(w_img - 1, cx + roi_r)
            y2 = min(h_img - 1, cy + roi_r)
            roi = img[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            harris_debug_plot = roi.copy()

            # Harris检测
            try:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                dst = cv2.cornerHarris(gray, 3, 3, 0.04)
                dst = cv2.dilate(dst, None)
                thresh = 0.05 * dst.max() if dst.max() > 0 else 1.0

                ret, thresh_img = cv2.threshold(dst, thresh, 255, cv2.THRESH_BINARY)
                thresh_img = np.uint8(thresh_img)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img)
                all_harris_global = [(int(c[0]) + x1, int(c[1]) + y1) for c in centroids[1:]]
                
                if attempt > 0:
                    for c in centroids[1:]:
                        cv2.circle(harris_debug_plot, (int(c[0]), int(c[1])), 4, (255, 0, 0), -1)
                    cv2.putText(harris_debug_plot, f"R={roi_r}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    extra_harris_plots.append(harris_debug_plot)
            except Exception as e:
                print(f"  [Harris Topo] Harris failed: {e}")
                continue

            if len(all_harris_global) == 0:
                continue

            print(f"  [Harris List] Detected {len(all_harris_global)} points for {corner_type} (roi_r={roi_r}): {all_harris_global}")

            # 候选点：Hough交点附近 search_r 范围内的Harris点
            hx, hy = hough_pt
            search_r = roi_r  # 搜索半径跟ROI一起扩大
            candidates = [p for p in all_harris_global
                        if abs(p[0] - hx) < search_r and abs(p[1] - hy) < search_r]

            if len(candidates) == 0:
                print(f"  [Harris Topo] No candidates near Hough point, retrying...")
                continue

            # 估算局部gap
            local_gap = self._estimate_local_gap_from_harris(all_harris_global)
            if local_gap is None:
                local_gap = expected_gap
            else:
                # 降低全局对局部的干扰，因为全局估算往往不准
                # 只有当局部估算极其离谱（比如 > 500px 或 < 10px）才放弃
                if local_gap > 500 or local_gap < 10:
                    print(f"  [Harris Topo] Local gap {local_gap:.1f} unreasonable, defaulting to global.")
                    local_gap = expected_gap
                else:
                    print(f"  [Harris Topo] Local gap={local_gap:.1f}px (global={expected_gap:.1f}px)")

            # 拓扑验证，找最高分候选（带噪点排除重试逻辑）
            confirmed_noise = set()
            MAX_NOISE_ROUNDS = 3  # 最多做3轮噪点排除

            for noise_round in range(MAX_NOISE_ROUNDS + 1):
                best_pt = None
                best_score = -999
                all_forbidden_suspects = []

                for cand in candidates:
                    is_valid, score, forbidden_suspects = self._validate_corner_by_harris_neighbors(
                        all_harris_global, cand, corner_type,
                        expected_gap=local_gap, angle1=angle1, angle2=angle2,
                        excluded_pts=confirmed_noise
                    )
                    
                    if is_valid:
                        # 距离惩罚：偏离 CNN 预测中心越远，分数越低
                        # 如果偏离超过 1.5 倍步长 (60px)，基本就是误检了
                        dist_from_cnn = np.sqrt((cand[0] - cx)**2 + (cand[1] - cy)**2)
                        dist_penalty = max(0, dist_from_cnn / 200.0) # 100px 扣 0.5分
                        
                        # 总分 = 拓扑分 (0~1) - 距离扣分
                        final_score = score - dist_penalty

                        # 核心校验：如果候选点被 CNN 判定为 Outer，绝对不可能是角点
                        # 只有当分数有竞争力时才做这个耗时的分类
                        if final_score > best_score:
                            c_label, _ = self.classify_patch(img, (int(cand[0]), int(cand[1])))
                            if c_label == 'Outer':
                                print(f"    [Harris Topo] Rejecting candidate {cand}: CNN labeled as Outer")
                                continue
                            
                            best_score = final_score
                            best_pt = cand
                    elif forbidden_suspects:
                        all_forbidden_suspects.extend(forbidden_suspects)

                if best_pt is not None:
                    dist_hough = np.sqrt((best_pt[0] - hx) ** 2 + (best_pt[1] - hy) ** 2)
                    print(f"  [Harris Topo] Validated (attempt {attempt}, noise_round {noise_round}): {hough_pt} -> {best_pt} "
                        f"(final_score={best_score:.3f}, dist_hough={dist_hough:.1f}px, roi_r={roi_r})")
                    return best_pt, extra_harris_plots

                # 没有通过验证，但有 forbidden suspects → 尝试确认噪点
                if not all_forbidden_suspects:
                    break  # 没有疑似噪点可排除，直接放弃这轮

                # 去重
                unique_suspects = set(all_forbidden_suspects) - confirmed_noise
                if not unique_suspects:
                    break  # 所有疑似噪点都已经被检查过了

                new_noise_found = False
                for suspect in unique_suspects:
                    is_noise = self._is_noise_point(
                        img, suspect, all_harris_global, local_gap,
                        angle1, angle2, patch_center,
                        excluded_pts=confirmed_noise  # 传入已确认噪点，避免虚高邻居数
                    )
                    if is_noise:
                        confirmed_noise.add(suspect)
                        new_noise_found = True
                        print(f"  [Harris Topo] Confirmed noise: {suspect}, will retry validation")

                if not new_noise_found:
                    print(f"  [Harris Topo] No new noise confirmed, forbidden neighbors are real grid points")
                    break  # forbidden 邻居不是噪点，此轮确实不合法

                print(f"  [Harris Topo] Excluded {len(confirmed_noise)} noise pts, retrying validation...")

            print(f"  [Harris Topo] No valid candidate at roi_r={roi_r}, {'retrying...' if attempt < MAX_EXPAND else 'giving up.'}")

        # 全部扩大失败，回退Hough
        print(f"  [Harris Topo] All attempts failed, fallback to Hough intersection")
        return hough_pt, extra_harris_plots




    def exact_recognize_corner(self, img, approx_pt, corner_type, angle1, angle2, expected_gap, debug_canvas=None):
        if corner_type not in ['TL', 'TR', 'BR', 'BL']:
            return approx_pt
            
        print(f"\n[Exact Corner] Searching {corner_type} starting from OpenCV approx: ({int(approx_pt[0])}, {int(approx_pt[1])})")
        # 1> & 2> 基于用户思想要加的重要流程：用CNN在四个方向轮流外推，直至Patch变为 Corner！
        step = 40
        if corner_type == 'TL': v_x, v_y = (-step, 0), (0, -step)
        elif corner_type == 'TR': v_x, v_y = (step, 0), (0, -step)
        elif corner_type == 'BL': v_x, v_y = (-step, 0), (0, step)
        else: v_x, v_y = (step, 0), (0, step)
            
        cx, cy = float(approx_pt[0]), float(approx_pt[1])

        backtrack_count = 0
        MAX_BACKTRACK = 12  # 给予更多机会绕路
        
        h, w = img.shape[:2]
        center_x, center_y = w / 2.0, h / 2.0
        
        visited_positions = set()
        
        # ========== Phase 1: 从 approx 向中心持续后退，穿过 Outer 和 Edge，直到碰到 Inner ==========
        # 记录最后一个 Edge 位置，作为 Phase 2 的起点
        last_edge_pos = None
        
        for k in range(30):
            label, conf = self.classify_patch(img, (int(cx), int(cy)))
            print(f"  [{corner_type} Step {k}]: ({int(cx)}, {int(cy)}) -> {label} (conf: {conf:.2f})")
            
            if debug_canvas is not None:
                color = (0, 255, 0) if label == 'Corner' else (0, 0, 255) if label == 'Outer' else (0, 255, 255) if label == 'Edge' else (255, 0, 0)
                sz = 64
                cv2.rectangle(debug_canvas, (int(cx)-sz, int(cy)-sz), (int(cx)+sz, int(cy)+sz), color, 2)
                cv2.circle(debug_canvas, (int(cx), int(cy)), 3, color, -1)
            
            if label == 'Corner':
                break
            
            if label == 'Edge':
                last_edge_pos = (cx, cy)  # 记录，继续退
            
            if label == 'Inner':
                # 到达 Inner，回到最后一个 Edge 位置，Phase 1 结束
                if last_edge_pos:
                    cx, cy = last_edge_pos
                    print(f"  [{corner_type}] Phase 1 done: Inner reached. Anchoring at last Edge ({int(cx)}, {int(cy)})")
                else:
                    # 直接从 Inner 开始（罕见），向外推一步
                    cx += v_x[0] + v_y[0]
                    cy += v_x[1] + v_y[1]
                    print(f"  [{corner_type}] Phase 1: started in Inner, pushing out to ({int(cx)}, {int(cy)})")
                    continue
                break
            
            # Outer 和 Edge 都继续向中心退
            dx, dy = center_x - cx, center_y - cy
            dist = (dx**2 + dy**2)**0.5
            if dist > 1e-3:
                cx += (dx / dist) * step
                cy += (dy / dist) * step
        
        if label == 'Corner':
            # Phase 1 就找到了 Corner，直接跳到最终精修
            pass
        else:
            # ========== Phase 2: 从深度 Edge 锚点开始，沿棋盘边缘搜索 Corner ==========
            print(f"  [{corner_type}] Phase 2: searching for Corner from ({int(cx)}, {int(cy)})")
            path_history = []
            locked_axis = None
            
            for k2 in range(30):
                label, conf = self.classify_patch(img, (int(cx), int(cy)))
                print(f"  [{corner_type} Search {k2}]: ({int(cx)}, {int(cy)}) -> {label} (conf: {conf:.2f})")
                
                if debug_canvas is not None:
                    color = (0, 255, 0) if label == 'Corner' else (0, 0, 255) if label == 'Outer' else (0, 255, 255) if label == 'Edge' else (255, 0, 0)
                    sz = 64
                    cv2.rectangle(debug_canvas, (int(cx)-sz, int(cy)-sz), (int(cx)+sz, int(cy)+sz), color, 2)
                    cv2.circle(debug_canvas, (int(cx), int(cy)), 3, color, -1)
                
                if label == 'Corner':
                    break
                
                if label == 'Outer':
                    # 推过头了，向中心退一步
                    dx, dy = center_x - cx, center_y - cy
                    dist = (dx**2 + dy**2)**0.5
                    if dist > 1e-3:
                        cx += (dx / dist) * step
                        cy += (dy / dist) * step
                    locked_axis = None
                    continue
                
                if label == 'Inner':
                    # 太深了，向外推
                    cx += v_x[0] + v_y[0]
                    cy += v_x[1] + v_y[1]
                    locked_axis = None
                    continue
                
                if label == 'Edge':
                    # 探测两个方向
                    lx_label, _ = self.classify_patch(img, (int(cx + v_x[0]), int(cy)))
                    ly_label, _ = self.classify_patch(img, (int(cx), int(cy + v_y[1])))
                    print(f"    [Probe] X->({int(cx+v_x[0])},{int(cy)})={lx_label}, Y->({int(cx)},{int(cy+v_y[1])})={ly_label}, locked={locked_axis}")
                    
                    if lx_label == 'Corner':
                        cx += v_x[0]; continue
                    if ly_label == 'Corner':
                        cy += v_y[1]; continue
                    
                    can_move_x = lx_label in ['Inner', 'Edge']
                    can_move_y = ly_label in ['Inner', 'Edge']
                    
                    # 如果某轴被锁死，禁止沿该轴移动
                    if locked_axis == 'X': can_move_x = False
                    elif locked_axis == 'Y': can_move_y = False
                    
                    # ★ 核心：如果两个方向都走不了（都是 Outer），不要 block！
                    #   直接跳过去，让主循环的 Outer 处理逻辑重新定向
                    if not can_move_x and not can_move_y:
                        # 强制前进，主循环会处理 Outer
                        next_cx = cx + v_x[0]
                        next_cy = cy + v_y[1]
                    else:
                        next_cx = cx + v_x[0] if can_move_x else cx
                        next_cy = cy + v_y[1] if can_move_y else cy
                    
                    next_pos = (int(next_cx), int(next_cy))
                    curr_pos = (int(cx), int(cy))
                    
                    if next_pos == curr_pos or next_pos in path_history:
                        reason = "same_pos" if next_pos == curr_pos else f"in_history({next_pos})"
                        print(f"    [Block reason] can_x={can_move_x}, can_y={can_move_y}, next={next_pos}, curr={curr_pos}, reason={reason}")
                        backtrack_count += 1
                        if backtrack_count > MAX_BACKTRACK:
                            print(f"  [{corner_type}] Max backtrack reached!")
                            break
                        
                        # 决定 retreat 方向
                        if locked_axis == 'X':
                            retreat_axis, r_dx, r_dy = 'Y', 0, -v_y[1]
                        elif locked_axis == 'Y':
                            retreat_axis, r_dx, r_dy = 'X', -v_x[0], 0
                        else:
                            if backtrack_count % 2 == 1:
                                retreat_axis, r_dx, r_dy = 'Y', 0, -v_y[1]
                            else:
                                retreat_axis, r_dx, r_dy = 'X', -v_x[0], 0
                        
                        print(f"  [{corner_type}] Blocked at {curr_pos}. Deep retreat along {retreat_axis}...")
                        
                        last_edge_cx, last_edge_cy = cx, cy
                        for r_step in range(6):
                            cx += r_dx; cy += r_dy
                            r_label, _ = self.classify_patch(img, (int(cx), int(cy)))
                            print(f"    [Retreat {r_step}]: ({int(cx)}, {int(cy)}) -> {r_label}")
                            if r_label == 'Edge': last_edge_cx, last_edge_cy = cx, cy
                            elif r_label in ['Inner', 'Board']: break
                            elif r_label == 'Corner': label = 'Corner'; break
                            elif r_label == 'Outer': break
                        
                        if label == 'Corner': break
                        cx, cy = last_edge_cx, last_edge_cy
                        
                        locked_axis = retreat_axis
                        print(f"  [{corner_type}] Locked {locked_axis}. Sliding {'X' if locked_axis == 'Y' else 'Y'} only.")
                        path_history = []
                        continue
                    
                    path_history.append(curr_pos)
                    if len(path_history) > 10:
                        path_history.pop(0)
                    cx, cy = next_cx, next_cy


        # 3> CNN将patch成功压在Corner后，抠出一个大图获取最边缘的L形交叉，此处必须传入全盘的主角度约束！
        if label != 'Corner' and 'Corner' not in [p.get('label', '') for p in []]: # Or just if we never hit corner... wait, label is the last state.
             # 实际上，如果最后状态不是 Corner，说明CNN直接滑偏迷失了，此时最好退回到原始的数学交点
            if label != 'Corner':
                print(f"  [{corner_type} Warning]: CNN never settled on Corner. Reverting to mathematical approx!")
                cx, cy = approx_pt[0], approx_pt[1]
                
        final_pt = self._extract_precise_opencv_l_shape(img, (cx, cy), corner_type, angle1, angle2, expected_gap, debug_canvas)
        print(f"[Exact Corner] OpenCV pinned {corner_type} EXACTLY at: ({int(final_pt[0])}, {int(final_pt[1])})\n")
        return final_pt

    def _estimate_local_gap_from_harris(self, harris_pts):
        """用harris点之间的距离分布，估算局部格子间距"""
        if len(harris_pts) < 2:
            return None
        pts = np.array(harris_pts, dtype=np.float32)
        dists = []
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                dists.append(d)
        dists.sort()
        
        # 我们只取前 N个最短距离（N取 1.5倍点数，足以涵盖最紧凑的轴线邻居）
        # 即使点很多，这个比例也能较好锁定邻居间距
        n_short = max(1, min(len(dists), int(len(harris_pts) * 1.5)))
        est_gap = float(np.median(dists[:n_short]))
        return est_gap

    def _extract_precise_opencv_l_shape(self, img, patch_center, corner_type, angle1, angle2, expected_gap, debug_canvas=None, roi_r=100):



        h_img, w_img = img.shape[:2]
        cx, cy = int(patch_center[0]), int(patch_center[1])
        x1, y1 = max(0, cx - roi_r), max(0, cy - roi_r)
        x2, y2 = min(w_img - 1, cx + roi_r), min(h_img - 1, cy + roi_r)
        
        roi = img[y1:y2, x1:x2].copy()
        if roi.size == 0: 
            print("  [OpenCV L-Shape] ROI is empty!")
            return patch_center

        # Initialize debug buffers for grid collection
        harris_roi = roi.copy()
        enhanced = roi.copy()
        edges = np.zeros(roi.shape[:2], dtype=np.uint8)
        roi_result = roi.copy()

        def collect_and_return(pt, extra_harris=None):
            # Finalize roi_result with the chosen point
            res = roi.copy()
            lx, ly = int(pt[0] - x1), int(pt[1] - y1)
            cv2.circle(res, (lx, ly), 8, (0, 0, 255), -1)
            cv2.circle(res, (lx, ly), 9, (255, 255, 255), 2)
            cv2.putText(res, f"Pinned {corner_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            
            nonlocal enhanced
            if len(enhanced.shape) == 2: enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            edg_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            patch_list = [roi, edg_bgr, enhanced, harris_roi, res]
            if extra_harris:
                patch_list.extend(extra_harris)
            self._corner_debug_patches[corner_type] = patch_list
            return pt

        # 1. Harris
        try:
            h_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            h_dst = cv2.cornerHarris(h_gray, 3, 3, 0.04)
            h_dst = cv2.dilate(h_dst, None)
            h_thresh = 0.05 * h_dst.max() if h_dst.max() > 0 else 1.0
            y_pts, x_pts = np.where(h_dst > h_thresh)
            for ry, rx in zip(y_pts, x_pts):
                cv2.circle(harris_roi, (rx, ry), 2, (255, 0, 0), -1)
        except: pass

        if debug_canvas is not None:
            cv2.rectangle(debug_canvas, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # 2. Enhanced & Canny
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blur)
        edges = cv2.Canny(enhanced, 50, 150)
        
        # 3. Hough
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=20, maxLineGap=10)
        if lines is None: 
            print("  [OpenCV L-Shape] No Hough Lines found in ROI!")
            return collect_and_return(patch_center)
        
        h_rhos, v_rhos = [], []
        for seg in lines:
            gx1, gy1 = seg[0][0] + x1, seg[0][1] + y1
            gx2, gy2 = seg[0][2] + x1, seg[0][3] + y1
            rt = segment_to_rho_theta(gx1, gy1, gx2, gy2)
            if not rt: continue
            rho, theta = rt
            if abs(cx * np.cos(theta) + cy * np.sin(theta) - rho) > 60: continue
            
            dx, dy = gx2 - gx1, gy2 - gy1
            if abs(dx) > abs(dy):
                if abs(circular_angle_diff(theta, angle1)) < np.radians(30):
                    h_rhos.append((rt, (gy1 + gy2)/2))
            else:
                if abs(circular_angle_diff(theta, angle2)) < np.radians(30):
                    v_rhos.append((rt, (gx1 + gx2)/2))
                
        if not h_rhos or not v_rhos: 
            print("  [OpenCV L-Shape] Missing H or V lines, reverting to CNN pinned pt.")
            if debug_canvas is not None:
                cv2.circle(debug_canvas, (int(patch_center[0]), int(patch_center[1])), 10, (0, 0, 255), -1)
            return collect_and_return(patch_center)
        
        def pick_second_outermost(candidates, key_fn, take_max=False):
            sorted_c = sorted(candidates, key=key_fn, reverse=take_max)
            return sorted_c[1][0] if len(sorted_c) >= 2 else sorted_c[0][0]

        best_h = pick_second_outermost(h_rhos, key_fn=lambda item: item[1], take_max=('B' in corner_type))
        best_v = pick_second_outermost(v_rhos, key_fn=lambda item: item[1], take_max=('R' in corner_type))
        intersect = intersect_lines(best_h[0], best_h[1], best_v[0], best_v[1])
        hough_pt = intersect if intersect else patch_center
        
        pinned_pt, extra_harris = self._snap_to_best_harris_corner(img, hough_pt, patch_center, corner_type,
                                               angle1, angle2, expected_gap, base_roi_r=roi_r)

        if debug_canvas is not None:
            cv2.circle(debug_canvas, (int(pinned_pt[0]), int(pinned_pt[1])), 10, (0, 0, 255), -1) 
            cv2.circle(debug_canvas, (int(pinned_pt[0]), int(pinned_pt[1])), 12, (255, 255, 255), 2)

        return collect_and_return(pinned_pt, extra_harris)



