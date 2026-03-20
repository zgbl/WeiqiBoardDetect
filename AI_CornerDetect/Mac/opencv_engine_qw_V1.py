import cv2
import numpy as np

# =====================================================================
# 核心底层几何算法 (V1 保守优化版)
# =====================================================================

def segment_to_rho_theta(x1, y1, x2, y2):
    """将线段转换为 (rho, theta) 极坐标表示"""
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-8:
        return None
    nx = -dy / length
    ny = dx / length
    rho = nx * x1 + ny * y1

    if rho < 0:
        rho = -rho
        nx = -nx
        ny = -ny

    theta = np.arctan2(ny, nx)
    if theta < 0:
        theta += 2 * np.pi

    return rho, theta


def circular_angle_diff(a1, a2, period=np.pi):
    """计算循环角度差"""
    d = abs(a1 - a2) % period
    return min(d, period - d)


def normalize_line(rho, theta):
    """标准化直线参数，确保 rho >= 0"""
    if rho < 0:
        rho = -rho
        theta = theta + np.pi
    theta = theta % (2 * np.pi)
    return rho, theta


def intersect_lines(rho1, theta1, rho2, theta2):
    """计算两条直线的交点"""
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b2 * rho1 - b1 * rho2) / det
    y = (a1 * rho2 - a2 * rho1) / det
    return (x, y)


def cluster_lines(lines, rho_threshold=8, theta_threshold_deg=3):
    """聚类线条，V1 放宽阈值适应透视"""
    if len(lines) == 0:
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
        if final_theta < 0:
            final_theta += 2 * np.pi
        result.append((final_rho, final_theta))

    return sorted(result, key=lambda x: x[0])


def separate_by_angle(lines, v_tol=20, h_tol=15):
    """
    V1 优化：大幅放宽角度容差适应透视畸变
    """
    if len(lines) == 0:
        return [], [], []
    
    normalized = []
    for rho, theta in lines:
        rho, theta = normalize_line(rho, theta)
        normalized.append((rho, theta))

    mapped_thetas = np.array([t % np.pi for _, t in normalized])

    n_bins = 180
    bin_size = np.pi / n_bins 
    hist = np.zeros(n_bins)
    for theta in mapped_thetas:
        bin_idx = int(theta / bin_size) % n_bins
        hist[bin_idx] += 1

    kernel_size = 9 
    kernel = np.ones(kernel_size) / kernel_size
    padded = np.concatenate([hist[-kernel_size:], hist, hist[:kernel_size]])
    smoothed = np.convolve(padded, kernel, mode='same')[kernel_size:-kernel_size]

    # 1. 寻找水平方向峰值 (90 度附近)
    best_h_score = -1
    peak1_bin = -1
    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        if circular_angle_diff(bin_angle, np.pi/2) < np.radians(20):
            if smoothed[i] > best_h_score:
                best_h_score = smoothed[i]
                peak1_bin = i

    if peak1_bin == -1 or best_h_score < 1e-3:
        peak1_bin = np.argmax(smoothed)

    peak1_angle = (peak1_bin + 0.5) * bin_size 

    # 2. 寻找垂直方向峰值 (大幅放宽到 25 度)
    best_v_score = -1
    peak2_bin = -1
    target_peak2 = (peak1_angle + np.pi/2) % np.pi

    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        if circular_angle_diff(bin_angle, target_peak2) < np.radians(25):
            if smoothed[i] > best_v_score:
                best_v_score = smoothed[i]
                peak2_bin = i

    if peak2_bin == -1 or best_v_score < 0.1 * smoothed[peak1_bin]:
        suppressed = smoothed.copy()
        for i in range(n_bins):
            if circular_angle_diff((i+0.5)*bin_size, peak1_angle) < np.radians(45):
                suppressed[i] = 0
        peak2_bin = np.argmax(suppressed)

    peak2_angle = (peak2_bin + 0.5) * bin_size
    angle_diff_val = np.degrees(circular_angle_diff(peak1_angle, peak2_angle))

    print(f"Prioritize Horizontal analysis: Peak1={np.degrees(peak1_angle):.1f}°, Peak2={np.degrees(peak2_angle):.1f}°, Separation={angle_diff_val:.1f}°")

    # V1 优化：根据分离度动态调整容差
    if angle_diff_val < 80:
        tol1 = np.radians(25)
        tol2 = np.radians(25)
        print(f"[WARN] Perspective distortion detected! Separation={angle_diff_val:.1f}°")
    else:
        tol1 = np.radians(h_tol)
        tol2 = np.radians(v_tol)

    def get_tol(angle):
        diff_to_90 = circular_angle_diff(angle, np.pi/2)
        if diff_to_90 < np.radians(45):
            return tol1, "Horizontal"
        else:
            return tol2, "Vertical"

    tol1, type1 = get_tol(peak1_angle)
    tol2, type2 = get_tol(peak2_angle)

    group1 = []
    group2 = []
    ignored = []
    for i, (rho, theta) in enumerate(normalized):
        mt = mapped_thetas[i]
        d1 = circular_angle_diff(mt, peak1_angle)
        d2 = circular_angle_diff(mt, peak2_angle)
        
        if d1 < tol1 and (d1 <= d2):
            group1.append((rho, theta))
        elif d2 < tol2:
            group2.append((rho, theta))
        else:
            ignored.append((rho, theta))

    print(f"Group assignment: G1({type1})={len(group1)}, G2({type2})={len(group2)}, Ignored={len(ignored)}")
    return group1, group2, ignored


def select_n_evenly_spaced(lines, n=19):
    """DP 筛选 19 条线，V1 放宽间距约束"""
    if len(lines) <= n:
        return lines
    rhos = np.array([l[0] for l in lines])
    N = len(rhos)

    span = rhos[-1] - rhos[0]
    expected_gap = span / (n - 1)

    diffs = np.diff(rhos)
    valid_diffs = diffs[diffs > 0.3 * expected_gap]
    typical_gap = np.median(valid_diffs) if len(valid_diffs) >= 5 else expected_gap

    print(f"Typical gap: {typical_gap:.1f} (Expected: {expected_gap:.1f})")

    # V1 优化：透视情况下放宽约束
    perspective_factor = typical_gap / expected_gap if expected_gap > 0 else 1
    if perspective_factor < 0.6:
        min_gap_ratio = 0.3
        max_gap_ratio = 5.0
        print(f"[WARN] Perspective distortion! Factor: {perspective_factor:.2f}")
    else:
        min_gap_ratio = 0.65
        max_gap_ratio = 2.8

    dp = np.full((n, N, N), np.inf)
    parent = np.full((n, N, N), -1, dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            gap = rhos[j] - rhos[i]
            if gap < min_gap_ratio * typical_gap:
                continue
            dp[1][j][i] = abs(gap - typical_gap) * 1.5

    for k in range(2, n):
        for j in range(k, N):
            for i in range(k - 1, j):
                if dp[k-1][i].min() == np.inf:
                    continue
                gap_ij = rhos[j] - rhos[i]
                if gap_ij < min_gap_ratio * typical_gap or gap_ij > max_gap_ratio * typical_gap:
                    continue
                    
                best_cost = np.inf
                best_p = -1
                for p in range(k - 2, i):
                    if dp[k-1][i][p] == np.inf:
                        continue
                    gap_pi = rhos[i] - rhos[p]
                    cost = dp[k-1][i][p] + abs(gap_ij - gap_pi) + abs(gap_ij - typical_gap) * 0.5
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
            if dp[n-1][j][i] < final_best_cost:
                final_best_cost = dp[n-1][j][i]
                best_j, best_i = j, i

    if best_i == -1:
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


def find_board_corners(intersections, img_shape, shrink_ratio=0.08):
    """
    V1 优化：添加角点内缩和边界钳制
    """
    if len(intersections) < 4:
        return intersections
    
    h, w = img_shape[:2]
    pts = np.array(intersections, dtype=np.float32)

    hull = cv2.convexHull(pts)
    hull_pts = hull.reshape(-1, 2)

    if len(hull_pts) < 4:
        return hull_pts.tolist()

    sums = pts[:, 0] + pts[:, 1]  
    diffs = pts[:, 0] - pts[:, 1] 

    top_left = pts[np.argmin(sums)]       
    bottom_right = pts[np.argmax(sums)]   
    top_right = pts[np.argmax(diffs)]     
    bottom_left = pts[np.argmin(diffs)]   

    corners = [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]
    
    # V1 优化 1: 计算中心并向内收缩
    cx = sum(c[0] for c in corners) / 4
    cy = sum(c[1] for c in corners) / 4
    
    refined_corners = []
    for x, y in corners:
        nx = x + (cx - x) * shrink_ratio
        ny = y + (cy - y) * shrink_ratio
        refined_corners.append((nx, ny))
    
    # V1 优化 2: 边界钳制
    final_corners = []
    for x, y in refined_corners:
        x_clamped = max(10, min(w - 10, x))
        y_clamped = max(10, min(h - 10, y))
        final_corners.append((x_clamped, y_clamped))
    
    return final_corners


# =====================================================================
# OpenCV 检测引擎类 (V1 保守优化版)
# =====================================================================

class OpenCVDetector:
    def __init__(self, debug_show=True):
        self.debug_show = debug_show
    
    def find_corners(self, img):
        display = img.copy()
        h, w = img.shape[:2]

        print("\n=== OpenCV Module (V1 Conservative Optimized) ===")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)

        edge2 = cv2.Canny(blur, 50, 200)
        if self.debug_show:
            cv2.imshow("[OpenCV V1] Edge2", edge2)

        raw_lines = cv2.HoughLinesP(edge2, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
        
        if raw_lines is not None:
            gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for line in raw_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(gray_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.debug_show:
                cv2.imshow("[OpenCV V1] Raw Lines", gray_color)
        else:
            print("No lines detected!")
            return []

        print(f"Raw Hough lines detected: {len(raw_lines)}")

        lines_list = []
        for l in raw_lines:
            x1, y1, x2, y2 = l[0]
            result = segment_to_rho_theta(x1, y1, x2, y2)
            if result is not None:
                lines_list.append(result)
        print(f"# of lines converted to (rho,theta): {len(lines_list)}")

        group1, group2, ignored_lines = separate_by_angle(lines_list, v_tol=20, h_tol=15)
        
        if len(group1) > 0:
            avg_a = np.degrees(np.mean([t % np.pi for _, t in group1]))
            print(f"Group 1 (angle ~{avg_a:.1f}°): {len(group1)} lines")
        else:
            print(f"Group 1: 0 lines")
            
        if len(group2) > 0:
            avg_a = np.degrees(np.mean([t % np.pi for _, t in group2]))
            print(f"Group 2 (angle ~{avg_a:.1f}°): {len(group2)} lines")
        else:
            print(f"Group 2: 0 lines")

        if len(group1) == 0 or len(group2) == 0:
            print("ERROR: Could not separate lines into two groups!")
            return []

        clustered1 = cluster_lines(group1, rho_threshold=8)
        clustered2 = cluster_lines(group2, rho_threshold=8)
        print(f"After clustering: Group1={len(clustered1)}, Group2={len(clustered2)}")

        selected1 = select_n_evenly_spaced(clustered1, n=19)
        selected2 = select_n_evenly_spaced(clustered2, n=19)
        print(f"Selected: Group1={len(selected1)}, Group2={len(selected2)}")

        line_img = img.copy()

        if len(selected1) >= 2 and len(selected2) >= 2:
            bound2_first = selected2[0]  
            bound2_last = selected2[-1]   
            bound1_first = selected1[0]
            bound1_last = selected1[-1]

            for rho, theta in selected1:
                pt_a = intersect_lines(rho, theta, bound2_first[0], bound2_first[1])
                pt_b = intersect_lines(rho, theta, bound2_last[0], bound2_last[1])
                if pt_a is not None and pt_b is not None:
                    cv2.line(line_img, (int(pt_a[0]), int(pt_a[1])),
                             (int(pt_b[0]), int(pt_b[1])), (0, 255, 0), 1)

            for rho, theta in selected2:
                pt_a = intersect_lines(rho, theta, bound1_first[0], bound1_first[1])
                pt_b = intersect_lines(rho, theta, bound1_last[0], bound1_last[1])
                if pt_a is not None and pt_b is not None:
                    cv2.line(line_img, (int(pt_a[0]), int(pt_a[1])),
                             (int(pt_b[0]), int(pt_b[1])), (255, 200, 0), 1)

        if self.debug_show:
            cv2.imshow("[OpenCV V1] Detected Grid Lines", line_img)

        intersections = []
        for rho1, theta1 in selected1:
            for rho2, theta2 in selected2:
                pt = intersect_lines(rho1, theta1, rho2, theta2)
                if pt is not None:
                    x, y = pt
                    margin = 50
                    if -margin <= x <= w + margin and -margin <= y <= h + margin:
                        intersections.append((x, y))

        print(f"Total grid intersections found: {len(intersections)}")

        intersect_img = line_img.copy()
        for x, y in intersections:
            cv2.circle(intersect_img, (int(x), int(y)), 3, (0, 200, 0), -1)

        corners = find_board_corners(intersections, img.shape, shrink_ratio=0.08)
        print(f"Board corners found: {len(corners)}")

        for cx, cy in corners:
            cv2.circle(intersect_img, (int(cx), int(cy)), 12, (0, 0, 255), -1)
            cv2.circle(intersect_img, (int(cx), int(cy)), 14, (255, 255, 255), 2)
            print(f"  Corner at ({int(cx)}, {int(cy)})")

        if self.debug_show:
            cv2.imshow("[OpenCV V1] OpenCV Find Corners", intersect_img)
        
        print("=============================================\n")
        return corners