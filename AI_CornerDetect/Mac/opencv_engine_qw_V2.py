import cv2
import numpy as np

# =====================================================================
# 核心底层几何算法 (V2 轮廓辅助版 - 与 V1 相同的部分省略)
# =====================================================================

def segment_to_rho_theta(x1, y1, x2, y2):
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
    d = abs(a1 - a2) % period
    return min(d, period - d)


def normalize_line(rho, theta):
    if rho < 0:
        rho = -rho
        theta = theta + np.pi
    theta = theta % (2 * np.pi)
    return rho, theta


def intersect_lines(rho1, theta1, rho2, theta2):
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b2 * rho1 - b1 * rho2) / det
    y = (a1 * rho2 - a2 * rho1) / det
    return (x, y)


def cluster_lines(lines, rho_threshold=8, theta_threshold_deg=3):
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


# =====================================================================
# V2 新增：轮廓检测辅助函数
# =====================================================================

def detect_board_contour(img):
    """
    V2 新增：检测棋盘外轮廓四边形
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour = None
    best_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000:
            continue
        
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4 and area > best_area:
            best_area = area
            best_contour = approx
    
    return best_contour, best_area


def get_quadrilateral_corners(contour):
    """
    V2 新增：从四边形轮廓提取排序后的角点
    """
    if contour is None or len(contour) < 4:
        return None
    
    pts = contour.reshape(-1, 2).astype(np.float32)
    
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    
    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmax(diffs)]
    bl = pts[np.argmin(diffs)]
    
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def refine_corners_with_contour(hough_corners, contour_corners, img_shape, blend_ratio=0.4):
    """
    V2 新增：融合霍夫检测角点和轮廓角点
    """
    if contour_corners is None or len(contour_corners) < 4:
        return hough_corners
    
    if len(hough_corners) < 4:
        return contour_corners
    
    h, w = img_shape[:2]
    refined = []
    
    for i in range(4):
        hx, hy = hough_corners[i]
        cx, cy = contour_corners[i]
        
        # 加权融合：轮廓角点更可靠，给更高权重
        rx = hx * (1 - blend_ratio) + cx * blend_ratio
        ry = hy * (1 - blend_ratio) + cy * blend_ratio
        
        # 边界钳制
        rx = max(10, min(w - 10, rx))
        ry = max(10, min(h - 10, ry))
        
        refined.append((rx, ry))
    
    return refined


def find_board_corners(intersections, img_shape, contour_corners=None, shrink_ratio=0.08):
    """
    V2 优化：支持轮廓角点融合
    """
    if len(intersections) < 4:
        if contour_corners:
            return contour_corners
        return intersections
    
    h, w = img_shape[:2]
    pts = np.array(intersections, dtype=np.float32)

    hull = cv2.convexHull(pts)
    hull_pts = hull.reshape(-1, 2)

    if len(hull_pts) < 4:
        if contour_corners:
            return contour_corners
        return hull_pts.tolist()

    sums = pts[:, 0] + pts[:, 1]  
    diffs = pts[:, 0] - pts[:, 1] 

    top_left = pts[np.argmin(sums)]       
    bottom_right = pts[np.argmax(sums)]   
    top_right = pts[np.argmax(diffs)]     
    bottom_left = pts[np.argmin(diffs)]   

    hough_corners = [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]
    
    # V2: 如果有轮廓角点，进行融合
    if contour_corners is not None and len(contour_corners) == 4:
        return refine_corners_with_contour(hough_corners, contour_corners, img_shape, blend_ratio=0.4)
    
    # 否则用内缩保护
    cx = sum(c[0] for c in hough_corners) / 4
    cy = sum(c[1] for c in hough_corners) / 4
    
    refined_corners = []
    for x, y in hough_corners:
        nx = x + (cx - x) * shrink_ratio
        ny = y + (cy - y) * shrink_ratio
        x_clamped = max(10, min(w - 10, nx))
        y_clamped = max(10, min(h - 10, ny))
        refined_corners.append((x_clamped, y_clamped))
    
    return refined_corners


# =====================================================================
# OpenCV 检测引擎类 (V2 轮廓辅助版)
# =====================================================================

class OpenCVDetector:
    def __init__(self, debug_show=True):
        self.debug_show = debug_show
    
    def find_corners(self, img):
        h, w = img.shape[:2]

        print("\n=== OpenCV Module (V2 Contour Assisted) ===")
        
        # V2 新增：先检测轮廓角点
        contour, contour_area = detect_board_contour(img)
        contour_corners = None
        
        if contour is not None:
            contour_corners = get_quadrilateral_corners(contour)
            print(f"[V2] Contour detected: area={contour_area}, corners={contour_corners}")
            
            if self.debug_show:
                contour_img = img.copy()
                cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)
                for i, (x, y) in enumerate(contour_corners):
                    cv2.circle(contour_img, (int(x), int(y)), 10, (0, 0, 255), -1)
                cv2.imshow("[OpenCV V2] Detected Contour", contour_img)
        else:
            print("[V2] No valid contour found, falling back to Hough only")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edge2 = cv2.Canny(blur, 50, 200)
        if self.debug_show:
            cv2.imshow("[OpenCV V2] Edge2", edge2)

        raw_lines = cv2.HoughLinesP(edge2, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
        
        if raw_lines is None:
            print("No lines detected!")
            if contour_corners:
                print("[V2] Returning contour corners as fallback")
                return contour_corners
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
            if contour_corners:
                print("[V2] Returning contour corners as fallback")
                return contour_corners
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
            cv2.imshow("[OpenCV V2] Detected Grid Lines", line_img)

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

        # V2: 传入轮廓角点进行融合
        corners = find_board_corners(intersections, img.shape, contour_corners=contour_corners, shrink_ratio=0.08)
        print(f"Board corners found: {len(corners)}")

        for cx, cy in corners:
            cv2.circle(intersect_img, (int(cx), int(cy)), 12, (0, 0, 255), -1)
            cv2.circle(intersect_img, (int(cx), int(cy)), 14, (255, 255, 255), 2)
            print(f"  Corner at ({int(cx)}, {int(cy)})")

        if self.debug_show:
            cv2.imshow("[OpenCV V2] OpenCV Find Corners", intersect_img)
        
        print("=============================================\n")
        return corners