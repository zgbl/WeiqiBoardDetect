import cv2
import numpy as np

# =====================================================================
# 核心底层几何算法 (V3 透视矫正版)
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


def cluster_lines(lines, rho_threshold=6, theta_threshold_deg=2):
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


def separate_by_angle(lines, v_tol=5, h_tol=2):
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

    kernel_size = 7 
    kernel = np.ones(kernel_size) / kernel_size
    padded = np.concatenate([hist[-kernel_size:], hist, hist[:kernel_size]])
    smoothed = np.convolve(padded, kernel, mode='same')[kernel_size:-kernel_size]

    best_h_score = -1
    peak1_bin = -1
    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        if circular_angle_diff(bin_angle, np.pi/2) < np.radians(10):
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
        if circular_angle_diff(bin_angle, target_peak2) < np.radians(10):
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
    valid_diffs = diffs[diffs > 0.4 * expected_gap]
    typical_gap = np.median(valid_diffs) if len(valid_diffs) >= 5 else expected_gap

    print(f"Typical gap: {typical_gap:.1f} (Expected: {expected_gap:.1f})")

    dp = np.full((n, N, N), np.inf)
    parent = np.full((n, N, N), -1, dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            gap = rhos[j] - rhos[i]
            if gap < 0.65 * typical_gap:
                continue
            dp[1][j][i] = abs(gap - typical_gap) * 1.5

    for k in range(2, n):
        for j in range(k, N):
            for i in range(k - 1, j):
                if dp[k-1][i].min() == np.inf:
                    continue
                gap_ij = rhos[j] - rhos[i]
                if gap_ij < 0.65 * typical_gap or gap_ij > 2.8 * typical_gap:
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


def find_board_corners(intersections, img_shape, shrink_ratio=0.05):
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
    
    cx = sum(c[0] for c in corners) / 4
    cy = sum(c[1] for c in corners) / 4
    
    refined_corners = []
    for x, y in corners:
        nx = x + (cx - x) * shrink_ratio
        ny = y + (cy - y) * shrink_ratio
        x_clamped = max(10, min(w - 10, nx))
        y_clamped = max(10, min(h - 10, ny))
        refined_corners.append((x_clamped, y_clamped))
    
    return refined_corners


# =====================================================================
# V3 新增：透视矫正核心函数
# =====================================================================

def detect_board_contour(img):
    """检测棋盘外轮廓四边形"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    
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


def order_quadrilateral_pts(pts):
    """排序四边形角点：左上、右上、右下、左下"""
    pts = pts.reshape(-1, 2).astype(np.float32)
    
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    
    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmax(diffs)]
    bl = pts[np.argmin(diffs)]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)


def apply_perspective_warp(img, contour):
    """
    V3 核心：透视矫正
    """
    if contour is None:
        return img, None, None
    
    src_pts = order_quadrilateral_pts(contour)
    
    w1 = np.linalg.norm(src_pts[1] - src_pts[0])
    w2 = np.linalg.norm(src_pts[2] - src_pts[3])
    h1 = np.linalg.norm(src_pts[3] - src_pts[0])
    h2 = np.linalg.norm(src_pts[2] - src_pts[1])
    
    max_w = int(max(w1, w2))
    max_h = int(max(h1, h2))
    
    dst_pts = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (max_w, max_h))
    
    return warped, M, src_pts


def inverse_transform_corners(warped_corners, M_inv, src_pts):
    """
    V3: 将矫正后图像的角点逆变换回原图坐标
    """
    original_corners = []
    for pt in warped_corners:
        pt_arr = np.array([[pt[0], pt[1]]], dtype=np.float32)
        orig_pt = cv2.perspectiveTransform(pt_arr, M_inv)[0][0]
        original_corners.append((orig_pt[0], orig_pt[1]))
    return original_corners


# =====================================================================
# OpenCV 检测引擎类 (V3 透视矫正版)
# =====================================================================

class OpenCVDetector:
    def __init__(self, debug_show=True):
        self.debug_show = debug_show
    
    def _find_corners_in_warped(self, warped_img):
        """在矫正后的图像上检测角点（内部方法）"""
        h, w = warped_img.shape[:2]
        
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edge2 = cv2.Canny(blur, 50, 200)

        raw_lines = cv2.HoughLinesP(edge2, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
        
        if raw_lines is None:
            return []

        lines_list = []
        for l in raw_lines:
            x1, y1, x2, y2 = l[0]
            result = segment_to_rho_theta(x1, y1, x2, y2)
            if result is not None:
                lines_list.append(result)

        group1, group2, _ = separate_by_angle(lines_list, v_tol=5, h_tol=2)
        
        if len(group1) == 0 or len(group2) == 0:
            return []

        clustered1 = cluster_lines(group1, rho_threshold=6)
        clustered2 = cluster_lines(group2, rho_threshold=6)

        selected1 = select_n_evenly_spaced(clustered1, n=19)
        selected2 = select_n_evenly_spaced(clustered2, n=19)

        intersections = []
        for rho1, theta1 in selected1:
            for rho2, theta2 in selected2:
                pt = intersect_lines(rho1, theta1, rho2, theta2)
                if pt is not None:
                    x, y = pt
                    if 0 <= x <= w and 0 <= y <= h:
                        intersections.append((x, y))

        corners = find_board_corners(intersections, warped_img.shape, shrink_ratio=0.05)
        return corners
    
    def find_corners(self, img):
        h, w = img.shape[:2]

        print("\n=== OpenCV Module (V3 Perspective Corrected) ===")
        
        # V3 核心：检测轮廓并透视矫正
        contour, contour_area = detect_board_contour(img)
        
        if contour is not None:
            print(f"[V3] Contour detected: area={contour_area}")
            
            warped_img, M, src_pts = apply_perspective_warp(img, contour)
            
            if self.debug_show:
                contour_img = img.copy()
                cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)
                cv2.imshow("[OpenCV V3] Detected Contour", contour_img)
                cv2.imshow("[OpenCV V3] Warped Image", warped_img)
            
            # 在矫正后的图像上检测角点
            warped_corners = self._find_corners_in_warped(warped_img)
            print(f"[V3] Warped corners found: {len(warped_corners)}")
            
            if len(warped_corners) == 4 and M is not None:
                # 逆变换回原图坐标
                M_inv = cv2.invert(M)[1]
                original_corners = inverse_transform_corners(warped_corners, M_inv, src_pts)
                
                print(f"[V3] Original corners after inverse transform:")
                for cx, cy in original_corners:
                    print(f"  Corner at ({int(cx)}, {int(cy)})")
                
                # 边界钳制
                final_corners = []
                for x, y in original_corners:
                    x_clamped = max(10, min(w - 10, x))
                    y_clamped = max(10, min(h - 10, y))
                    final_corners.append((x_clamped, y_clamped))
                
                if self.debug_show:
                    result_img = img.copy()
                    for i, (x, y) in enumerate(final_corners):
                        cv2.circle(result_img, (int(x), int(y)), 15, (0, 0, 255), -1)
                        cv2.putText(result_img, f"C{i+1}", (int(x)-30, int(y)-30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("[OpenCV V3] Final Corners", result_img)
                
                print("=============================================\n")
                return final_corners
        
        # 降级：无透视矫正，用原方法
        print("[V3] Contour detection failed, falling back to direct Hough")
        return self._find_corners_in_warped(img)