import cv2
import numpy as np

# =====================================================================
# 核心底层几何算法 (100% 完整保留 CornerDetect3-EmptyBoardFine.py 逻辑)
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

def angle_diff(a1, a2):
    d = abs(a1 - a2) % np.pi
    return min(d, np.pi - d)

def line_to_params(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    c = -rho
    return a, b, c

def intersect_lines(rho1, theta1, rho2, theta2):
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b2 * rho1 - b1 * rho2) / det
    y = (a1 * rho2 - a2 * rho1) / det
    return (x, y)

def cluster_lines(lines, rho_threshold=20):
    if len(lines) == 0:
        return []

    lines = sorted(lines, key=lambda l: l[0])
    clusters = []
    current_cluster = [lines[0]]

    for i in range(1, len(lines)):
        rho_cur, theta_cur = lines[i]
        rho_prev, theta_prev = current_cluster[-1]

        if abs(rho_cur - rho_prev) < rho_threshold:
            current_cluster.append(lines[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [lines[i]]

    clusters.append(current_cluster)

    result = []
    for cluster in clusters:
        avg_rho = np.mean([l[0] for l in cluster])
        avg_theta = np.mean([l[1] for l in cluster])
        result.append((avg_rho, avg_theta))

    return result

def normalize_line(rho, theta):
    if rho < 0:
        rho = -rho
        theta = theta + np.pi
    theta = theta % (2 * np.pi)
    return rho, theta

def circular_angle_diff(a1, a2, period=np.pi):
    d = abs(a1 - a2) % period
    return min(d, period - d)

def separate_by_angle(lines, angle_tolerance_deg=2 ):
    if len(lines) == 0:
        return [], []

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

    peak1_bin = np.argmax(smoothed)
    peak1_angle = (peak1_bin + 0.5) * bin_size 

    suppressed = smoothed.copy()
    suppress_range = 30 
    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        if circular_angle_diff(bin_angle, peak1_angle) < np.radians(suppress_range):
            suppressed[i] = 0

    peak2_bin = np.argmax(suppressed)
    peak2_angle = (peak2_bin + 0.5) * bin_size

    if suppressed[peak2_bin] == 0:
        print(f"WARNING: Only one dominant angle found (~{np.degrees(peak1_angle):.1f}°)")
        return normalized, []

    print(f"Dominant angles: peak1={np.degrees(peak1_angle):.1f}°, peak2={np.degrees(peak2_angle):.1f}°")
    angle_between = circular_angle_diff(peak1_angle, peak2_angle)
    print(f"Angle between peaks: {np.degrees(angle_between):.1f}° (expect ~90° for Go board)")

    tolerance = np.radians(angle_tolerance_deg)
    group1 = []
    group2 = []
    ignored = 0
    for i, (rho, theta) in enumerate(normalized):
        mt = mapped_thetas[i]
        d1 = circular_angle_diff(mt, peak1_angle)
        d2 = circular_angle_diff(mt, peak2_angle)
        if d1 < d2 and d1 < tolerance:
            group1.append((rho, theta))
        elif d2 <= d1 and d2 < tolerance:
            group2.append((rho, theta))
        else:
            ignored += 1

    print(f"Group assignment: G1={len(group1)}, G2={len(group2)}, Ignored noise={ignored}")
    return group1, group2


def select_n_evenly_spaced(lines, n=19):
    if len(lines) <= n:
        return lines

    rhos = np.array([l[0] for l in lines])
    N = len(rhos)
    
    diffs = np.diff(rhos)
    valid_diffs = diffs[diffs > 5]  
    if len(valid_diffs) == 0:
        return lines[:n]
    median_gap = np.median(valid_diffs)

    dp = np.full((n, N, N), np.inf)
    parent = np.full((n, N, N), -1, dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            gap = rhos[j] - rhos[i]
            dp[1][j][i] = abs(gap - median_gap) * 0.5

    for k in range(2, n):
        for j in range(k, N):
            for i in range(k - 1, j):
                gap_ij = rhos[j] - rhos[i]
                
                if gap_ij > 3 * median_gap or gap_ij < 0.2 * median_gap:
                    continue
                    
                best_cost = np.inf
                best_p = -1
                
                for p in range(k - 2, i):
                    if dp[k-1][i][p] == np.inf:
                        continue
                        
                    gap_pi = rhos[i] - rhos[p]
                    cost = abs(gap_ij - gap_pi) + abs(gap_ij - median_gap) * 0.3
                    
                    total_cost = dp[k-1][i][p] + cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_p = p
                        
                if best_p != -1:
                    dp[k][j][i] = best_cost
                    parent[k][j][i] = best_p

    best_cost = np.inf
    best_j, best_i = -1, -1
    for i in range(n - 2, N):
        for j in range(i + 1, N):
            if dp[n-1][j][i] < best_cost:
                best_cost = dp[n-1][j][i]
                best_j, best_i = j, i

    if best_i == -1 or best_j == -1:
        print("WARNING: DP failed to find 19 consistent lines. Falling back.")
        return [lines[i] for i in np.linspace(0, N-1, n, dtype=int)]

    path = [best_j, best_i]
    curr_k = n - 1
    curr_j = best_j
    curr_i = best_i
    
    while curr_k > 1:
        p = parent[curr_k][curr_j][curr_i]
        path.append(p)
        curr_j = curr_i
        curr_i = p
        curr_k -= 1
        
    path.reverse()
    return [lines[idx] for idx in path]


def find_board_corners(intersections, img_shape):
    if len(intersections) < 4:
        return intersections

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

    return [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]


# =====================================================================
# OpenCV 黑盒引擎类
# =====================================================================

class OpenCVDetector:
    def __init__(self, debug_show=True):
        self.debug_show = debug_show

    def find_corners(self, img):
        """
        100% 重现 CornerDetect3-EmptyBoardFine.py 的逻辑
        完全去除了坑人的差补逻辑！用纯净版 DP 和 Canny(50, 200)。
        """
        display = img.copy()
        h, w = img.shape[:2]

        print("\n=== OpenCV Module (CornerDetect3-EmptyBoardFine Engine) ===")
        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)

        # 【重点】原味的 Canny 和 Hough 参数！
        edge2 = cv2.Canny(blur, 50, 200)
        if self.debug_show:
            cv2.imshow("[OpenCV Engine] Edge2", edge2)

        raw_lines = cv2.HoughLinesP(edge2, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
        
        if raw_lines is not None:
            gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for line in raw_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(gray_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.debug_show:
                cv2.imshow("[OpenCV Engine] Raw Lines", gray_color)
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

        group1, group2 = separate_by_angle(lines_list)
        
        if len(group1) > 0:
            print(f"Group 1 (angle ~{np.degrees(np.mean([t % np.pi for _, t in group1])):.1f}°): {len(group1)} lines")
        else:
            print(f"Group 1: 0 lines")
            
        if len(group2) > 0:
            print(f"Group 2 (angle ~{np.degrees(np.mean([t % np.pi for _, t in group2])):.1f}°): {len(group2)} lines")
        else:
            print(f"Group 2: 0 lines")

        if len(group1) == 0 or len(group2) == 0:
            print("ERROR: Could not separate lines into two groups!")
            return []

        # 【重点】恢复原版 15 的阈值
        clustered1 = cluster_lines(group1, rho_threshold=15)
        clustered2 = cluster_lines(group2, rho_threshold=15)
        print(f"After clustering: Group1={len(clustered1)}, Group2={len(clustered2)}")

        # 【重点】无污染版纯 DP 查找 19 条线，没有瞎推算！
        selected1 = select_n_evenly_spaced(clustered1, n=19)
        selected2 = select_n_evenly_spaced(clustered2, n=19)
        print(f"Selected: Group1={len(selected1)}, Group2={len(selected2)}")

        # Draw selected lines CLIPPED to board boundary (互相裁剪)
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
        else:
            for rho, theta in selected1:
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
                pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
                cv2.line(line_img, pt1, pt2, (0, 255, 0), 1)
            for rho, theta in selected2:
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
                pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
                cv2.line(line_img, pt1, pt2, (255, 200, 0), 1)

        if self.debug_show:
            cv2.imshow("[OpenCV Engine] Detected Grid Lines", line_img)

        # Find intersections
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

        corners = find_board_corners(intersections, img.shape)
        print(f"Board corners found: {len(corners)}")

        for cx, cy in corners:
            cv2.circle(intersect_img, (int(cx), int(cy)), 12, (0, 0, 255), -1)
            cv2.circle(intersect_img, (int(cx), int(cy)), 14, (255, 255, 255), 2)
            print(f"  Corner at ({int(cx)}, {int(cy)})")

        if self.debug_show:
            cv2.imshow("[OpenCV Engine] OpenCV Find Corners", intersect_img)
        
        print("=============================================\n")
        return corners
