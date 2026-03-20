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

def cluster_lines(lines, rho_threshold=6, theta_threshold_deg=2):
    """
    改进的聚类算法：使用严格的质心邻域检查，防止“链式反应”导致不相关的线被焊在一起。
    同时加入角度约束，只有角度和距离都接近的线段才属于同一条网格线。
    """
    if len(lines) == 0:
        return []

    # 按 rho 排序有利于加速查找
    lines = sorted(lines, key=lambda l: l[0])
    clusters = []

    for rho, theta in lines:
        found_cluster = False
        for cluster in clusters:
            # 计算当前簇的平均值 (质心)
            avg_rho = np.mean([item[0] for item in cluster])
            
            # 角度平均需要处理绕回 (使用向量平均)
            sum_sin = np.sum([np.sin(item[1]) for item in cluster])
            sum_cos = np.sum([np.cos(item[1]) for item in cluster])
            avg_theta = np.arctan2(sum_sin, sum_cos)

            # 严格检查新线段是否靠近质心
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

    return sorted(result, key=lambda x: x[0])

def normalize_line(rho, theta):
    if rho < 0:
        rho = -rho
        theta = theta + np.pi
    theta = theta % (2 * np.pi)
    return rho, theta

def circular_angle_diff(a1, a2, period=np.pi):
    d = abs(a1 - a2) % period
    return min(d, period - d)

def separate_by_angle(lines, v_tol=5, h_tol=2):
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

    # 1. 【水平优先】严格锁定水平方向 (90 +/- 5 度) 寻找基准峰值
    best_h_score = -1
    peak1_bin = -1
    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        #print("bin_angle is:", bin_angle)
        # 允许正负 5 度的摆动
        if circular_angle_diff(bin_angle, np.pi/2) < np.radians(5):   # 15 改成的5？ >> 不是这里
            if smoothed[i] > best_h_score:
                best_h_score = smoothed[i]
                peak1_bin = i

    # 如果水平方向没找到明显的线，再降级找全局最高峰
    if peak1_bin == -1 or best_h_score < 1e-3:
        peak1_bin = np.argmax(smoothed)

    peak1_angle = (peak1_bin + 0.5) * bin_size 

    # 2. 【强化正交约束】强制要求第二个峰值必须在 Peak 1 的垂直方向正负 15 度范围内
    best_v_score = -1
    peak2_bin = -1
    target_peak2 = (peak1_angle + np.pi/2) % np.pi
    
    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        if circular_angle_diff(bin_angle, target_peak2) < np.radians(15):
            if smoothed[i] > best_v_score:
                best_v_score = smoothed[i]
                peak2_bin = i

    # 如果在正交正负 15 度范围内完全没找到峰值（可能由于视角太斜或线太模糊）
    if peak2_bin == -1:
        print("ERROR: No valid orthogonal peak found in 75-105 range! Discarding result.")
        return [], [], []
    
    # 强制丢弃信号太弱的“伪峰”（确保垂直方向确实有线存活）
    if best_v_score < 0.05 * smoothed[peak1_bin]:
        print(f"ERROR: Orthogonal peak too weak ({best_v_score:.2f}). Giving up.")
        return [], [], []

    peak2_angle = (peak2_bin + 0.5) * bin_size
    angle_diff_val = np.degrees(circular_angle_diff(peak1_angle, peak2_angle))
    
    print(f"Prioritize Horizontal analysis: Peak1={np.degrees(peak1_angle):.1f}°, Peak2={np.degrees(peak2_angle):.1f}°, Separation={angle_diff_val:.1f}°")

    def get_tol(angle):
        diff_to_90 = circular_angle_diff(angle, np.pi/2)
        if diff_to_90 < np.radians(45):
            return np.radians(h_tol), "Horizontal"
        else:
            return np.radians(v_tol), "Vertical"

    tol1, type1 = get_tol(peak1_angle)
    tol2, type2 = get_tol(peak2_angle)

    group1 = []
    group2 = []
    ignored = []
    
    print(f"DEBUG: peak1_angle={np.degrees(peak1_angle):.1f}°, target_peak2={np.degrees(target_peak2):.1f}°")
    
    for i, (rho, theta) in enumerate(normalized):
        mt = mapped_thetas[i]
        d1 = circular_angle_diff(mt, peak1_angle)
        d2 = circular_angle_diff(mt, peak2_angle)
        
        # 严格匹配
        if d1 < tol1 and (d1 <= d2):
            group1.append((rho, theta))
        elif d2 < tol2:
            # 计算到理想垂直方向的差距
            diff_to_target = circular_angle_diff(mt, target_peak2)
            if diff_to_target < np.radians(15):
                group2.append((rho, theta))
            else:
                # 如果这个值被本该过滤掉却还是进来了，在这里打印
                if np.degrees(mt) > 20: 
                    # print(f"  [Filtered] mt={np.degrees(mt):.1f}° diff={np.degrees(diff_to_target):.1f}° > 15°")
                    pass
                ignored.append((rho, theta))
        else:
            ignored.append((rho, theta))

    def get_vector_avg_deg(group):
        if not group: return 0.0
        # 使用向量平均法处理角度绕回问题 (2 * theta 是因为周期是 pi)
        sum_sin = np.sum([np.sin(2 * t) for _, t in group])
        sum_cos = np.sum([np.cos(2 * t) for _, t in group])
        avg_theta = 0.5 * np.arctan2(sum_sin, sum_cos)
        return np.degrees(avg_theta % np.pi)

    g2_avg = get_vector_avg_deg(group2)
    print(f"DEBUG Filter Result: G2 final size={len(group2)}, vector_avg={g2_avg:.1f}°")
    print(f"Group assignment: G1({type1})={len(group1)}, G2({type2})={len(group2)}, Ignored={len(ignored)}")
    return group1, group2, ignored, peak1_angle, peak2_angle


def select_n_evenly_spaced(lines, n=19, group_peak_angle=0):
    """
    改进的 DP 筛选算法：强制分散，杜绝扎堆。
    【优化】大幅调高间距门槛，防止被局部高密度干扰线（如木纹、纸巾盒）带偏。
    """
    if len(lines) <= n:
        return lines

    # 1. 预过滤：解决“密集垃圾线”问题
    # 围棋盘 19 条线，间距应该比较大。如果间距小于预期的 70%，基本可以断定是干扰。
    lines = sorted(lines, key=lambda l: l[0])
    rhos = np.array([l[0] for l in lines])
    span = rhos[-1] - rhos[0]
    expected_gap = span / (n - 1)
    
    cleaned_lines = []
    print("number is line is:", len(lines))
    #print("在这个之前，棋盘纵线已经丢失了")
    
    if len(lines) > 0:
        cleaned_lines.append(lines[0])
        for i in range(1, len(lines)):
            curr_rho, curr_theta = lines[i]
            prev_rho, prev_theta = cleaned_lines[-1]
            
            # 【提高门槛】间距小于预期的 40% 说明太密了，二选一
            if (curr_rho - prev_rho) < 0.1 * expected_gap:
                dist_curr = circular_angle_diff(curr_theta, group_peak_angle)
                dist_prev = circular_angle_diff(prev_theta, group_peak_angle)
                # 角度更准的留下
                if dist_curr < dist_prev:
                    cleaned_lines[-1] = lines[i]
                else:
                    continue 
            else:
                cleaned_lines.append(lines[i])
   
    print(f"Pre-filter: {len(lines)} -> {len(cleaned_lines)} (threshold: {0.4 * expected_gap:.1f}px)")
    lines = cleaned_lines
    if len(lines) <= n: return lines

    # 2. 重新估计 Typical Gap
    rhos = np.array([l[0] for l in lines])
    N = len(rhos)
    span = rhos[-1] - rhos[0]
    expected_gap = span / (n - 1)
    
    # 【核心修正】计算统计中位间距时，必须排除掉那些依然存在的“局部高密度”噪声
    diffs = np.diff(rhos)
    #valid_diffs = diffs[diffs > 0.75 * expected_gap] # 过滤掉所有过小的间距  # 0.75 可能门槛太高了，丢了线。 3.18.2026 TXY
    valid_diffs = diffs[diffs > 0.5 * expected_gap]
    typical_gap = np.median(valid_diffs) if len(valid_diffs) >= 5 else expected_gap
    
    # 安全锁：防止由于某些极端透视导致 typical_gap 缩水太厉害
    typical_gap = max(typical_gap, 0.8 * expected_gap)
    
    print(f"Refined spacing: Typical={typical_gap:.1f}, Expected={expected_gap:.1f}")

    dp = np.full((n, N, N), np.inf)
    parent = np.full((n, N, N), -1, dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            gap = rhos[j] - rhos[i]
            # [硬约束] 严禁选取比平均间距小太多的组合
            if gap < 0.65 * typical_gap: continue
            dp[1][j][i] = abs(gap - typical_gap) * 1.5

    for k in range(2, n):
        for j in range(k, N):
            for i in range(k - 1, j):
                if dp[k-1][i].min() == np.inf: continue
                gap_ij = rhos[j] - rhos[i]
                if gap_ij < 0.65 * typical_gap or gap_ij > 2.8 * typical_gap: continue
                    
                best_cost = np.inf
                best_p = -1
                for p in range(k - 2, i):
                    if dp[k-1][i][p] == np.inf: continue
                    gap_pi = rhos[i] - rhos[p]
                    # 惩罚步长不连续和偏离全局步长
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
    
def filter_angle_outliers(clustered_lines, reference_angle, max_dev_deg=5):
    """踢掉角度偏离 reference_angle 超过 max_dev_deg 的线。
    reference_angle 应该用 separate_by_angle 返回的峰值角度，
    不要用 np.median —— 角度在 0°/180° 边界会算错。
    """
    if len(clustered_lines) < 5:
        return clustered_lines
    ref = reference_angle % np.pi
    filtered = []
    for rho, theta in clustered_lines:
        dev = circular_angle_diff(theta % np.pi, ref)
        if dev < np.radians(max_dev_deg):
            filtered.append((rho, theta))
    print(f"Angle filter: {len(clustered_lines)} -> {len(filtered)} (ref={np.degrees(ref):.1f}°, ±{max_dev_deg}°)")
    if len(filtered) < 19:
        print(f"  警告: 只剩 {len(filtered)} 条线，不够19")
    return filtered


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

        # 【最终优化】使用正交引导后，容差可以收缩到合理范围 (18度纵线，10度横线)
        # 修改：让 it 返回波峰角度，供后续精挑细选
        group1, group2, ignored_lines, p1_angle, p2_angle = separate_by_angle(lines_list, v_tol=18, h_tol=10)
        print("p1_angle", p1_angle, "p2_angle", p2_angle)
        
        def get_group_angle(group):
            if not group: return 0.0
            sum_sin = np.sum([np.sin(2 * t) for _, t in group])
            sum_cos = np.sum([np.cos(2 * t) for _, t in group])
            return np.degrees(0.5 * np.arctan2(sum_sin, sum_cos) % np.pi)

        if len(group1) > 0:
            avg_a = get_group_angle(group1)
            print(f"Group 1 (angle ~{avg_a:.1f}°): {len(group1)} lines")
        else:
            print(f"Group 1: 0 lines")
            
        if len(group2) > 0:
            avg_a = get_group_angle(group2)
            print(f"Group 2 (angle ~{avg_a:.1f}°): {len(group2)} lines")
        else:
            print(f"Group 2: 0 lines")

        if len(group1) == 0 or len(group2) == 0:
            print("ERROR: Could not separate lines into two groups!")
            return []

        # 【优化】降低 rho_threshold 阈值（从 15 降到 6），防止由于透视或角度误差导致的 grid line 合并。
        # 让 DP 算法去筛选 19 条，而不是在这里就把线“粘”在一起。
        clustered1 = cluster_lines(group1, rho_threshold=6)
        clustered2 = cluster_lines(group2, rho_threshold=6)
        print(f"After clustering, 在clusterLine之后，根数: Group1={len(clustered1)}, Group2={len(clustered2)}")

        # 角度过滤：横线用紧容差(±3°)，竖线用宽容差(±15°) — 透视会让竖线扇形展开
        h_dev = 3; v_dev = 15
        p1_is_h = circular_angle_diff(p1_angle, np.pi/2) < np.radians(45)
        clustered1 = filter_angle_outliers(clustered1, reference_angle=p1_angle, max_dev_deg=h_dev if p1_is_h else v_dev)
        clustered2 = filter_angle_outliers(clustered2, reference_angle=p2_angle, max_dev_deg=v_dev if p1_is_h else h_dev)
        print("filter_angle_outliers 之后 clustered1 的根数 is now:", len(clustered1), "clustered2 的根数is now:", len(clustered2))


        # 【重点】调用时带上波峰角度，进行“浅交叉”预过滤
        selected1 = select_n_evenly_spaced(clustered1, n=19, group_peak_angle=p1_angle)
        selected2 = select_n_evenly_spaced(clustered2, n=19, group_peak_angle=p2_angle)
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

        # 【新增调试窗口】分类结果：候选线(绿) vs 被踢出的线(红) vs 最终选中的线(黄)
        if self.debug_show:
            debug_img = img.copy()
            def draw_inf_line(canvas, rho, theta, color, thickness):
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (int(x0 + 3000 * (-b)), int(y0 + 3000 * a))
                pt2 = (int(x0 - 3000 * (-b)), int(y0 - 3000 * a))
                cv2.line(canvas, pt1, pt2, color, thickness)

            # 红色: 被 separate_by_angle 过滤掉的“噪声”
            for r, t in ignored_lines:
                draw_inf_line(debug_img, r, t, (0, 0, 255), 1)
            # 绿色: 聚类后的有效候选网格线
            for r, t in clustered1 + clustered2:
                draw_inf_line(debug_img, r, t, (0, 255, 0), 1)
            # 黄色: DP 算法最终挑选出的线
            for r, t in selected1 + selected2:
                draw_inf_line(debug_img, r, t, (0, 255, 255), 2)
                
            cv2.imshow("[Debug] All(Red) -> Candidates(Green) -> Selected(Yellow)", debug_img)

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
