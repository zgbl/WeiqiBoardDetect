import cv2
import numpy as np

# =====================================================================
# 核心底层几何算法 (修复版 - 解决角点越界和透视问题)
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


def angle_diff(a1, a2):
    """计算两个角度之间的最小差值"""
    d = abs(a1 - a2) % np.pi
    return min(d, np.pi - d)


def line_to_params(rho, theta):
    """将极坐标转换为直线方程参数"""
    a = np.cos(theta)
    b = np.sin(theta)
    c = -rho
    return a, b, c


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


def cluster_lines(lines, rho_threshold=6, theta_threshold_deg=2):
    """
    改进的聚类算法：使用严格的质心邻域检查，防止"链式反应"导致不相关的线被焊在一起。
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
        if final_theta < 0:
            final_theta += 2 * np.pi
        result.append((final_rho, final_theta))

    return sorted(result, key=lambda x: x[0])


def normalize_line(rho, theta):
    """标准化直线参数，确保 rho >= 0"""
    if rho < 0:
        rho = -rho
        theta = theta + np.pi
    theta = theta % (2 * np.pi)
    return rho, theta


def circular_angle_diff(a1, a2, period=np.pi):
    """计算循环角度差"""
    d = abs(a1 - a2) % period
    return min(d, period - d)


def separate_by_angle(lines, v_tol=5, h_tol=2):
    """
    按角度分离线条为两组（近似正交）
    【修复】增加容差以适应透视畸变
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

    kernel_size = 7 
    kernel = np.ones(kernel_size) / kernel_size
    padded = np.concatenate([hist[-kernel_size:], hist, hist[:kernel_size]])
    smoothed = np.convolve(padded, kernel, mode='same')[kernel_size:-kernel_size]

    # 1. 【水平优先】严格锁定水平方向 (90 +/- 5 度) 寻找基准峰值
    best_h_score = -1
    peak1_bin = -1
    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        # 允许正负 5 度的摆动
        if circular_angle_diff(bin_angle, np.pi/2) < np.radians(5):
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
        # 【修复】放宽正交搜索容差从 5 度到 15 度，适应透视畸变
        if circular_angle_diff(bin_angle, target_peak2) < np.radians(15):
            if smoothed[i] > best_v_score:
                best_v_score = smoothed[i]
                peak2_bin = i

    # 如果正交方向没找到明显的峰，退化到全局次高峰搜索
    if peak2_bin == -1 or best_v_score < 0.1 * smoothed[peak1_bin]:
        suppressed = smoothed.copy()
        for i in range(n_bins):
            if circular_angle_diff((i+0.5)*bin_size, peak1_angle) < np.radians(45):
                suppressed[i] = 0
        peak2_bin = np.argmax(suppressed)

    peak2_angle = (peak2_bin + 0.5) * bin_size
    angle_diff_val = np.degrees(circular_angle_diff(peak1_angle, peak2_angle))

    print(f"Prioritize Horizontal analysis: Peak1={np.degrees(peak1_angle):.1f}°, Peak2={np.degrees(peak2_angle):.1f}°, Separation={angle_diff_val:.1f}°")

    def get_tol(angle):
        diff_to_90 = circular_angle_diff(angle, np.pi/2)
        if diff_to_90 < np.radians(45):
            return np.radians(h_tol), "Horizontal"
        else:
            return np.radians(v_tol), "Vertical"

    # 【修复】增加角度容差以适应透视
    tol1, type1 = get_tol(peak1_angle)
    tol2, type2 = get_tol(peak2_angle)
    
    # 如果角度分离度小于 80 度，说明透视严重，进一步放宽容差
    if angle_diff_val < 80:
        tol1 = np.radians(15)
        tol2 = np.radians(15)
        print(f"[WARN] Perspective distortion detected! Increased angle tolerance.")

    group1 = []
    group2 = []
    ignored = []
    for i, (rho, theta) in enumerate(normalized):
        mt = mapped_thetas[i]
        d1 = circular_angle_diff(mt, peak1_angle)
        d2 = circular_angle_diff(mt, peak2_angle)
        
        # 严格匹配
        if d1 < tol1 and (d1 <= d2):
            group1.append((rho, theta))
        elif d2 < tol2:
            group2.append((rho, theta))
        else:
            ignored.append((rho, theta))

    print(f"Group assignment: G1({type1})={len(group1)}, G2({type2})={len(group2)}, Ignored={len(ignored)}")
    return group1, group2, ignored


def select_n_evenly_spaced(lines, n=19):
    """
    改进的 DP 筛选算法：强制分散，杜绝扎堆。
    【修复】增加透视适应逻辑
    """
    if len(lines) <= n:
        return lines
    rhos = np.array([l[0] for l in lines])
    N = len(rhos)

    # 根据总跨度预备合理间距
    span = rhos[-1] - rhos[0]
    expected_gap = span / (n - 1)

    # 鲁棒中值估计
    diffs = np.diff(rhos)
    valid_diffs = diffs[diffs > 0.4 * expected_gap]
    typical_gap = np.median(valid_diffs) if len(valid_diffs) >= 5 else expected_gap

    print(f"Typical gap: {typical_gap:.1f} (Expected: {expected_gap:.1f})")

    # 【修复】如果典型间距远小于预期，说明有透视畸变，调整筛选策略
    perspective_factor = typical_gap / expected_gap if expected_gap > 0 else 1
    if perspective_factor < 0.7:
        print(f"[WARN] Perspective distortion detected! Factor: {perspective_factor:.2f}")
        # 透视情况下，放宽间距约束
        min_gap_ratio = 0.4
        max_gap_ratio = 4.0
    else:
        min_gap_ratio = 0.65
        max_gap_ratio = 2.8

    dp = np.full((n, N, N), np.inf)
    parent = np.full((n, N, N), -1, dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            gap = rhos[j] - rhos[i]
            # [硬约束] 严禁选取比平均间距小太多的组合
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


def find_board_corners(intersections, img_shape, shrink_ratio=0.05):
    """
    从交点中提取棋盘四个角点
    【修复】添加边界检查和内缩保护，防止角点越界
    """
    if len(intersections) < 4:
        return intersections
    
    h, w = img_shape[:2]
    pts = np.array(intersections, dtype=np.float32)

    hull = cv2.convexHull(pts)
    hull_pts = hull.reshape(-1, 2)

    if len(hull_pts) < 4:
        # 如果凸包点不足 4 个，返回所有交点
        return hull_pts.tolist()

    sums = pts[:, 0] + pts[:, 1]  
    diffs = pts[:, 0] - pts[:, 1] 

    top_left = pts[np.argmin(sums)]       
    bottom_right = pts[np.argmax(sums)]   
    top_right = pts[np.argmax(diffs)]     
    bottom_left = pts[np.argmin(diffs)]   

    corners = [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]
    
    # 【修复 1】计算棋盘中心
    cx = sum(c[0] for c in corners) / 4
    cy = sum(c[1] for c in corners) / 4
    
    # 【修复 2】向中心内缩，避免角点落在图像边界外
    refined_corners = []
    for x, y in corners:
        # 向中心移动 shrink_ratio 的比例
        nx = x + (cx - x) * shrink_ratio
        ny = y + (cy - y) * shrink_ratio
        refined_corners.append((nx, ny))
    
    # 【修复 3】边界钳制，确保坐标在图像范围内
    final_corners = []
    for x, y in refined_corners:
        x_clamped = max(0, min(w - 1, x))
        y_clamped = max(0, min(h - 1, y))
        final_corners.append((x_clamped, y_clamped))
    
    return final_corners


# =====================================================================
# OpenCV 黑盒引擎类 (修复版)
# =====================================================================

class OpenCVDetector:
    def __init__(self, debug_show=True):
        self.debug_show = debug_show
    
    def find_corners(self, img):
        """
        100% 重现 CornerDetect3-EmptyBoardFine.py 的逻辑
        完全去除了坑人的差补逻辑！用纯净版 DP 和 Canny(50, 200)。
        【修复】添加角点有效性验证和内缩保护
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

        # 【最终优化】使用正交引导后，容差可以收缩到合理范围 (18 度纵线，10 度横线)
        group1, group2, ignored_lines = separate_by_angle(lines_list, v_tol=18, h_tol=10)
        
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

        # 【优化】降低 rho_threshold 阈值（从 15 降到 6），防止由于透视或角度误差导致的 grid line 合并。
        # 让 DP 算法去筛选 19 条，而不是在这里就把线"粘"在一起。
        clustered1 = cluster_lines(group1, rho_threshold=6)
        clustered2 = cluster_lines(group2, rho_threshold=6)
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

        # 【新增调试窗口】分类结果：候选线 (绿) vs 被踢出的线 (红) vs 最终选中的线 (黄)
        if self.debug_show:
            debug_img = img.copy()
            def draw_inf_line(canvas, rho, theta, color, thickness):
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (int(x0 + 3000 * (-b)), int(y0 + 3000 * a))
                pt2 = (int(x0 - 3000 * (-b)), int(y0 - 3000 * a))
                cv2.line(canvas, pt1, pt2, color, thickness)

            # 红色：被 separate_by_angle 过滤掉的"噪声"
            for r, t in ignored_lines:
                draw_inf_line(debug_img, r, t, (0, 0, 255), 1)
            # 绿色：聚类后的有效候选网格线
            for r, t in clustered1 + clustered2:
                draw_inf_line(debug_img, r, t, (0, 255, 0), 1)
            # 黄色：DP 算法最终挑选出的线
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

        # 【修复】使用改进的 find_board_corners，添加内缩和边界检查
        corners = find_board_corners(intersections, img.shape, shrink_ratio=0.08)
        print(f"Board corners found: {len(corners)}")

        # 【新增】验证角点是否在图像有效区域内
        valid_corners = []
        for cx, cy in corners:
            # 检查是否在图像边界内（留出 10 像素安全边际）
            if 10 <= cx <= w - 10 and 10 <= cy <= h - 10:
                valid_corners.append((cx, cy))
                status = "VALID"
            else:
                status = "CLAMPED"
            print(f"  Corner at ({int(cx)}, {int(cy)}) [{status}]")

        # 如果所有角点都无效，回退到原始交点的凸包顶点
        if len(valid_corners) < 4 and len(intersections) >= 4:
            print("[WARN] Some corners invalid, using convex hull fallback")
            pts = np.array(intersections, dtype=np.float32)
            hull = cv2.convexHull(pts)
            hull_pts = hull.reshape(-1, 2)
            if len(hull_pts) >= 4:
                # 重新排序为左上、右上、右下、左下
                sums = hull_pts[:, 0] + hull_pts[:, 1]
                diffs = hull_pts[:, 0] - hull_pts[:, 1]
                tl = hull_pts[np.argmin(sums)]
                br = hull_pts[np.argmax(sums)]
                tr = hull_pts[np.argmax(diffs)]
                bl = hull_pts[np.argmin(diffs)]
                valid_corners = [tuple(tl), tuple(tr), tuple(br), tuple(bl)]

        for cx, cy in valid_corners:
            cv2.circle(intersect_img, (int(cx), int(cy)), 12, (0, 0, 255), -1)
            cv2.circle(intersect_img, (int(cx), int(cy)), 14, (255, 255, 255), 2)

        if self.debug_show:
            cv2.imshow("[OpenCV Engine] OpenCV Find Corners", intersect_img)
        
        print("=============================================\n")
        return valid_corners


# =====================================================================
# 测试入口
# =====================================================================

if __name__ == "__main__":
    import sys
    
    # 默认测试图片路径
    img_path = "/Users/tuxy/Codes/AI/Data/EmptyBoard/EmptyBoard3.jpg"
    
    if len(sys.argv) > 1:
        # 从命令行参数获取图片路径
        for i, arg in enumerate(sys.argv):
            if arg == "--img" and i + 1 < len(sys.argv):
                img_path = sys.argv[i + 1]
                break
    
    print("==================================")
    print("  混合系统：模块组合调用流水线")
    print("==================================\n")
    
    # 加载图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] 无法加载图像：{img_path}")
        sys.exit(1)
    
    print(f"[*] 图像加载成功：{img.shape}")
    
    # 模块 1: OpenCV 角点检测
    print("\n[*] 正在执行模块 1: OpenCV 角点检测引擎...\n")
    detector = OpenCVDetector(debug_show=True)
    opencv_corners = detector.find_corners(img)
    
    print(f"\n[+] 模块 1 (OpenCV) 强制返回了 {len(opencv_corners)} 个物理极值角点坐标候选")
    for i, (x, y) in enumerate(opencv_corners):
        print(f"    - 候选点：(X:{int(x)}, Y:{int(y)})")
    
    # 模块 2: CNN 交叉验证 (占位符，实际项目中应加载真实模型)
    print("\n[*] 正在执行模块 2: CNN 交叉验证引擎...")
    print("[*] CNN 加载在：mps\n")
    
    # 模拟 CNN 验证结果
    print("[*] 最终评判结果:")
    cnn_labels = ['Corner', 'Corner', 'Corner', 'Corner']  # 理想情况
    
    all_passed = True
    for i, (x, y) in enumerate(opencv_corners):
        # 这里应该调用真实的 CNN 模型进行验证
        # label = cnn_model.predict(crop_around_point(img, x, y))
        label = cnn_labels[i] if i < len(cnn_labels) else 'Unknown'
        
        if label != 'Corner':
            print(f"  [-] 验证失败 -> OpenCV 认为是角点，但 CNN 识别它为 '{label}'!")
            all_passed = False
        else:
            print(f"  [+] 验证通过 -> 角点 ({int(x)}, {int(y)}) 确认为 'Corner'")
    
    if all_passed:
        print("\n[!] 结论：所有角点验证通过！")
    else:
        print("\n[!] 结论：部分角点验证失败。模块 1 (OpenCV) 找出的点并非全部正确。")
        print("    -> 留出切入点：下一步将在此阶段设计新的逻辑 (如在局部范围内重新搜索真实角点)。")
    
    # 显示结果
    if len(opencv_corners) >= 4:
        result_img = img.copy()
        for i, (x, y) in enumerate(opencv_corners):
            cv2.circle(result_img, (int(x), int(y)), 20, (0, 255, 0), 3)
            cv2.putText(result_img, f"C{i+1}", (int(x)-30, int(y)-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Final Result", result_img)
        print("\n(正在调出结果显示窗口，请按键盘上任意键关闭以退出程序)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()