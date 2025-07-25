import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def is_similar(line1, line2, dist_thresh=3, angle_thresh=5):
    x1, y1, x2, y2 = line1
    a1 = np.arctan2(y2 - y1, x2 - x1)
    x3, y3, x4, y4 = line2
    a2 = np.arctan2(y4 - y3, x4 - x3)
    angle_diff = abs(a1 - a2) * 180 / np.pi
    angle_diff = min(angle_diff, 180 - angle_diff)

    # 判断角度是否接近
    if angle_diff > angle_thresh:
        return False

    # 判断起点和终点是否接近（任选一个点即可）
    dist = np.hypot(x1 - x3, y1 - y3)
    return dist < dist_thresh

def group_lines_by_orientation(lines, angle_threshold=10):
    horizontal = []
    vertical = []

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold:
            horizontal.append((x1, y1, x2, y2))
        elif abs(angle - 90) < angle_threshold or abs(angle + 90) < angle_threshold:
            vertical.append((x1, y1, x2, y2))
    
    return horizontal, vertical

def fit_lines(line_group, axis='h'):
    # axis='h': 横线，按 y 聚合；axis='v': 纵线，按 x 聚合
    bins = defaultdict(list)
    for x1, y1, x2, y2 in line_group:
        key = y1 if axis == 'h' else x1
        bins[key // 5 * 5].append((x1, y1, x2, y2))  # 用 5 像素为窗口聚合

    fitted_lines = []
    for group in bins.values():
        xs, ys = [], []
        for x1, y1, x2, y2 in group:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        if axis == 'h':
            y_avg = int(np.mean(ys))
            x_min, x_max = min(xs), max(xs)
            fitted_lines.append((x_min, y_avg, x_max, y_avg))
        else:
            x_avg = int(np.mean(xs))
            y_min, y_max = min(ys), max(ys)
            fitted_lines.append((x_avg, y_min, x_avg, y_max))
    return fitted_lines

def classify_lines_by_angle(lines, angle_thresh=10):
    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < angle_thresh or abs(angle - 180) < angle_thresh:
            horizontals.append((x1, y1, x2, y2))
        elif abs(angle - 90) < angle_thresh or abs(angle + 90) < angle_thresh:
            verticals.append((x1, y1, x2, y2))
    return horizontals, verticals

def filter_board_lines_lenient(line_group, img_shape, axis='h', margin_ratio=0.03):
    """
    更宽松的线条过滤，只过滤明显的噪声线条
    """
    if not line_group:
        return []
    
    height, width = img_shape[:2]
    margin_h = int(height * margin_ratio)  # 减少到3%
    margin_w = int(width * margin_ratio)
    
    filtered_lines = []
    
    for x1, y1, x2, y2 in line_group:
        # 只过滤掉非常靠近边缘的短线条
        if axis == 'h':
            y_avg = (y1 + y2) / 2
            line_length = abs(x2 - x1)
            # 对于靠近边缘的线条，要求更长的长度
            min_length_ratio = 0.15 if (y_avg < margin_h or y_avg > height - margin_h) else 0.1
            if line_length >= width * min_length_ratio:
                filtered_lines.append((x1, y1, x2, y2))
        else:
            x_avg = (x1 + x2) / 2
            line_length = abs(y2 - y1)
            # 对于靠近边缘的线条，要求更长的长度
            min_length_ratio = 0.15 if (x_avg < margin_w or x_avg > width - margin_w) else 0.1
            if line_length >= height * min_length_ratio:
                filtered_lines.append((x1, y1, x2, y2))
    
    return filtered_lines

def cluster_lines_conservative(line_group, expected_count, axis='h'):
    """
    保守的聚类方法，尽量保留所有检测到的线条
    """
    if not line_group:
        return []
    
    # 取坐标用于聚类
    coords = []
    for x1, y1, x2, y2 in line_group:
        coord = (y1 + y2) / 2 if axis == 'h' else (x1 + x2) / 2
        coords.append([coord])

    if len(coords) < 2:
        return line_group  # 如果线条太少，直接返回

    # 只在线条数量远超预期时才进行异常值过滤
    coord_values = [c[0] for c in coords]
    if len(coord_values) > expected_count * 2:  # 提高阈值到2倍
        # 使用更宽松的异常值检测
        q1 = np.percentile(coord_values, 15)  # 从25%改为15%
        q3 = np.percentile(coord_values, 85)  # 从75%改为85%
        iqr = q3 - q1
        
        # 使用更宽松的异常值阈值
        lower_bound = q1 - 3.0 * iqr  # 从2.0改为3.0
        upper_bound = q3 + 3.0 * iqr
        
        # 过滤异常值
        filtered_coords = []
        filtered_lines = []
        for i, coord in enumerate(coord_values):
            if lower_bound <= coord <= upper_bound:
                filtered_coords.append([coord])
                filtered_lines.append(line_group[i])
        
        if len(filtered_coords) >= 2:
            coords = filtered_coords
            line_group = filtered_lines
            print(f"异常值过滤: {axis}轴从{len(coord_values)}条线减少到{len(coords)}条")

    # 动态确定聚类数量
    n_clusters = min(len(coords), max(expected_count, len(coords) // 2))
    
    if n_clusters < 2:
        return line_group

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)
    centers = sorted([int(c[0]) for c in kmeans.cluster_centers_])

    # 对每个中心，选取该簇中所有点，拟合出一条代表线
    final_lines = []
    for i, center in enumerate(centers):
        group_lines = [line for idx, line in enumerate(line_group) if labels[idx] == i]
        xs, ys = [], []
        for x1, y1, x2, y2 in group_lines:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        if axis == 'h':
            x_min, x_max = min(xs), max(xs)
            final_lines.append((x_min, center, x_max, center))
        else:
            y_min, y_max = min(ys), max(ys)
            final_lines.append((center, y_min, center, y_max))
    return final_lines

def get_dominant_line_length(lines, axis='h'):
    """
    获取主导线条长度（最常见的完整线条长度）
    """
    lengths = []
    for x1, y1, x2, y2 in lines:
        if axis == 'h':
            length = abs(x2 - x1)
        else:
            length = abs(y2 - y1)
        lengths.append(length)
    
    if not lengths:
        return 0
    
    # 使用直方图找到最常见的长度范围
    hist, bins = np.histogram(lengths, bins=10)
    max_freq_idx = np.argmax(hist)
    
    # 返回该范围内的平均长度
    range_start = bins[max_freq_idx]
    range_end = bins[max_freq_idx + 1]
    
    # 找到在这个范围内的所有长度
    lengths_in_range = [l for l in lengths if range_start <= l <= range_end]
    
    if lengths_in_range:
        return int(np.mean(lengths_in_range))
    else:
        return int(np.mean(lengths))

def calculate_average_spacing(positions):
    """
    计算平均间距
    """
    if len(positions) < 2:
        return 0
    
    positions = sorted(positions)
    spacings = []
    for i in range(1, len(positions)):
        spacings.append(positions[i] - positions[i-1])
    
    return np.mean(spacings)

def regularize_board_lines_improved(h_lines, v_lines, n_lines=9):
    """
    改进的规整化函数，更好地处理边缘线条
    """
    print(f"输入: 横线 {len(h_lines)} 条, 竖线 {len(v_lines)} 条")
    
    regularized_h_lines = []
    regularized_v_lines = []
    
    # 处理横线
    if h_lines:
        h_positions = []
        for x1, y1, x2, y2 in h_lines:
            h_positions.append((y1 + y2) / 2)
        h_positions = sorted(h_positions)
        print(f"原始横线位置: {[int(p) for p in h_positions]}")
        
        # 智能选择线条子集
        selected_h_positions = select_best_lines(h_positions, n_lines)
        print(f"选择的横线位置: {[int(p) for p in selected_h_positions]}")
        
        if len(selected_h_positions) >= 2:
            # 基于选定的线条计算等间距
            total_h_span = selected_h_positions[-1] - selected_h_positions[0]
            ideal_h_spacing = total_h_span / (n_lines - 1) if n_lines > 1 else 0
            start_h_pos = selected_h_positions[0]
            
            ideal_h_positions = []
            for i in range(n_lines):
                ideal_h_positions.append(int(start_h_pos + i * ideal_h_spacing))
            
            print(f"理想横线位置: {ideal_h_positions}")
            
            # 确定横线的x坐标范围
            if v_lines:
                v_positions = [(x1 + x2) / 2 for x1, y1, x2, y2 in v_lines]
                x_min = int(min(v_positions))
                x_max = int(max(v_positions))
            else:
                all_x = []
                for x1, y1, x2, y2 in h_lines:
                    all_x.extend([x1, x2])
                x_min = min(all_x)
                x_max = max(all_x)
            
            # 生成规整的横线
            for y_pos in ideal_h_positions:
                regularized_h_lines.append((x_min, y_pos, x_max, y_pos))
    
    # 处理竖线（类似逻辑）
    if v_lines:
        v_positions = []
        for x1, y1, x2, y2 in v_lines:
            v_positions.append((x1 + x2) / 2)
        v_positions = sorted(v_positions)
        print(f"原始竖线位置: {[int(p) for p in v_positions]}")
        
        # 智能选择线条子集
        selected_v_positions = select_best_lines(v_positions, n_lines)
        print(f"选择的竖线位置: {[int(p) for p in selected_v_positions]}")
        
        if len(selected_v_positions) >= 2:
            # 基于选定的线条计算等间距
            total_v_span = selected_v_positions[-1] - selected_v_positions[0]
            ideal_v_spacing = total_v_span / (n_lines - 1) if n_lines > 1 else 0
            start_v_pos = selected_v_positions[0]
            
            ideal_v_positions = []
            for i in range(n_lines):
                ideal_v_positions.append(int(start_v_pos + i * ideal_v_spacing))
            
            print(f"理想竖线位置: {ideal_v_positions}")
            
            # 确定竖线的y坐标范围
            if h_lines:
                h_positions_for_v = [(y1 + y2) / 2 for x1, y1, x2, y2 in h_lines]
                y_min = int(min(h_positions_for_v))
                y_max = int(max(h_positions_for_v))
            else:
                all_y = []
                for x1, y1, x2, y2 in v_lines:
                    all_y.extend([y1, y2])
                y_min = min(all_y)
                y_max = max(all_y)
            
            # 生成规整的竖线
            for x_pos in ideal_v_positions:
                regularized_v_lines.append((x_pos, y_min, x_pos, y_max))
    
    print(f"输出: 横线 {len(regularized_h_lines)} 条, 竖线 {len(regularized_v_lines)} 条")
    return regularized_h_lines, regularized_v_lines

def select_best_lines(positions, target_count):
    """
    智能选择最佳的线条子集
    """
    if len(positions) <= target_count:
        return positions
    
    # 如果线条数量过多，选择分布最均匀的子集
    positions = sorted(positions)
    
    # 方法1：如果有明显的两个边界，选择包含边界的均匀分布
    total_span = positions[-1] - positions[0]
    
    # 计算所有可能的子集，选择间距最均匀的
    best_subset = positions[:target_count]
    best_variance = float('inf')
    
    # 使用滑动窗口寻找最佳子集
    for start_idx in range(len(positions) - target_count + 1):
        subset = positions[start_idx:start_idx + target_count]
        
        # 计算间距的方差
        spacings = []
        for i in range(1, len(subset)):
            spacings.append(subset[i] - subset[i-1])
        
        if spacings:
            variance = np.var(spacings)
            if variance < best_variance:
                best_variance = variance
                best_subset = subset
    
    return best_subset

def print_line_analysis(h_lines, v_lines):
    """
    打印线条分析信息
    """
    print("=== 线条分析 ===")
    print(f"检测到横线数量: {len(h_lines)}")
    print(f"检测到竖线数量: {len(v_lines)}")
    
    if h_lines:
        h_lengths = [abs(x2-x1) for x1,y1,x2,y2 in h_lines]
        h_positions = [(y1+y2)/2 for x1,y1,x2,y2 in h_lines]
        print(f"横线长度范围: {min(h_lengths):.1f} - {max(h_lengths):.1f}")
        print(f"横线主导长度: {get_dominant_line_length(h_lines, 'h')}")
        print(f"横线平均间距: {calculate_average_spacing(h_positions):.1f}")
    
    if v_lines:
        v_lengths = [abs(y2-y1) for x1,y1,x2,y2 in v_lines]
        v_positions = [(x1+x2)/2 for x1,y1,x2,y2 in v_lines]
        print(f"竖线长度范围: {min(v_lengths):.1f} - {max(v_lengths):.1f}")
        print(f"竖线主导长度: {get_dominant_line_length(v_lines, 'v')}")
        print(f"竖线平均间距: {calculate_average_spacing(v_positions):.1f}")
    print("================")

# 1. 读图 & 转灰度
#img = cv2.imread("../data/raw/BasicStraightLine2.jpg")
img = cv2.imread("../data/raw/Pic1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#先做高斯模糊 + 提升对比度
blur = cv2.GaussianBlur(gray, (3, 3), 0)
equalized = cv2.equalizeHist(blur)

#显示原图
cv2.imshow("Original Pic", img) 
cv2.moveWindow("Original Pic", 0, 0)

# 2. 边缘检测
edges = cv2.Canny(equalized, 50, 150)

#设定检测 N 路棋盘
N = 9

print("Equalized shape:", equalized.shape, "dtype:", equalized.dtype)
print("Equalized min/max:", np.min(equalized), np.max(equalized))
cv2.imshow("Equalizeed", equalized)
cv2.moveWindow("Equalizeed", 320, 0)

# 3. 使用 HoughLinesP 检测"线段"
lines = cv2.HoughLinesP(edges, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=50,
                        minLineLength=50,
                        maxLineGap=10)

# 创建两个副本用于对比
img_original_detection = img.copy()
img_regularized = img.copy()

# 4. 检测和处理
if lines is not None:
    # 拆分横线和竖线
    h_lines, v_lines = classify_lines_by_angle(lines)
    
    # 更宽松的过滤
    h_lines_filtered = filter_board_lines_lenient(h_lines, img.shape, axis='h')
    v_lines_filtered = filter_board_lines_lenient(v_lines, img.shape, axis='v')
    
    # 保守的聚类合并
    merged_h = cluster_lines_conservative(h_lines_filtered, expected_count=N, axis='h')
    merged_v = cluster_lines_conservative(v_lines_filtered, expected_count=N, axis='v')

    # 原始检测结果（绿色线条）
    for x1, y1, x2, y2 in merged_h + merged_v:
        cv2.line(img_original_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 打印分析信息
    print_line_analysis(merged_h, merged_v)
    
    # 5. 改进的规整化处理
    regularized_h, regularized_v = regularize_board_lines_improved(merged_h, merged_v, N)
    
    print(f"\n规整化后:")
    print(f"横线数量: {len(regularized_h)}")
    print(f"竖线数量: {len(regularized_v)}")
    
    # 绘制规整化后的线条（红色线条）
    for x1, y1, x2, y2 in regularized_h + regularized_v:
        cv2.line(img_regularized, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 6. 显示对比结果
cv2.imshow("Original Detection", img_original_detection)
cv2.moveWindow("Original Detection", 0, 300)

cv2.imshow("Regularized Board", img_regularized)
cv2.moveWindow("Regularized Board", 320, 300)

cv2.waitKey(0)
cv2.destroyAllWindows()