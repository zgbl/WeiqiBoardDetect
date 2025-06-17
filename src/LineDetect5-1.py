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

def filter_board_lines(line_group, img_shape, axis='h', margin_ratio=0.05):
    """
    过滤掉边缘误判的线条和质量不高的线条
    """
    if not line_group:
        return []
    
    height, width = img_shape[:2]
    margin_h = int(height * margin_ratio)
    margin_w = int(width * margin_ratio)
    
    filtered_lines = []
    
    for x1, y1, x2, y2 in line_group:
        # 过滤掉太靠近边缘的线条
        if axis == 'h':
            # 横线：检查y坐标是否太靠近上下边缘
            y_avg = (y1 + y2) / 2
            if margin_h <= y_avg <= height - margin_h:
                # 检查线条长度是否合理（至少跨越图像的一定比例）
                line_length = abs(x2 - x1)
                if line_length >= width * 0.3:  # 至少30%的图像宽度
                    filtered_lines.append((x1, y1, x2, y2))
        else:
            # 竖线：检查x坐标是否太靠近左右边缘
            x_avg = (x1 + x2) / 2
            if margin_w <= x_avg <= width - margin_w:
                # 检查线条长度是否合理
                line_length = abs(y2 - y1)
                if line_length >= height * 0.3:  # 至少30%的图像高度
                    filtered_lines.append((x1, y1, x2, y2))
    
    return filtered_lines

def cluster_lines_adaptive(line_group, expected_count, axis='h', tolerance=0.1):
    """
    自适应聚类，动态调整聚类数量
    """
    if not line_group:
        return []
    
    # 取横向：按 y 聚类，取竖向：按 x 聚类
    coords = []
    for x1, y1, x2, y2 in line_group:
        coord = (y1 + y2) / 2 if axis == 'h' else (x1 + x2) / 2
        coords.append([coord])

    if len(coords) < 2:
        return []
    
    # 首先尝试期望的聚类数量
    n_clusters = min(expected_count, len(coords))
    
    # 如果线条数量明显超过期望值，可能有误判，先尝试过滤
    if len(coords) > expected_count * 1.5:
        # 根据线条质量（长度）进行过滤
        line_qualities = []
        for x1, y1, x2, y2 in line_group:
            if axis == 'h':
                length = abs(x2 - x1)
            else:
                length = abs(y2 - y1)
            line_qualities.append((length, (x1, y1, x2, y2)))
        
        # 保留质量较高的线条
        line_qualities.sort(reverse=True)
        keep_count = min(expected_count * 2, len(line_qualities))  # 最多保留2倍的期望数量
        line_group = [line for _, line in line_qualities[:keep_count]]
        
        # 重新计算坐标
        coords = []
        for x1, y1, x2, y2 in line_group:
            coord = (y1 + y2) / 2 if axis == 'h' else (x1 + x2) / 2
            coords.append([coord])
        
        n_clusters = min(expected_count, len(coords))

    if n_clusters < 2:
        return []

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

def regularize_board_lines(h_lines, v_lines, n_lines=9):
    """
    根据围棋盘的规律性，规整化线条位置和长度
    """
    # 1. 处理横线
    regularized_h_lines = []
    if h_lines:
        # 提取所有横线的y坐标
        h_positions = []
        for x1, y1, x2, y2 in h_lines:
            h_positions.append((y1 + y2) / 2)
        
        # 计算平均间距
        avg_h_spacing = calculate_average_spacing(h_positions)
        
        # 获取主导线条长度
        dominant_h_length = get_dominant_line_length(h_lines, 'h')
        
        # 计算理想的起始和结束位置
        h_positions_sorted = sorted(h_positions)
        
        # 使用最小二乘法拟合等间距的线条位置
        if len(h_positions_sorted) >= 2:
            # 计算理想的起始位置和间距
            total_span = h_positions_sorted[-1] - h_positions_sorted[0]
            ideal_spacing = total_span / (n_lines - 1) if n_lines > 1 else avg_h_spacing
            start_pos = h_positions_sorted[0]
            
            # 获取横线的左右边界
            all_x_coords = []
            for x1, y1, x2, y2 in h_lines:
                all_x_coords.extend([x1, x2])
            
            x_min = min(all_x_coords)
            x_max = max(all_x_coords)
            
            # 如果主导长度更合理，使用主导长度
            if dominant_h_length > 0:
                line_length = dominant_h_length
                # 居中对齐
                center_x = (x_min + x_max) / 2
                x_start = int(center_x - line_length / 2)
                x_end = int(center_x + line_length / 2)
            else:
                x_start = x_min
                x_end = x_max
            
            # 生成规整的横线
            for i in range(n_lines):
                y_pos = int(start_pos + i * ideal_spacing)
                regularized_h_lines.append((x_start, y_pos, x_end, y_pos))
    
    # 2. 处理竖线
    regularized_v_lines = []
    if v_lines:
        # 提取所有竖线的x坐标
        v_positions = []
        for x1, y1, x2, y2 in v_lines:
            v_positions.append((x1 + x2) / 2)
        
        # 计算平均间距
        avg_v_spacing = calculate_average_spacing(v_positions)
        
        # 获取主导线条长度
        dominant_v_length = get_dominant_line_length(v_lines, 'v')
        
        # 计算理想的起始和结束位置
        v_positions_sorted = sorted(v_positions)
        
        # 使用最小二乘法拟合等间距的线条位置
        if len(v_positions_sorted) >= 2:
            # 计算理想的起始位置和间距
            total_span = v_positions_sorted[-1] - v_positions_sorted[0]
            ideal_spacing = total_span / (n_lines - 1) if n_lines > 1 else avg_v_spacing
            start_pos = v_positions_sorted[0]
            
            # 获取竖线的上下边界
            all_y_coords = []
            for x1, y1, x2, y2 in v_lines:
                all_y_coords.extend([y1, y2])
            
            y_min = min(all_y_coords)
            y_max = max(all_y_coords)
            
            # 如果主导长度更合理，使用主导长度
            if dominant_v_length > 0:
                line_length = dominant_v_length
                # 居中对齐
                center_y = (y_min + y_max) / 2
                y_start = int(center_y - line_length / 2)
                y_end = int(center_y + line_length / 2)
            else:
                y_start = y_min
                y_end = y_max
            
            # 生成规整的竖线
            for i in range(n_lines):
                x_pos = int(start_pos + i * ideal_spacing)
                regularized_v_lines.append((x_pos, y_start, x_pos, y_end))
    
    return regularized_h_lines, regularized_v_lines

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
#edges = cv2.Canny(gray, 40, 160, apertureSize=3)
edges = cv2.Canny(equalized, 50, 150)   # 提升了对比度后的Canny

#设定检测 N 路棋盘
N = 9

print("Equalized shape:", equalized.shape, "dtype:", equalized.dtype)
print("Equalized min/max:", np.min(equalized), np.max(equalized))
cv2.imshow("Equalizeed", equalized)
cv2.moveWindow("Equalizeed", 600, 0)

# 3. 使用 HoughLinesP 检测"线段"（不是无限延长线）
# 调整参数以提高检测质量
lines = cv2.HoughLinesP(edges, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=50,  # 提高threshold减少噪声线条
                        minLineLength=50,  # 增加最小线条长度
                        maxLineGap=10)  # 允许更大的gap来连接断开的线条

# 创建两个副本用于对比
img_original_detection = img.copy()
img_regularized = img.copy()

# 4. 原始检测结果
lineCount = 0
if lines is not None:
    # 拆分横线和竖线
    horizontals, verticals = group_lines_by_orientation(lines)

    # 分组并拟合
    h_lines, v_lines = classify_lines_by_angle(lines)
    merged_h = cluster_lines_adaptive(h_lines, expected_count=N, axis='h')
    merged_v = cluster_lines_adaptive(v_lines, expected_count=N, axis='v')

    # 原始检测结果（绿色线条）
    for x1, y1, x2, y2 in merged_h + merged_v:
        cv2.line(img_original_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 打印分析信息
    print_line_analysis(merged_h, merged_v)
    
    # 5. 规整化处理
    regularized_h, regularized_v = regularize_board_lines(merged_h, merged_v, N)
    
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
cv2.moveWindow("Regularized Board", 600, 300)

cv2.waitKey(0)
cv2.destroyAllWindows()