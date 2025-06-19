#参考了 网上的代码 Claude 修改
# 修改版本：结合轮廓检测来准确识别棋盘边框 
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def detect_board_boundary(img):
    """
    使用轮廓检测来找到棋盘边界，参考 cndb1.py 的方法
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(blur, 50, 150)
    
    # 轮廓提取
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到最大的轮廓，即棋盘的轮廓
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    if max_contour is not None:
        # 找到最小外接矩形，即棋盘的四个角点
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # 返回边界框的坐标
        x_coords = box[:, 0]
        y_coords = box[:, 1]
        
        return {
            'x_min': int(np.min(x_coords)),
            'x_max': int(np.max(x_coords)),
            'y_min': int(np.min(y_coords)),
            'y_max': int(np.max(y_coords)),
            'box': box
        }
    
    return None

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

def filter_board_lines(line_group, img_shape, axis='h', margin_ratio=0.02, board_boundary=None):
    """
    改进版过滤函数：如果有棋盘边界信息，则不过滤边界线条
    """
    if not line_group:
        return []
    
    height, width = img_shape[:2]
    margin_h = int(height * margin_ratio)
    margin_w = int(width * margin_ratio)
    
    filtered_lines = []
    
    for x1, y1, x2, y2 in line_group:
        # 如果有棋盘边界信息，检查线条是否在边界附近
        is_boundary_line = False
        if board_boundary:
            if axis == 'h':
                y_avg = (y1 + y2) / 2
                # 检查是否是上下边界线
                if (abs(y_avg - board_boundary['y_min']) < 20 or 
                    abs(y_avg - board_boundary['y_max']) < 20):
                    is_boundary_line = True
            else:
                x_avg = (x1 + x2) / 2
                # 检查是否是左右边界线
                if (abs(x_avg - board_boundary['x_min']) < 20 or 
                    abs(x_avg - board_boundary['x_max']) < 20):
                    is_boundary_line = True
        
        # 如果是边界线，直接保留
        if is_boundary_line:
            filtered_lines.append((x1, y1, x2, y2))
            continue
        
        # 对非边界线应用原来的过滤逻辑
        if axis == 'h':
            y_avg = (y1 + y2) / 2
            if margin_h <= y_avg <= height - margin_h:
                line_length = abs(x2 - x1)
                if line_length >= width * 0.3:
                    filtered_lines.append((x1, y1, x2, y2))
        else:
            x_avg = (x1 + x2) / 2
            if margin_w <= x_avg <= width - margin_w:
                line_length = abs(y2 - y1)
                if line_length >= height * 0.3:
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
        keep_count = min(expected_count * 2, len(line_qualities))
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

def regularize_board_lines_with_boundary(h_lines, v_lines, n_lines=19, board_boundary=None):
    """
    改进版规整化函数：使用棋盘边界信息来确保边框线条的正确性
    """
    regularized_h_lines = []
    regularized_v_lines = []
    
    # 如果有边界信息，使用边界来确定线条范围
    if board_boundary:
        x_min = board_boundary['x_min']
        x_max = board_boundary['x_max']
        y_min = board_boundary['y_min']
        y_max = board_boundary['y_max']
        
        # 生成等间距的横线位置
        for i in range(n_lines):
            y_pos = int(y_min + i * (y_max - y_min) / (n_lines - 1))
            regularized_h_lines.append((x_min, y_pos, x_max, y_pos))
        
        # 生成等间距的竖线位置
        for i in range(n_lines):
            x_pos = int(x_min + i * (x_max - x_min) / (n_lines - 1))
            regularized_v_lines.append((x_pos, y_min, x_pos, y_max))
        
        return regularized_h_lines, regularized_v_lines
    
    # 如果没有边界信息，使用原来的方法
    h_positions = []
    if h_lines:
        for x1, y1, x2, y2 in h_lines:
            h_positions.append((y1 + y2) / 2)
        h_positions = sorted(h_positions)
    
    v_positions = []
    if v_lines:
        for x1, y1, x2, y2 in v_lines:
            v_positions.append((x1 + x2) / 2)
        v_positions = sorted(v_positions)
    
    # 计算理想的等间距位置
    ideal_h_positions = []
    ideal_v_positions = []
    
    if len(h_positions) >= 2:
        total_h_span = h_positions[-1] - h_positions[0]
        ideal_h_spacing = total_h_span / (n_lines - 1) if n_lines > 1 else 0
        start_h_pos = h_positions[0]
        
        for i in range(n_lines):
            ideal_h_positions.append(int(start_h_pos + i * ideal_h_spacing))
    
    if len(v_positions) >= 2:
        total_v_span = v_positions[-1] - v_positions[0]
        ideal_v_spacing = total_v_span / (n_lines - 1) if n_lines > 1 else 0
        start_v_pos = v_positions[0]
        
        for i in range(n_lines):
            ideal_v_positions.append(int(start_v_pos + i * ideal_v_spacing))
    
    # 生成规整的线条
    if ideal_h_positions and ideal_v_positions:
        x_min = min(ideal_v_positions)
        x_max = max(ideal_v_positions)
        
        for y_pos in ideal_h_positions:
            regularized_h_lines.append((x_min, y_pos, x_max, y_pos))
    
    if ideal_v_positions and ideal_h_positions:
        y_min = min(ideal_h_positions)
        y_max = max(ideal_h_positions)
        
        for x_pos in ideal_v_positions:
            regularized_v_lines.append((x_pos, y_min, x_pos, y_max))
    
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

# 主程序
# 1. 读图 & 转灰度
#img = cv2.imread("../data/raw/BasicStraightLine2.jpg")
#img = cv2.imread("../data/raw/OGS2.jpg")
img = cv2.imread("../data/raw/bd317d54.webp")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 检测棋盘边界（参考 cndb1.py 的方法）
print("=== 检测棋盘边界 ===")
board_boundary = detect_board_boundary(img)
if board_boundary:
    print(f"棋盘边界: x({board_boundary['x_min']}, {board_boundary['x_max']}), y({board_boundary['y_min']}, {board_boundary['y_max']})")
    # 绘制边界框
    img_with_boundary = img.copy()
    cv2.rectangle(img_with_boundary, 
                  (board_boundary['x_min'], board_boundary['y_min']), 
                  (board_boundary['x_max'], board_boundary['y_max']), 
                  (255, 0, 0), 3)  # 蓝色边界框
else:
    print("未能检测到棋盘边界")

# 先做高斯模糊 + 提升对比度
blur = cv2.GaussianBlur(gray, (3, 3), 0)
equalized = cv2.equalizeHist(blur)

# 显示原图
cv2.imshow("Original Pic", img) 
cv2.moveWindow("Original Pic", 0, 0)

if board_boundary:
    cv2.imshow("Board Boundary", img_with_boundary)
    cv2.moveWindow("Board Boundary", 310, 0)

# 3. 边缘检测
edges = cv2.Canny(equalized, 50, 150)

# 设定检测 N 路棋盘
N = 19

print("Equalized shape:", equalized.shape, "dtype:", equalized.dtype)
print("Equalized min/max:", np.min(equalized), np.max(equalized))
cv2.imshow("Equalizeed", equalized)
cv2.moveWindow("Equalizeed", 620, 0)

# 4. 使用 HoughLinesP 检测"线段"
lines = cv2.HoughLinesP(edges, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=50,
                        minLineLength=50,
                        maxLineGap=10)

# 创建副本用于对比
img_original_detection = img.copy()
img_regularized = img.copy()

# 5. 原始检测结果
if lines is not None:
    # 拆分横线和竖线
    horizontals, verticals = group_lines_by_orientation(lines)

    # 分组并拟合（使用改进的过滤函数）
    h_lines, v_lines = classify_lines_by_angle(lines)
    
    # 使用改进的过滤函数，传入棋盘边界信息
    filtered_h = filter_board_lines(h_lines, img.shape, axis='h', board_boundary=board_boundary)
    filtered_v = filter_board_lines(v_lines, img.shape, axis='v', board_boundary=board_boundary)
    
    merged_h = cluster_lines_adaptive(filtered_h, expected_count=N, axis='h')
    merged_v = cluster_lines_adaptive(filtered_v, expected_count=N, axis='v')

    # 原始检测结果（绿色线条）
    for x1, y1, x2, y2 in merged_h + merged_v:
        cv2.line(img_original_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 打印分析信息
    print_line_analysis(merged_h, merged_v)
    
    # 6. 规整化处理（使用边界信息）
    regularized_h, regularized_v = regularize_board_lines_with_boundary(
        merged_h, merged_v, N, board_boundary)
    
    print(f"\n规整化后:")
    print(f"横线数量: {len(regularized_h)}")
    print(f"竖线数量: {len(regularized_v)}")
    
    # 绘制规整化后的线条（红色线条）
    for x1, y1, x2, y2 in regularized_h + regularized_v:
        cv2.line(img_regularized, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 7. 显示对比结果
cv2.imshow("Original Detection", img_original_detection)
cv2.moveWindow("Original Detection", 0, 300)

cv2.imshow("Regularized Board", img_regularized)
cv2.moveWindow("Regularized Board", 310, 300)

cv2.waitKey(0)
cv2.destroyAllWindows()