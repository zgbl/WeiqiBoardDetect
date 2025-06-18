# 改进版本：更好地处理粗线条的检测
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

    if angle_diff > angle_thresh:
        return False

    dist = np.hypot(x1 - x3, y1 - y3)
    return dist < dist_thresh

def group_lines_by_orientation(lines, angle_threshold=15):  # 增加角度容忍度
    horizontal = []
    vertical = []

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold:
            horizontal.append((x1, y1, x2, y2))
        elif abs(angle - 90) < angle_threshold or abs(angle + 90) < angle_threshold:
            vertical.append((x1, y1, x2, y2))
    
    return horizontal, vertical

def classify_lines_by_angle(lines, angle_thresh=15):  # 增加角度容忍度
    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < angle_thresh or abs(angle - 180) < angle_thresh:
            horizontals.append((x1, y1, x2, y2))
        elif abs(angle - 90) < angle_thresh or abs(angle + 90) < angle_thresh:
            verticals.append((x1, y1, x2, y2))
    return horizontals, verticals

def filter_board_lines(line_group, img_shape, axis='h', margin_ratio=0.05):
    if not line_group:
        return []
    
    height, width = img_shape[:2]
    margin_h = int(height * margin_ratio)
    margin_w = int(width * margin_ratio)
    
    filtered_lines = []
    
    for x1, y1, x2, y2 in line_group:
        if axis == 'h':
            y_avg = (y1 + y2) / 2
            if margin_h <= y_avg <= height - margin_h:
                line_length = abs(x2 - x1)
                if line_length >= width * 0.2:  # 降低长度要求
                    filtered_lines.append((x1, y1, x2, y2))
        else:
            x_avg = (x1 + x2) / 2
            if margin_w <= x_avg <= width - margin_w:
                line_length = abs(y2 - y1)
                if line_length >= height * 0.2:  # 降低长度要求
                    filtered_lines.append((x1, y1, x2, y2))
    
    return filtered_lines

def cluster_lines_adaptive(line_group, expected_count, axis='h', tolerance=0.1):
    """改进的聚类算法，更好地处理粗线条"""
    if not line_group:
        return []
    
    # 取坐标
    coords = []
    for x1, y1, x2, y2 in line_group:
        coord = (y1 + y2) / 2 if axis == 'h' else (x1 + x2) / 2
        coords.append([coord])

    if len(coords) < 2:
        return line_group  # 如果线条太少，直接返回

    # 动态确定聚类数量
    coords_1d = [c[0] for c in coords]
    coords_sorted = sorted(coords_1d)
    
    # 计算相邻点之间的距离
    distances = []
    for i in range(1, len(coords_sorted)):
        distances.append(coords_sorted[i] - coords_sorted[i-1])
    
    # 如果距离都很小，说明可能是粗线条的边缘，应该合并
    if distances:
        avg_distance = np.mean(distances)
        min_cluster_distance = max(10, avg_distance * 0.5)  # 最小聚类距离
        
        # 基于距离进行聚类
        clusters = []
        current_cluster = [coords_sorted[0]]
        
        for i in range(1, len(coords_sorted)):
            if coords_sorted[i] - coords_sorted[i-1] <= min_cluster_distance:
                current_cluster.append(coords_sorted[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [coords_sorted[i]]
        clusters.append(current_cluster)
        
        # 为每个簇计算代表位置
        final_lines = []
        for cluster in clusters:
            center = int(np.mean(cluster))
            
            # 找到属于这个簇的所有线条
            cluster_lines = []
            for x1, y1, x2, y2 in line_group:
                line_coord = (y1 + y2) / 2 if axis == 'h' else (x1 + x2) / 2
                if any(abs(line_coord - c) <= min_cluster_distance for c in cluster):
                    cluster_lines.append((x1, y1, x2, y2))
            
            if cluster_lines:
                # 计算这个簇的边界
                xs, ys = [], []
                for x1, y1, x2, y2 in cluster_lines:
                    xs.extend([x1, x2])
                    ys.extend([y1, y2])
                
                if axis == 'h':
                    x_min, x_max = min(xs), max(xs)
                    final_lines.append((x_min, center, x_max, center))
                else:
                    y_min, y_max = min(ys), max(ys)
                    final_lines.append((center, y_min, center, y_max))
        
        return final_lines
    
    # 如果上述方法失败，回退到传统聚类
    n_clusters = min(expected_count, len(coords))
    if n_clusters < 2:
        return line_group

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)
    centers = sorted([int(c[0]) for c in kmeans.cluster_centers_])

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
    lengths = []
    for x1, y1, x2, y2 in lines:
        if axis == 'h':
            length = abs(x2 - x1)
        else:
            length = abs(y2 - y1)
        lengths.append(length)
    
    if not lengths:
        return 0
    
    hist, bins = np.histogram(lengths, bins=10)
    max_freq_idx = np.argmax(hist)
    
    range_start = bins[max_freq_idx]
    range_end = bins[max_freq_idx + 1]
    
    lengths_in_range = [l for l in lengths if range_start <= l <= range_end]
    
    if lengths_in_range:
        return int(np.mean(lengths_in_range))
    else:
        return int(np.mean(lengths))

def calculate_average_spacing(positions):
    if len(positions) < 2:
        return 0
    
    positions = sorted(positions)
    spacings = []
    for i in range(1, len(positions)):
        spacings.append(positions[i] - positions[i-1])
    
    return np.mean(spacings)

def regularize_board_lines(h_lines, v_lines, n_lines=9, img_shape=None):
    """
    改进版规整化函数，可以处理只有横线或只有竖线的情况
    """
    regularized_h_lines = []
    regularized_v_lines = []
    
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
    
    ideal_h_positions = []
    ideal_v_positions = []
    
    # 计算理想横线位置
    if len(h_positions) >= 2:
        total_h_span = h_positions[-1] - h_positions[0]
        ideal_h_spacing = total_h_span / (n_lines - 1) if n_lines > 1 else 0
        start_h_pos = h_positions[0]
        
        for i in range(n_lines):
            ideal_h_positions.append(int(start_h_pos + i * ideal_h_spacing))
    elif len(h_positions) == 1:
        # 只有一条横线的情况
        ideal_h_positions = [int(h_positions[0])]
    
    # 计算理想竖线位置  
    if len(v_positions) >= 2:
        total_v_span = v_positions[-1] - v_positions[0]
        ideal_v_spacing = total_v_span / (n_lines - 1) if n_lines > 1 else 0
        start_v_pos = v_positions[0]
        
        for i in range(n_lines):
            ideal_v_positions.append(int(start_v_pos + i * ideal_v_spacing))
    elif len(v_positions) == 1:
        # 只有一条竖线的情况
        ideal_v_positions = [int(v_positions[0])]
    
    # 生成规整化的横线
    if ideal_h_positions:
        if ideal_v_positions:
            # 有竖线时，横线从最左边的竖线延伸到最右边的竖线
            x_min = min(ideal_v_positions)
            x_max = max(ideal_v_positions)
        else:
            # 没有竖线时，使用原始横线的范围或图像范围
            if h_lines:
                all_x = []
                for x1, y1, x2, y2 in h_lines:
                    all_x.extend([x1, x2])
                x_min, x_max = min(all_x), max(all_x)
            elif img_shape:
                # 使用图像宽度的80%作为线条长度
                margin = int(img_shape[1] * 0.1)
                x_min, x_max = margin, img_shape[1] - margin
            else:
                x_min, x_max = 0, 400  # 默认值
        
        for y_pos in ideal_h_positions:
            regularized_h_lines.append((x_min, y_pos, x_max, y_pos))
    
    # 生成规整化的竖线
    if ideal_v_positions:
        if ideal_h_positions:
            # 有横线时，竖线从最上边的横线延伸到最下边的横线
            y_min = min(ideal_h_positions)
            y_max = max(ideal_h_positions)
        else:
            # 没有横线时，使用原始竖线的范围或图像范围
            if v_lines:
                all_y = []
                for x1, y1, x2, y2 in v_lines:
                    all_y.extend([y1, y2])
                y_min, y_max = min(all_y), max(all_y)
            elif img_shape:
                # 使用图像高度的80%作为线条长度
                margin = int(img_shape[0] * 0.1)
                y_min, y_max = margin, img_shape[0] - margin
            else:
                y_min, y_max = 0, 400  # 默认值
        
        for x_pos in ideal_v_positions:
            regularized_v_lines.append((x_pos, y_min, x_pos, y_max))
    
    return regularized_h_lines, regularized_v_lines

def print_line_analysis(h_lines, v_lines):
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
#img = cv2.imread("../data/raw/GIMP1.jpg")  # 替换为你的图片路径
img = cv2.imread("../data/raw/bd317d54.webp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 改进预处理：更适合粗线条
blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 增加模糊程度
equalized = cv2.equalizeHist(blur)

# 显示原图
cv2.imshow("Original Pic", img) 
cv2.moveWindow("Original Pic", 0, 0)

# 改进边缘检测参数：降低阈值以捕获更多边缘
edges = cv2.Canny(equalized, 30, 100, apertureSize=3)  # 大幅降低阈值

# 设定检测线数
N = 19

print("Equalized shape:", equalized.shape, "dtype:", equalized.dtype)
print("Equalized min/max:", np.min(equalized), np.max(equalized))
cv2.imshow("Equalized", equalized)
cv2.moveWindow("Equalized", 620, 0)

# 显示边缘检测结果
cv2.imshow("Edges", edges)
cv2.moveWindow("Edges", 0, 600)

# 改进霍夫变换参数：更适合粗线条
lines = cv2.HoughLinesP(edges, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=30,      # 降低阈值
                        minLineLength=30,  # 降低最小长度
                        maxLineGap=20)     # 增加最大间隙

# 创建副本用于对比
img_original_detection = img.copy()
img_regularized = img.copy()

# 处理检测结果
lineCount = 0
if lines is not None:
    print(f"检测到原始线段数量: {len(lines)}")
    
    # 分类线条
    h_lines, v_lines = classify_lines_by_angle(lines)
    print(f"分类后 - 横线: {len(h_lines)}, 竖线: {len(v_lines)}")
    
    # 过滤线条
    h_lines = filter_board_lines(h_lines, img.shape, axis='h')
    v_lines = filter_board_lines(v_lines, img.shape, axis='v')
    print(f"过滤后 - 横线: {len(h_lines)}, 竖线: {len(v_lines)}")
    
    # 聚类合并
    merged_h = cluster_lines_adaptive(h_lines, expected_count=N, axis='h')
    merged_v = cluster_lines_adaptive(v_lines, expected_count=N, axis='v')

    # 绘制原始检测结果（绿色）
    for x1, y1, x2, y2 in merged_h + merged_v:
        cv2.line(img_original_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 打印分析信息
    print_line_analysis(merged_h, merged_v)
    
    # 规整化处理（传入图像尺寸）
    regularized_h, regularized_v = regularize_board_lines(merged_h, merged_v, N, img.shape)
    
    print(f"\n规整化后:")
    print(f"横线数量: {len(regularized_h)}")
    print(f"竖线数量: {len(regularized_v)}")
    
    # 绘制规整化后的线条（红色，细线）
    for x1, y1, x2, y2 in regularized_h + regularized_v:
        cv2.line(img_regularized, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 改为细线
else:
    print("没有检测到任何线条!")

# 显示结果
cv2.imshow("Original Detection", img_original_detection)
cv2.moveWindow("Original Detection", 0, 300)

cv2.imshow("Regularized Board", img_regularized)
cv2.moveWindow("Regularized Board", 620, 300)

cv2.waitKey(0)
cv2.destroyAllWindows()