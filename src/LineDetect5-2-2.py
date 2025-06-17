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

def classify_lines_by_angle(lines, angle_thresh=15):  # 放宽角度阈值
    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # 规范化角度到[-90, 90]
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
            
        if abs(angle) < angle_thresh:  # 水平线
            horizontals.append((x1, y1, x2, y2))
        elif abs(angle - 90) < angle_thresh or abs(angle + 90) < angle_thresh:  # 垂直线
            verticals.append((x1, y1, x2, y2))
    return horizontals, verticals

def filter_edge_lines(line_group, img_shape, axis='h', margin_ratio=0.08):
    """
    更严格地过滤边缘线条
    """
    if not line_group:
        return []
    
    height, width = img_shape[:2]
    margin_h = int(height * margin_ratio)
    margin_w = int(width * margin_ratio)
    
    filtered_lines = []
    
    for x1, y1, x2, y2 in line_group:
        if axis == 'h':
            # 横线：检查y坐标
            y_avg = (y1 + y2) / 2
            # 更严格的边缘过滤
            if margin_h < y_avg < height - margin_h:
                line_length = abs(x2 - x1)
                # 要求横线至少跨越图像的40%宽度
                if line_length >= width * 0.4:
                    filtered_lines.append((x1, y1, x2, y2))
        else:
            # 竖线：检查x坐标
            x_avg = (x1 + x2) / 2
            if margin_w < x_avg < width - margin_w:
                line_length = abs(y2 - y1)
                # 要求竖线至少跨越图像的40%高度
                if line_length >= height * 0.4:
                    filtered_lines.append((x1, y1, x2, y2))
    
    return filtered_lines

def detect_lines_multi_threshold(edges, img_shape):
    """
    使用多个阈值来检测线条，确保不遗漏重要线条
    """
    all_lines = []
    
    # 调整参数，减少噪声线条
    thresholds = [50, 60, 70]  # 提高阈值
    min_lengths = [50, 60, 80]  # 增加最小长度要求
    
    for threshold, min_length in zip(thresholds, min_lengths):
        lines = cv2.HoughLinesP(edges, 
                               rho=1, 
                               theta=np.pi/180, 
                               threshold=threshold,
                               minLineLength=min_length,
                               maxLineGap=8)  # 减小gap以避免连接不相关的边缘
        
        if lines is not None:
            print(f"阈值{threshold}, 最小长度{min_length}: 检测到{len(lines)}条线段")
            for line in lines:
                all_lines.append(line)
    
    if not all_lines:
        return None
    
    return np.array(all_lines)

def remove_duplicate_lines(lines, distance_thresh=5):
    """
    去除重复和相似的线条 - 优化版本
    """
    if lines is None or len(lines) == 0:
        return []
    
    # 转换所有线条为统一格式
    processed_lines = []
    for line in lines:
        if isinstance(line, np.ndarray) and len(line.shape) > 0:
            if len(line.shape) == 2:
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line
        else:
            x1, y1, x2, y2 = line
        processed_lines.append((x1, y1, x2, y2))
    
    if len(processed_lines) > 5000:  # 如果线条太多，先进行粗过滤
        print(f"线条数量过多({len(processed_lines)})，进行粗过滤...")
        # 按长度过滤，保留较长的线条
        line_lengths = []
        for x1, y1, x2, y2 in processed_lines:
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            line_lengths.append((length, (x1, y1, x2, y2)))
        
        # 排序并保留前50%的长线条
        line_lengths.sort(reverse=True)
        processed_lines = [line for _, line in line_lengths[:len(line_lengths)//2]]
        print(f"粗过滤后剩余 {len(processed_lines)} 条线段")
    
    unique_lines = []
    
    for i, (x1, y1, x2, y2) in enumerate(processed_lines):
        if i % 1000 == 0 and i > 0:
            print(f"处理进度: {i}/{len(processed_lines)}")
            
        is_duplicate = False
        
        for ex1, ey1, ex2, ey2 in unique_lines:
            # 计算两条线的距离
            dist1 = np.sqrt((x1 - ex1)**2 + (y1 - ey1)**2)
            dist2 = np.sqrt((x2 - ex2)**2 + (y2 - ey2)**2)
            
            # 也检查交叉情况
            dist3 = np.sqrt((x1 - ex2)**2 + (y1 - ey2)**2)
            dist4 = np.sqrt((x2 - ex1)**2 + (y2 - ey1)**2)
            
            # 如果两条线很接近，认为是重复
            if (dist1 < distance_thresh and dist2 < distance_thresh) or \
               (dist3 < distance_thresh and dist4 < distance_thresh):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_lines.append((x1, y1, x2, y2))
    
    return unique_lines

def adaptive_clustering(line_group, expected_count, axis='h', img_shape=None):
    """
    改进的自适应聚类
    """
    if not line_group:
        return []
    
    # 先按位置排序，便于分析
    if axis == 'h':
        line_group = sorted(line_group, key=lambda line: (line[1] + line[3]) / 2)
        coords = [[(line[1] + line[3]) / 2] for line in line_group]
    else:
        line_group = sorted(line_group, key=lambda line: (line[0] + line[2]) / 2)
        coords = [[(line[0] + line[2]) / 2] for line in line_group]
    
    if len(coords) < 2:
        return line_group
    
    # 分析间距，确定合理的聚类数
    positions = [coord[0] for coord in coords]
    spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    
    if spacings:
        median_spacing = np.median(spacings)
        
        # 基于间距估计应该有多少条线
        total_span = positions[-1] - positions[0]
        estimated_lines = int(total_span / median_spacing) + 1
        
        # 在估计值和期望值之间取平衡
        n_clusters = min(max(estimated_lines, expected_count), len(coords))
    else:
        n_clusters = min(expected_count, len(coords))
    
    if n_clusters <= 1:
        return line_group
    
    # 执行聚类
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)
    
    # 生成最终线条
    final_lines = []
    for i in range(n_clusters):
        cluster_lines = [line for idx, line in enumerate(line_group) if labels[idx] == i]
        if cluster_lines:
            # 合并同一簇的线条
            xs, ys = [], []
            for x1, y1, x2, y2 in cluster_lines:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
            
            if axis == 'h':
                y_avg = int(np.mean(ys))
                x_min, x_max = min(xs), max(xs)
                final_lines.append((x_min, y_avg, x_max, y_avg))
            else:
                x_avg = int(np.mean(xs))
                y_min, y_max = min(ys), max(ys)
                final_lines.append((x_avg, y_min, x_avg, y_max))
    
    return final_lines

def regularize_board_lines(h_lines, v_lines, n_lines=19):
    """
    规整化线条
    """
    regularized_h_lines = []
    regularized_v_lines = []
    
    # 处理横线位置
    if h_lines:
        h_positions = sorted([(y1 + y2) / 2 for x1, y1, x2, y2 in h_lines])
        if len(h_positions) >= 2:
            total_h_span = h_positions[-1] - h_positions[0]
            ideal_h_spacing = total_h_span / (n_lines - 1) if n_lines > 1 else 0
            start_h_pos = h_positions[0]
            ideal_h_positions = [int(start_h_pos + i * ideal_h_spacing) for i in range(n_lines)]
        else:
            ideal_h_positions = h_positions
    else:
        ideal_h_positions = []
    
    # 处理竖线位置
    if v_lines:
        v_positions = sorted([(x1 + x2) / 2 for x1, y1, x2, y2 in v_lines])
        if len(v_positions) >= 2:
            total_v_span = v_positions[-1] - v_positions[0]
            ideal_v_spacing = total_v_span / (n_lines - 1) if n_lines > 1 else 0
            start_v_pos = v_positions[0]
            ideal_v_positions = [int(start_v_pos + i * ideal_v_spacing) for i in range(n_lines)]
        else:
            ideal_v_positions = v_positions
    else:
        ideal_v_positions = []
    
    # 生成规整的线条
    if ideal_h_positions and ideal_v_positions:
        x_min, x_max = min(ideal_v_positions), max(ideal_v_positions)
        for y_pos in ideal_h_positions:
            regularized_h_lines.append((x_min, y_pos, x_max, y_pos))
        
        y_min, y_max = min(ideal_h_positions), max(ideal_h_positions)
        for x_pos in ideal_v_positions:
            regularized_v_lines.append((x_pos, y_min, x_pos, y_max))
    
    return regularized_h_lines, regularized_v_lines

def print_line_analysis(h_lines, v_lines, title="线条分析"):
    """
    打印详细的线条分析信息
    """
    print(f"=== {title} ===")
    print(f"检测到横线数量: {len(h_lines)}")
    print(f"检测到竖线数量: {len(v_lines)}")
    
    if h_lines:
        h_positions = [(y1+y2)/2 for x1,y1,x2,y2 in h_lines]
        h_positions = sorted(h_positions)
        print(f"横线位置: {[int(pos) for pos in h_positions]}")
        if len(h_positions) > 1:
            spacings = [h_positions[i+1] - h_positions[i] for i in range(len(h_positions)-1)]
            print(f"横线间距: {[int(s) for s in spacings]}")
    
    if v_lines:
        v_positions = [(x1+x2)/2 for x1,y1,x2,y2 in v_lines]
        v_positions = sorted(v_positions)
        print(f"竖线位置: {[int(pos) for pos in v_positions]}")
        if len(v_positions) > 1:
            spacings = [v_positions[i+1] - v_positions[i] for i in range(len(v_positions)-1)]
            print(f"竖线间距: {[int(s) for s in spacings]}")
    print("=" * (len(title) + 8))

# 主程序
def main():
    # 读取图像
    #img = cv2.imread("../data/raw/bd317d54.webp")
    img = cv2.imread("../data/raw/Pic1.jpg")
    if img is None:
        print("无法读取图像文件")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 图像预处理 - 多种方法组合
    # 1. 高斯模糊
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. 直方图均衡化
    equalized = cv2.equalizeHist(blur)
    
    # 3. 可选：自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 显示原图和预处理结果
    cv2.imshow("Original", img)
    cv2.moveWindow("Original", 0, 0)
    
    cv2.imshow("Enhanced", enhanced)
    cv2.moveWindow("Enhanced", 400, 0)
    
    # 边缘检测 - 使用多个参数
    edges1 = cv2.Canny(equalized, 30, 120)  # 更低的阈值
    edges2 = cv2.Canny(enhanced, 40, 150)   # 标准阈值
    
    # 合并边缘
    edges_combined = cv2.bitwise_or(edges1, edges2)
    
    cv2.imshow("Edges Combined", edges_combined)
    cv2.moveWindow("Edges Combined", 800, 0)
    
    # 使用多阈值检测线条
    lines = detect_lines_multi_threshold(edges_combined, img.shape)
    
    if lines is None:
        print("未检测到任何线条")
        return
    
    print(f"原始检测到 {len(lines)} 条线段")
    
    # 去除重复线条
    unique_lines = remove_duplicate_lines(lines)
    print(f"去重后剩余 {len(unique_lines)} 条线段")
    
    # 转换为numpy数组格式
    lines_array = np.array([[line] for line in unique_lines])
    
    # 分类线条
    h_lines, v_lines = classify_lines_by_angle(lines_array, angle_thresh=15)
    
    print(f"分类后: 横线 {len(h_lines)} 条, 竖线 {len(v_lines)} 条")
    
    # 过滤边缘线条
    h_lines_filtered = filter_edge_lines(h_lines, img.shape, axis='h', margin_ratio=0.08)
    v_lines_filtered = filter_edge_lines(v_lines, img.shape, axis='v', margin_ratio=0.08)
    
    print(f"过滤边缘后: 横线 {len(h_lines_filtered)} 条, 竖线 {len(v_lines_filtered)} 条")
    
    # 自适应聚类
    N = 9
    merged_h = adaptive_clustering(h_lines_filtered, expected_count=N, axis='h', img_shape=img.shape)
    merged_v = adaptive_clustering(v_lines_filtered, expected_count=N, axis='v', img_shape=img.shape)
    
    # 创建显示图像
    img_detection = img.copy()
    img_regularized = img.copy()
    
    # 绘制检测结果
    for x1, y1, x2, y2 in merged_h + merged_v:
        cv2.line(img_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 打印分析
    print_line_analysis(merged_h, merged_v, "聚类后检测结果")
    
    # 规整化
    regularized_h, regularized_v = regularize_board_lines(merged_h, merged_v, N)
    
    # 绘制规整化结果
    for x1, y1, x2, y2 in regularized_h + regularized_v:
        cv2.line(img_regularized, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    print_line_analysis(regularized_h, regularized_v, "规整化后结果")
    
    # 显示结果
    cv2.imshow("Detection Result", img_detection)
    cv2.moveWindow("Detection Result", 0, 400)
    
    cv2.imshow("Regularized Result", img_regularized)
    cv2.moveWindow("Regularized Result", 400, 400)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()