# 这是Deepseek 给的第一版，他看过 Claude的 5-2 版本
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def classify_lines_by_angle(lines, angle_thresh=5):
    """更精确的线条角度分类"""
    horizontals, verticals = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0] if isinstance(line, np.ndarray) else line
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

def filter_edge_lines(line_group, img_shape, axis='h', margin_ratio=0.1):
    """更严格的边缘线条过滤"""
    if not line_group:
        return []
    
    height, width = img_shape[:2]
    margin_h = int(height * margin_ratio)
    margin_w = int(width * margin_ratio)
    
    filtered_lines = []
    
    for x1, y1, x2, y2 in line_group:
        if axis == 'h':
            y_avg = (y1 + y2) / 2
            if margin_h < y_avg < height - margin_h:
                line_length = abs(x2 - x1)
                if line_length >= width * 0.5:  # 要求横线至少跨越图像的50%宽度
                    filtered_lines.append((x1, y1, x2, y2))
        else:
            x_avg = (x1 + x2) / 2
            if margin_w < x_avg < width - margin_w:
                line_length = abs(y2 - y1)
                if line_length >= height * 0.5:  # 要求竖线至少跨越图像的50%高度
                    filtered_lines.append((x1, y1, x2, y2))
    
    return filtered_lines

def adaptive_clustering(line_group, expected_count, axis='h'):
    """改进的自适应聚类"""
    if not line_group or expected_count <= 0:
        return []
    
    # 获取坐标点
    coords = []
    for x1, y1, x2, y2 in line_group:
        coord = (y1 + y2) / 2 if axis == 'h' else (x1 + x2) / 2
        coords.append([coord])
    
    if len(coords) < 2:
        return []
    
    # 使用KMeans聚类
    n_clusters = min(expected_count, len(coords))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)
    
    # 生成最终线条
    final_lines = []
    for i in range(n_clusters):
        cluster_lines = [line for idx, line in enumerate(line_group) if labels[idx] == i]
        if cluster_lines:
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
    """改进的规整化方法，避免边缘偏移"""
    regularized_h = []
    regularized_v = []
    
    # 获取有效位置（排除可能的边缘线）
    if h_lines:
        h_positions = sorted([(y1 + y2) / 2 for x1, y1, x2, y2 in h_lines])
        if len(h_positions) >= 2:
            # 使用中间90%的位置来计算，避免边缘影响
            start_idx = int(len(h_positions) * 0.05)
            end_idx = int(len(h_positions) * 0.95)
            valid_h = h_positions[start_idx:end_idx]
            if valid_h:
                total_h_span = valid_h[-1] - valid_h[0]
                ideal_h_spacing = total_h_span / (n_lines - 1)
                start_h = valid_h[0] - ideal_h_spacing * start_idx
                ideal_h_positions = [int(start_h + i * ideal_h_spacing) for i in range(n_lines)]
    
    if v_lines:
        v_positions = sorted([(x1 + x2) / 2 for x1, y1, x2, y2 in v_lines])
        if len(v_positions) >= 2:
            # 使用中间90%的位置来计算
            start_idx = int(len(v_positions) * 0.05)
            end_idx = int(len(v_positions) * 0.95)
            valid_v = v_positions[start_idx:end_idx]
            if valid_v:
                total_v_span = valid_v[-1] - valid_v[0]
                ideal_v_spacing = total_v_span / (n_lines - 1)
                start_v = valid_v[0] - ideal_v_spacing * start_idx
                ideal_v_positions = [int(start_v + i * ideal_v_spacing) for i in range(n_lines)]
    
    # 生成规整线条
    if 'ideal_h_positions' in locals() and 'ideal_v_positions' in locals():
        x_min, x_max = min(ideal_v_positions), max(ideal_v_positions)
        for y_pos in ideal_h_positions:
            regularized_h.append((x_min, y_pos, x_max, y_pos))
        
        y_min, y_max = min(ideal_h_positions), max(ideal_h_positions)
        for x_pos in ideal_v_positions:
            regularized_v.append((x_pos, y_min, x_pos, y_max))
    
    return regularized_h, regularized_v

def main():
    # 读取图像
    img = cv2.imread("../data/raw/Pic1.jpg")
    if img is None:
        print("无法读取图像文件")
        return
    
    # 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 更强的模糊
    equalized = cv2.equalizeHist(blur)
    
    # 边缘检测 - 使用较高阈值减少噪声
    edges = cv2.Canny(equalized, 60, 180)
    
    # 线条检测 - 使用较高阈值和最小长度
    lines = cv2.HoughLinesP(edges, 
                           rho=1, 
                           theta=np.pi/180, 
                           threshold=80,
                           minLineLength=100,
                           maxLineGap=8)
    
    if lines is None:
        print("未检测到任何线条")
        return
    
    # 分类和过滤线条
    h_lines, v_lines = classify_lines_by_angle(lines)
    h_lines = filter_edge_lines(h_lines, img.shape, 'h', 0.1)
    v_lines = filter_edge_lines(v_lines, img.shape, 'v', 0.1)
    
    # 自适应聚类
    N = 19  # 19x19围棋棋盘
    merged_h = adaptive_clustering(h_lines, N, 'h')
    merged_v = adaptive_clustering(v_lines, N, 'v')
    
    # 规整化处理
    regularized_h, regularized_v = regularize_board_lines(merged_h, merged_v, N)
    
    # 绘制结果
    img_result = img.copy()
    for x1, y1, x2, y2 in regularized_h + regularized_v:
        cv2.line(img_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Edges", edges)
    cv2.moveWindow("Edges", 320, 0)
    cv2.imshow("Result", img_result)
    cv2.moveWindow("Result", 640, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()