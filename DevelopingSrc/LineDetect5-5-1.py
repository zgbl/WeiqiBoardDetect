# 这个版本是from Germini 2.5 pro, 基于 Claude版的 5-2-2的改进版。表现没有什么改进。
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

    # Check proximity of midpoints
    mid_x1, mid_y1 = (x1 + x2) / 2, (y1 + y2) / 2
    mid_x2, mid_y2 = (x3 + x4) / 2, (y3 + y4) / 2
    dist = np.hypot(mid_x1 - mid_x2, mid_y1 - mid_y2)
    
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
    使用多个阈值来检测线条，并进行初步过滤以减少数量
    """
    all_lines = []
    
    # 调整参数，进一步减少噪声线条
    # Higher thresholds mean only stronger lines are detected.
    # Longer min_line_length ensures only significant lines are considered.
    # Smaller max_line_gap prevents connecting unrelated segments.
    thresholds = [80, 90, 100]  # Increased thresholds
    min_lengths = [80, 100, 120]  # Increased minimum length requirement
    max_gaps = [5, 5, 5]  # Smaller max gap
    
    for threshold, min_length, max_gap in zip(thresholds, min_lengths, max_gaps):
        lines = cv2.HoughLinesP(edges, 
                               rho=1, 
                               theta=np.pi/180, 
                               threshold=threshold,
                               minLineLength=min_length,
                               maxLineGap=max_gap)
        
        if lines is not None:
            print(f"阈值{threshold}, 最小长度{min_length}, 最大间隙{max_gap}: 检测到{len(lines)}条线段")
            for line in lines:
                all_lines.append(line)
    
    if not all_lines:
        return None
    
    # Convert to a flat list of tuples (x1, y1, x2, y2)
    processed_lines = []
    for line in all_lines:
        x1, y1, x2, y2 = line[0]
        processed_lines.append((x1, y1, x2, y2))
        
    return processed_lines


def remove_duplicate_lines(lines, dist_thresh=10, angle_thresh=2):
    """
    去除重复和相似的线条 - 优化版本，考虑角度和距离
    """
    if not lines:
        return []

    # Sort lines to help with spatial locality
    # Sort by midpoint y then x for better processing of horizontal/vertical lines
    lines.sort(key=lambda line: ( (line[1] + line[3]) / 2, (line[0] + line[2]) / 2 ))

    unique_lines = []
    
    for i, current_line in enumerate(lines):
        is_duplicate = False
        
        # Only compare with lines already considered unique
        for unique_line in unique_lines:
            if is_similar(current_line, unique_line, dist_thresh=dist_thresh, angle_thresh=angle_thresh):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_lines.append(current_line)
            
    print(f"去重后剩余 {len(unique_lines)} 条线段")
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
    
    # Analyze spacing to determine reasonable number of clusters
    positions = [coord[0] for coord in coords]
    spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    
    if spacings:
        median_spacing = np.median(spacings)
        
        # Estimate number of lines based on median spacing and total span
        total_span = positions[-1] - positions[0]
        # Add a small value to total_span to avoid division by zero or too few clusters if spacing is large
        estimated_lines = int(total_span / (median_spacing + 1e-6)) + 1
        
        # Balance between estimated and expected count, but don't exceed actual lines
        n_clusters = min(max(estimated_lines, expected_count), len(coords))
        # Ensure at least 2 clusters if there are enough lines to cluster
        if n_clusters < 2 and len(coords) >= 2:
            n_clusters = 2
        elif n_clusters == 0 and len(coords) > 0: # Ensure at least 1 cluster if there's any line
            n_clusters = 1
    else:
        n_clusters = min(expected_count, len(coords))
        if n_clusters == 0 and len(coords) > 0:
            n_clusters = 1
    
    if n_clusters <= 1:
        # If clustering results in 1 or fewer clusters, return the group as is
        # Or, if clustering is not applicable, return lines
        if len(coords) > 0:
             # Just take the average line if only one cluster is formed or requested.
            xs, ys = [], []
            for x1, y1, x2, y2 in line_group:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
            
            if axis == 'h':
                y_avg = int(np.mean(ys))
                x_min, x_max = min(xs), max(xs)
                return [(x_min, y_avg, x_max, y_avg)]
            else:
                x_avg = int(np.mean(xs))
                y_min, y_max = min(ys), max(ys)
                return [(x_avg, y_min, x_avg, y_max)]
        return []

    # Execute clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)
    
    # Generate final lines
    final_lines = []
    for i in range(n_clusters):
        cluster_lines = [line for idx, line in enumerate(line_group) if labels[idx] == i]
        if cluster_lines:
            # Merge lines within the same cluster
            xs, ys = [], []
            for x1, y1, x2, y2 in cluster_lines:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
            
            if axis == 'h':
                y_avg = int(np.mean(ys))
                # Take the min and max x from the cluster lines to define the horizontal line span
                x_min_cluster = min([min(line[0], line[2]) for line in cluster_lines])
                x_max_cluster = max([max(line[0], line[2]) for line in cluster_lines])
                final_lines.append((x_min_cluster, y_avg, x_max_cluster, y_avg))
            else:
                x_avg = int(np.mean(xs))
                # Take the min and max y from the cluster lines to define the vertical line span
                y_min_cluster = min([min(line[1], line[3]) for line in cluster_lines])
                y_max_cluster = max([max(line[1], line[3]) for line in cluster_lines])
                final_lines.append((x_avg, y_min_cluster, x_avg, y_max_cluster))
    
    return final_lines

def regularize_board_lines(h_lines, v_lines, n_lines=19):
    """
    规整化线条
    """
    regularized_h_lines = []
    regularized_v_lines = []
    
    # Handle horizontal line positions
    if h_lines:
        h_positions = sorted([(y1 + y2) / 2 for x1, y1, x2, y2 in h_lines])
        if len(h_positions) >= 2:
            total_h_span = h_positions[-1] - h_positions[0]
            ideal_h_spacing = total_h_span / (n_lines - 1) if n_lines > 1 else 0
            start_h_pos = h_positions[0]
            ideal_h_positions = [int(start_h_pos + i * ideal_h_spacing) for i in range(n_lines)]
        else:
            ideal_h_positions = h_positions # If only one or no lines, use the existing
    else:
        ideal_h_positions = []
    
    # Handle vertical line positions
    if v_lines:
        v_positions = sorted([(x1 + x2) / 2 for x1, y1, x2, y2 in v_lines])
        if len(v_positions) >= 2:
            total_v_span = v_positions[-1] - v_positions[0]
            ideal_v_spacing = total_v_span / (n_lines - 1) if n_lines > 1 else 0
            start_v_pos = v_positions[0]
            ideal_v_positions = [int(start_v_pos + i * ideal_v_spacing) for i in range(n_lines)]
        else:
            ideal_v_positions = v_positions # If only one or no lines, use the existing
    else:
        ideal_v_positions = []
    
    # Generate regularized lines
    # Ensure both horizontal and vertical positions are available to form a grid
    if ideal_h_positions and ideal_v_positions:
        # Determine the overall bounding box for the regularized grid
        # Use the extent of the detected lines to define the grid boundaries
        all_h_xs = [coord for line in h_lines for coord in [line[0], line[2]]] if h_lines else []
        all_v_ys = [coord for line in v_lines for coord in [line[1], line[3]]] if v_lines else []

        # If we have detected lines, use their min/max extent
        grid_x_min = min(all_h_xs) if all_h_xs else 0
        grid_x_max = max(all_h_xs) if all_h_xs else img_shape[1]

        grid_y_min = min(all_v_ys) if all_v_ys else 0
        grid_y_max = max(all_v_ys) if all_v_ys else img_shape[0]

        # Use the spans from vertical lines for horizontal lines, and vice versa
        # This creates a perfect grid within the detected outer boundaries.
        for y_pos in ideal_h_positions:
            regularized_h_lines.append((grid_x_min, y_pos, grid_x_max, y_pos))
        
        for x_pos in ideal_v_positions:
            regularized_v_lines.append((x_pos, grid_y_min, x_pos, grid_y_max))
    
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
            # print(f"横线长度: {[int(np.sqrt((x2-x1)**2 + (y2-y1)**2)) for x1,y1,x2,y2 in h_lines]}")
    
    if v_lines:
        v_positions = [(x1+x2)/2 for x1,y1,x2,y2 in v_lines]
        v_positions = sorted(v_positions)
        print(f"竖线位置: {[int(pos) for pos in v_positions]}")
        if len(v_positions) > 1:
            spacings = [v_positions[i+1] - v_positions[i] for i in range(len(v_positions)-1)]
            print(f"竖线间距: {[int(s) for s in spacings]}")
            # print(f"竖线长度: {[int(np.sqrt((x2-x1)**2 + (y2-y1)**2)) for x1,y1,x2,y2 in v_lines]}")
    print("=" * (len(title) + 8))

# Main program
def main():
    # Read image
    img = cv2.imread("../data/raw/bd317d54.webp")
    if img is None:
        print("无法读取图像文件")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Image preprocessing - combination of methods
    # 1. Gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. Histogram equalization
    equalized = cv2.equalizeHist(blur)
    
    # 3. Optional: Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray) # Apply CLAHE to original gray, or blur. Let's try gray.
    
    # Display original and preprocessed results
    cv2.imshow("Original", img)
    cv2.moveWindow("Original", 0, 0)
    
    cv2.imshow("Enhanced (CLAHE)", enhanced)
    cv2.moveWindow("Enhanced (CLAHE)", 400, 0)
    
    cv2.imshow("Equalized (Hist)", equalized)
    cv2.moveWindow("Equalized (Hist)", 800, 0)

    # Edge detection - use stricter parameters
    # Higher thresholds to get fewer, but stronger edges.
    edges1 = cv2.Canny(equalized, 50, 150)  # Stricter thresholds
    edges2 = cv2.Canny(enhanced, 60, 180)   # Even stricter thresholds
    
    # Combine edges
    edges_combined = cv2.bitwise_or(edges1, edges2)
    
    cv2.imshow("Edges Combined (Stricter Canny)", edges_combined)
    cv2.moveWindow("Edges Combined (Stricter Canny)", 1200, 0)
    
    # Detect lines with multiple thresholds and initial filtering
    lines = detect_lines_multi_threshold(edges_combined, img.shape)
    
    if lines is None:
        print("未检测到任何线条")
        return
    
    print(f"原始检测到 {len(lines)} 条线段 (来自Hough)")
    
    # Remove duplicate lines (optimized)
    unique_lines = remove_duplicate_lines(lines, dist_thresh=10, angle_thresh=2) # Slightly larger dist_thresh, tighter angle_thresh
    
    # Convert to numpy array format for classify_lines_by_angle
    # Ensure lines are in the format expected by classify_lines_by_angle
    lines_array_for_classify = np.array([[[x1, y1, x2, y2]] for (x1, y1, x2, y2) in unique_lines])
    
    # Classify lines
    h_lines, v_lines = classify_lines_by_angle(lines_array_for_classify, angle_thresh=10) # Tighter angle for classification
    
    print(f"分类后: 横线 {len(h_lines)} 条, 竖线 {len(v_lines)} 条")
    
    # Filter edge lines more strictly
    h_lines_filtered = filter_edge_lines(h_lines, img.shape, axis='h', margin_ratio=0.1) # Increased margin
    v_lines_filtered = filter_edge_lines(v_lines, img.shape, axis='v', margin_ratio=0.1) # Increased margin
    
    print(f"过滤边缘后: 横线 {len(h_lines_filtered)} 条, 竖线 {len(v_lines_filtered)} 条")
    
    # Adaptive clustering
    N = 19 # Assuming 19x19 Go board
    merged_h = adaptive_clustering(h_lines_filtered, expected_count=N, axis='h', img_shape=img.shape)
    merged_v = adaptive_clustering(v_lines_filtered, expected_count=N, axis='v', img_shape=img.shape)
    
    # Create display images
    img_detection = img.copy()
    img_regularized = img.copy()
    
    # Draw detected and merged lines
    for x1, y1, x2, y2 in merged_h + merged_v:
        cv2.line(img_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Print analysis
    print_line_analysis(merged_h, merged_v, "聚类后检测结果")
    
    # Regularize
    regularized_h, regularized_v = regularize_board_lines(merged_h, merged_v, N)
    
    # Draw regularized results
    for x1, y1, x2, y2 in regularized_h + regularized_v:
        cv2.line(img_regularized, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    print_line_analysis(regularized_h, regularized_v, "规整化后结果")
    
    # Display results
    cv2.imshow("Detection Result (Clustered)", img_detection)
    cv2.moveWindow("Detection Result (Clustered)", 0, 500)
    
    cv2.imshow("Regularized Result", img_regularized)
    cv2.moveWindow("Regularized Result", 400, 500)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()