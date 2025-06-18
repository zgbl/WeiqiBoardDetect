# 改进的真实棋盘线条检测
# 不依赖边缘坐标，专门针对真实物理棋盘
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import math

def enhance_board_image(img):
    """
    增强图像，突出棋盘线条
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 轻微高斯模糊减少噪声
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return blurred

def detect_primary_lines(img, min_line_length_ratio=0.4):
    """
    检测主要线条，使用多种参数组合
    """
    edges = cv2.Canny(img, 30, 100, apertureSize=3)
    
    height, width = img.shape
    min_line_length = int(min(width, height) * min_line_length_ratio)
    
    # 使用多组参数检测线条
    all_lines = []
    
    # 参数组合1: 较严格，检测主要线条
    lines1 = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(min_line_length*0.8), 
                            minLineLength=min_line_length, maxLineGap=20)
    if lines1 is not None:
        all_lines.extend(lines1)
    
    # 参数组合2: 较宽松，检测更多可能的线条
    lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(min_line_length*0.6),
                            minLineLength=int(min_line_length*0.7), maxLineGap=30)
    if lines2 is not None:
        all_lines.extend(lines2)
    
    return all_lines, edges

def calculate_line_angle(x1, y1, x2, y2):
    """计算线条角度（度）"""
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    # 规范化到 [-90, 90]
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle

def classify_lines_by_orientation(lines, angle_threshold=15):
    """
    根据角度分类线条为水平和垂直
    """
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = calculate_line_angle(x1, y1, x2, y2)
        
        # 水平线: 角度接近0度
        if abs(angle) <= angle_threshold:
            horizontal_lines.append((x1, y1, x2, y2))
        # 垂直线: 角度接近90度或-90度
        elif abs(abs(angle) - 90) <= angle_threshold:
            vertical_lines.append((x1, y1, x2, y2))
    
    return horizontal_lines, vertical_lines

def merge_similar_lines(lines, axis='horizontal', distance_threshold=15):
    """
    合并相似的线条
    """
    if not lines:
        return []
    
    # 根据轴向选择排序和分组的坐标
    if axis == 'horizontal':
        # 水平线按y坐标分组
        key_func = lambda line: (line[1] + line[3]) / 2  # 平均y坐标
    else:
        # 垂直线按x坐标分组
        key_func = lambda line: (line[0] + line[2]) / 2  # 平均x坐标
    
    # 按关键坐标排序
    sorted_lines = sorted(lines, key=key_func)
    
    merged_lines = []
    current_group = [sorted_lines[0]]
    
    for i in range(1, len(sorted_lines)):
        current_key = key_func(sorted_lines[i])
        prev_key = key_func(current_group[-1])
        
        if abs(current_key - prev_key) <= distance_threshold:
            # 相似线条，加入当前组
            current_group.append(sorted_lines[i])
        else:
            # 不同组，处理当前组并开始新组
            merged_line = merge_line_group(current_group, axis)
            if merged_line:
                merged_lines.append(merged_line)
            current_group = [sorted_lines[i]]
    
    # 处理最后一组
    merged_line = merge_line_group(current_group, axis)
    if merged_line:
        merged_lines.append(merged_line)
    
    return merged_lines

def merge_line_group(line_group, axis):
    """
    合并一组相似线条为单条线
    """
    if not line_group:
        return None
    
    # 收集所有端点
    all_points = []
    for x1, y1, x2, y2 in line_group:
        all_points.extend([(x1, y1), (x2, y2)])
    
    if axis == 'horizontal':
        # 水平线：固定y坐标，取x的范围
        y_coords = [p[1] for p in all_points]
        y_avg = int(np.mean(y_coords))
        
        x_coords = [p[0] for p in all_points]
        x_min, x_max = min(x_coords), max(x_coords)
        
        return (x_min, y_avg, x_max, y_avg)
    else:
        # 垂直线：固定x坐标，取y的范围
        x_coords = [p[0] for p in all_points]
        x_avg = int(np.mean(x_coords))
        
        y_coords = [p[1] for p in all_points]
        y_min, y_max = min(y_coords), max(y_coords)
        
        return (x_avg, y_min, x_avg, y_max)

def filter_lines_by_length(lines, img_shape, axis='horizontal', min_length_ratio=0.3):
    """
    根据长度过滤线条
    """
    height, width = img_shape[:2]
    
    if axis == 'horizontal':
        min_length = width * min_length_ratio
        length_func = lambda line: abs(line[2] - line[0])
    else:
        min_length = height * min_length_ratio
        length_func = lambda line: abs(line[3] - line[1])
    
    return [line for line in lines if length_func(line) >= min_length]

def cluster_lines_kmeans(lines, n_clusters, axis='horizontal'):
    """
    使用K-means聚类线条
    """
    if not lines or n_clusters <= 0:
        return []
    
    # 提取位置坐标
    if axis == 'horizontal':
        positions = [(line[1] + line[3]) / 2 for line in lines]  # y坐标
    else:
        positions = [(line[0] + line[2]) / 2 for line in lines]  # x坐标
    
    if len(set(positions)) < n_clusters:
        # 如果唯一位置少于聚类数，直接返回去重结果
        unique_positions = sorted(set(positions))
        result_lines = []
        
        for pos in unique_positions:
            # 找到该位置的所有线条
            matching_lines = [line for line in lines 
                            if abs(((line[1] + line[3]) / 2 if axis == 'horizontal' 
                                  else (line[0] + line[2]) / 2) - pos) < 1]
            if matching_lines:
                merged = merge_line_group(matching_lines, axis)
                if merged:
                    result_lines.append(merged)
        
        return result_lines
    
    # 执行K-means聚类
    positions_array = np.array(positions).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(positions_array)
    
    # 为每个聚类生成代表线条
    clustered_lines = []
    for i in range(n_clusters):
        cluster_lines = [lines[j] for j in range(len(lines)) if labels[j] == i]
        if cluster_lines:
            merged = merge_line_group(cluster_lines, axis)
            if merged:
                clustered_lines.append(merged)
    
    return clustered_lines

def generate_regular_grid(h_lines, v_lines, n_lines=19):
    """
    基于检测到的线条生成规则网格
    """
    if not h_lines or not v_lines:
        return [], []
    
    # 获取边界
    h_positions = sorted([(line[1] + line[3]) / 2 for line in h_lines])
    v_positions = sorted([(line[0] + line[2]) / 2 for line in v_lines])
    
    # 计算网格范围
    y_min, y_max = h_positions[0], h_positions[-1]
    x_min, x_max = v_positions[0], v_positions[-1]
    
    # 生成等间距网格
    regular_h_lines = []
    regular_v_lines = []
    
    # 生成19条水平线
    for i in range(n_lines):
        y_pos = int(y_min + i * (y_max - y_min) / (n_lines - 1))
        regular_h_lines.append((int(x_min), y_pos, int(x_max), y_pos))
    
    # 生成19条垂直线
    for i in range(n_lines):
        x_pos = int(x_min + i * (x_max - x_min) / (n_lines - 1))
        regular_v_lines.append((x_pos, int(y_min), x_pos, int(y_max)))
    
    return regular_h_lines, regular_v_lines

def analyze_line_distribution(lines, axis='horizontal'):
    """
    分析线条分布，帮助调试
    """
    if not lines:
        return {}
    
    if axis == 'horizontal':
        positions = [(line[1] + line[3]) / 2 for line in lines]
        lengths = [abs(line[2] - line[0]) for line in lines]
    else:
        positions = [(line[0] + line[2]) / 2 for line in lines]
        lengths = [abs(line[3] - line[1]) for line in lines]
    
    positions = sorted(positions)
    spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    
    return {
        'count': len(lines),
        'positions': positions,
        'lengths': lengths,
        'avg_length': np.mean(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'spacings': spacings,
        'avg_spacing': np.mean(spacings) if spacings else 0,
        'spacing_std': np.std(spacings) if spacings else 0
    }

def detect_board_lines(img_path, target_lines=19, debug=True):
    """
    主函数：检测棋盘线条
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None, None, None
    
    if debug:
        print(f"图像尺寸: {img.shape}")
    
    # 图像增强
    enhanced = enhance_board_image(img)

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    edges = cv2.Canny(blur, 50, 150)
    cv2.imshow('edges', edges)

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

    # 找到最小外接矩形，即棋盘的四个角点
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # 绘制轮廓和角点
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
    for point in box:
        cv2.circle(img, tuple(point), 5, (0, 255, 0), -1)
    width = int(rect[1][0])
    height = int(rect[1][1])

    
    # 检测线条
    all_lines, edges = detect_primary_lines(enhanced)
    
    if not all_lines:
        print("未检测到任何线条")
        return img, [], []
    
    if debug:
        print(f"原始检测到 {len(all_lines)} 条线段")
    
    # 分类线条
    h_lines_raw, v_lines_raw = classify_lines_by_orientation(all_lines)
    
    if debug:
        print(f"原始分类: {len(h_lines_raw)} 条水平线, {len(v_lines_raw)} 条垂直线")
    
    # 合并相似线条
    h_lines_merged = merge_similar_lines(h_lines_raw, 'horizontal')
    v_lines_merged = merge_similar_lines(v_lines_raw, 'vertical')
    
    if debug:
        print(f"合并后: {len(h_lines_merged)} 条水平线, {len(v_lines_merged)} 条垂直线")
    
    # 过滤短线条
    h_lines_filtered = filter_lines_by_length(h_lines_merged, img.shape, 'horizontal')
    v_lines_filtered = filter_lines_by_length(v_lines_merged, img.shape, 'vertical')
    
    if debug:
        print(f"过滤后: {len(h_lines_filtered)} 条水平线, {len(v_lines_filtered)} 条垂直线")
        
        # 分析线条分布
        if h_lines_filtered:
            h_analysis = analyze_line_distribution(h_lines_filtered, 'horizontal')
            print(f"水平线分析: 平均长度={h_analysis['avg_length']:.1f}, 平均间距={h_analysis['avg_spacing']:.1f}")
        
        if v_lines_filtered:
            v_analysis = analyze_line_distribution(v_lines_filtered, 'vertical')
            print(f"垂直线分析: 平均长度={v_analysis['avg_length']:.1f}, 平均间距={v_analysis['avg_spacing']:.1f}")
    
    # 如果检测到的线条数量接近目标，尝试聚类
    final_h_lines = h_lines_filtered
    final_v_lines = v_lines_filtered
    
    if len(h_lines_filtered) > target_lines * 0.7:
        clustered_h = cluster_lines_kmeans(h_lines_filtered, target_lines, 'horizontal')
        if len(clustered_h) >= target_lines * 0.8:
            final_h_lines = clustered_h
    
    if len(v_lines_filtered) > target_lines * 0.7:
        clustered_v = cluster_lines_kmeans(v_lines_filtered, target_lines, 'vertical')
        if len(clustered_v) >= target_lines * 0.8:
            final_v_lines = clustered_v
    
    if debug:
        print(f"最终结果: {len(final_h_lines)} 条水平线, {len(final_v_lines)} 条垂直线")
    
    return img, final_h_lines, final_v_lines

def draw_lines(img, h_lines, v_lines, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制线条
    """
    result = img.copy()
    
    for line in h_lines + v_lines:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    
    return result

# 主程序示例
if __name__ == "__main__":
    # 修改为您的图像路径
    img_path = "../data/raw/bd317d54.webp"
    #img_path = "../data/raw/OGS4.jpg"
    
    # 检测棋盘线条
    original_img, h_lines, v_lines = detect_board_lines(img_path, target_lines=19, debug=True)
    
    if original_img is not None:
        # 绘制检测结果
        detected_img = draw_lines(original_img, h_lines, v_lines, (0, 255, 0), 2)
        
        # 生成规则网格（如果检测效果不理想）
        if h_lines and v_lines:
            regular_h, regular_v = generate_regular_grid(h_lines, v_lines, 19)
            regular_img = draw_lines(original_img, regular_h, regular_v, (0, 0, 255), 2)
            
            # 显示结果
            cv2.imshow("Original", original_img)
            cv2.moveWindow("Original", 0, 0)
            
            cv2.imshow("Detected Lines", detected_img)
            cv2.moveWindow("Detected Lines", 400, 0)
            
            cv2.imshow("Regular Grid", regular_img)
            cv2.moveWindow("Regular Grid", 800, 0)
            
            print("\n按任意键退出...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("检测失败，未找到足够的线条")
    else:
        print("图像加载失败")