# 这个版本不行。抛弃。
import cv2
import numpy as np
from collections import defaultdict
import math

def detect_board_corners(img):
    """
    检测棋盘的四个角点
    返回: [(左上), (左下), (右上), (右下)] 或 None
    """
    print("开始检测棋盘角点...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 自适应阈值处理，突出棋盘线条
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    
    # 2. 形态学操作，去除噪声但保留线条结构
    kernel = np.ones((3,3), np.uint8)
    morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    
    # 3. 边缘检测 - 使用较保守的参数避免过多噪声
    edges = cv2.Canny(morphed, 50, 150, apertureSize=3)
    
    # 4. 找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未找到任何轮廓")
        return None
    
    # 5. 筛选可能的棋盘轮廓
    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * 0.1  # 棋盘至少占图片10%的面积
    max_area = img_area * 0.9  # 棋盘最多占图片90%的面积
    
    board_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # 近似轮廓为多边形
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 寻找四边形
            if len(approx) == 4:
                # 检查是否是凸四边形
                if cv2.isContourConvex(approx):
                    board_candidates.append((contour, approx, area))
    
    if not board_candidates:
        print("未找到四边形棋盘轮廓")
        return None
    
    # 6. 选择最大的凸四边形作为棋盘
    board_candidates.sort(key=lambda x: x[2], reverse=True)  # 按面积排序
    best_contour, best_approx, best_area = board_candidates[0]
    
    print(f"找到棋盘轮廓，面积: {best_area}")
    
    # 7. 提取四个角点并排序
    corners = best_approx.reshape(4, 2)
    
    # 计算重心
    center = np.mean(corners, axis=0)
    
    # 根据相对于重心的位置对角点进行分类
    def classify_corner(point, center):
        if point[0] < center[0] and point[1] < center[1]:
            return 0  # 左上
        elif point[0] < center[0] and point[1] > center[1]:
            return 1  # 左下
        elif point[0] > center[0] and point[1] < center[1]:
            return 2  # 右上
        else:
            return 3  # 右下
    
    # 按照 [左上, 左下, 右上, 右下] 的顺序排列
    sorted_corners = [None] * 4
    for corner in corners:
        idx = classify_corner(corner, center)
        sorted_corners[idx] = corner
    
    # 如果有任何角点为None，使用距离排序作为备选方案
    if None in sorted_corners:
        print("使用距离排序方法重新排列角点")
        # 按到左上角的距离排序
        corners = sorted(corners, key=lambda p: p[0] + p[1])
        
        # 重新分类
        sorted_corners = [None] * 4
        sorted_corners[0] = corners[0]  # 左上角（距离原点最近）
        
        # 剩余三个点中，找左下角（x最小的）
        remaining = corners[1:]
        remaining.sort(key=lambda p: p[0])
        sorted_corners[1] = remaining[0]  # 左下角
        
        # 在剩余两个点中，y较小的是右上角
        if remaining[1][1] < remaining[2][1]:
            sorted_corners[2] = remaining[1]  # 右上角
            sorted_corners[3] = remaining[2]  # 右下角
        else:
            sorted_corners[2] = remaining[2]  # 右上角
            sorted_corners[3] = remaining[1]  # 右下角
    
    return sorted_corners

def auto_resize_image(image, target_width=1500):
    """自动缩放图片到合适的尺寸进行处理"""
    height, width = image.shape[:2]
    print(f"原图尺寸: {width} x {height}")
    
    if width > 2500:
        scale_factor = int((width * 10 / target_width))
        new_width = width * 10 // scale_factor
        new_height = height * 10 // scale_factor
        
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"缩放后尺寸: {new_width} x {new_height}")
        return resized_img, scale_factor
    else:
        print("图片尺寸合适，无需缩放")
        return image, 1.0

def get_dynamic_params(width):
    """根据图片宽度获取动态参数"""
    if width <= 1000:
        return {
            'perspective_size': 400, 'perspective_corner': 390,
            'hough_threshold': 40, 'min_line_length': 50,
            'max_line_gap': 8, 'line_filter_length': 25,
            'merge_tolerance': 8, 'row_tolerance': 12,
            'min_dist_threshold': 12, 'min_radius': 4,
            'max_radius': 15, 'min_dist_circles': 12,
            'min_points_per_row': 8, 'intersection_threshold': 60
        }
    elif width <= 1500:
        return {
            'perspective_size': 500, 'perspective_corner': 490,
            'hough_threshold': 60, 'min_line_length': 70,
            'max_line_gap': 10, 'line_filter_length': 35,
            'merge_tolerance': 12, 'row_tolerance': 15,
            'min_dist_threshold': 15, 'min_radius': 6,
            'max_radius': 20, 'min_dist_circles': 15,
            'min_points_per_row': 12, 'intersection_threshold': 80
        }
    else:
        return {
            'perspective_size': 660, 'perspective_corner': 650,
            'hough_threshold': 80, 'min_line_length': 100,
            'max_line_gap': 10, 'line_filter_length': 50,
            'merge_tolerance': 15, 'row_tolerance': 20,
            'min_dist_threshold': 20, 'min_radius': 8,
            'max_radius': 25, 'min_dist_circles': 20,
            'min_points_per_row': 15, 'intersection_threshold': 100
        }

def detect_lines_and_grid(img, params):
    """检测棋盘线条和网格交点"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测 - 不使用dilate，保持边缘精确
    edges = cv2.Canny(blur, 50, 150)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           threshold=params['hough_threshold'],
                           minLineLength=params['min_line_length'],
                           maxLineGap=params['max_line_gap'])
    
    if lines is None:
        return None, None, None
    
    # 分离和过滤线条
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if length > params['line_filter_length']:
            if angle < 10 or angle > 170:  # 水平线
                horizontal_lines.append(line[0])
            elif 80 < angle < 100:  # 垂直线
                vertical_lines.append(line[0])
    
    # 合并相近的平行线
    def merge_lines(lines, is_horizontal=True):
        if not lines:
            return []
        
        merged = []
        tolerance = params['merge_tolerance']
        
        if is_horizontal:
            lines.sort(key=lambda line: (line[1] + line[3]) / 2)
        else:
            lines.sort(key=lambda line: (line[0] + line[2]) / 2)
        
        current_group = [lines[0]]
        
        for i in range(1, len(lines)):
            if is_horizontal:
                current_pos = (current_group[-1][1] + current_group[-1][3]) / 2
                new_pos = (lines[i][1] + lines[i][3]) / 2
            else:
                current_pos = (current_group[-1][0] + current_group[-1][2]) / 2
                new_pos = (lines[i][0] + lines[i][2]) / 2
            
            if abs(new_pos - current_pos) < tolerance:
                current_group.append(lines[i])
            else:
                if current_group:
                    avg_line = np.mean(current_group, axis=0).astype(int)
                    merged.append(avg_line)
                current_group = [lines[i]]
        
        if current_group:
            avg_line = np.mean(current_group, axis=0).astype(int)
            merged.append(avg_line)
        
        return merged
    
    merged_horizontal = merge_lines(horizontal_lines, True)
    merged_vertical = merge_lines(vertical_lines, False)
    
    return merged_horizontal, merged_vertical, edges

def main():
    # 读取图像
    img_path = '../data/raw/IMG20160706171004-16.jpg'  # 根据需要修改路径
    img = cv2.imread(img_path)
    
    if img is None:
        print("没找到照片")
        return
    
    # 缩放图片
    img, scale_factor = auto_resize_image(img, target_width=1500)
    original_img = img.copy()
    
    # 获取动态参数
    params = get_dynamic_params(img.shape[1])
    print(f"使用参数集: {params}")
    
    cv2.imshow("Original Image", img)
    
    # 检测棋盘角点
    rect = detect_board_corners(img)
    
    if rect is None:
        print('未能检测到棋盘角点！使用原图进行后续处理...')
        processed_img = original_img
    else:
        print('检测到棋盘角点:')
        lt, lb, rt, rb = rect
        print(f'\t左上角: {lt}')
        print(f'\t左下角: {lb}')
        print(f'\t右上角: {rt}')
        print(f'\t右下角: {rb}')
        
        # 显示检测到的角点
        corner_img = img.copy()
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255)]  # 绿, 黄, 蓝, 紫
        labels = ['LT', 'LB', 'RT', 'RB']
        
        for i, (point, color, label) in enumerate(zip(rect, colors, labels)):
            cv2.circle(corner_img, tuple(point), 15, color, -1)
            cv2.putText(corner_img, label, (point[0]-10, point[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Detected Corners', corner_img)
        
        # 执行透视变换
        margin = 20
        target_size = params['perspective_size'] - 2 * margin
        
        # 目标坐标
        pts1 = np.float32([
            [margin, margin],                           # 左上
            [margin, margin + target_size],             # 左下
            [margin + target_size, margin],             # 右上
            [margin + target_size, margin + target_size]  # 右下
        ])
        
        # 源坐标
        pts2 = np.float32([lt, lb, rt, rb])
        
        # 生成透视矩阵并变换
        m = cv2.getPerspectiveTransform(pts2, pts1)
        processed_img = cv2.warpPerspective(original_img, m, 
                                          (params['perspective_size'], params['perspective_size']))
        
        cv2.imshow('Perspective Corrected', processed_img)
        print("透视变换完成！")
    
    # 检测线条和网格
    h_lines, v_lines, edges = detect_lines_and_grid(processed_img, params)
    
    if h_lines is None:
        print("未检测到棋盘线条")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    print(f"检测到水平线: {len(h_lines)} 条")
    print(f"检测到垂直线: {len(v_lines)} 条")
    
    # 显示检测结果
    cv2.imshow('Edges', edges)
    
    # 绘制检测到的线条
    lines_img = processed_img.copy()
    for line in h_lines:
        cv2.line(lines_img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
    for line in v_lines:
        cv2.line(lines_img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
    
    cv2.imshow('Detected Lines', lines_img)
    
    print("处理完成！按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()