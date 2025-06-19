# 增强版本：提高鲁棒性，支持高分辨率图片和更稳定的边框检测
import cv2
import numpy as np
from collections import defaultdict
import math

def resize_image_if_needed(img, max_width=1200, max_height=1200):
    """
    如果图片太大，按比例缩小到合适尺寸
    返回缩放后的图片和缩放比例
    """
    height, width = img.shape[:2]
    scale = 1.0
    
    if width > max_width or height > max_height:
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"图片从 {width}x{height} 缩放到 {new_width}x{new_height}, 缩放比例: {scale:.3f}")
        return img_resized, scale
    
    return img, scale

def enhance_board_detection(img):
    """
    增强棋盘边框检测的预处理
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 自适应直方图均衡化，增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 多尺度高斯模糊
    blur1 = cv2.GaussianBlur(enhanced, (3,3), 0)
    blur2 = cv2.GaussianBlur(enhanced, (5,5), 0)
    
    return enhanced, blur1, blur2

def find_board_contour_robust(img):
    """
    更鲁棒的棋盘轮廓检测
    """
    enhanced, blur1, blur2 = enhance_board_detection(img)
    
    best_rect = None
    best_area = 0
    
    # 尝试多组参数进行边缘检测
    edge_params = [
        (30, 50),   # 原始参数
        (50, 100),  # 中等阈值
        (20, 60),   # 低阈值，检测更多边缘
        (40, 120),  # 高阈值，检测强边缘
        (25, 75),   # 平衡参数
    ]
    
    for i, (low, high) in enumerate(edge_params):
        # 对不同的模糊版本尝试边缘检测
        for j, blur_img in enumerate([blur1, blur2]):
            edges = cv2.Canny(blur_img, low, high)
            
            # 形态学处理，连接断裂的边缘
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 分析每个轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 面积筛选：应该是图片的一个合理比例
                img_area = img.shape[0] * img.shape[1]
                if area < img_area * 0.1 or area > img_area * 0.95:
                    continue
                
                # 寻找凸包
                hull = cv2.convexHull(contour)
                
                # 多种epsilon值尝试多边形拟合
                perimeter = cv2.arcLength(hull, True)
                for epsilon_factor in [0.02, 0.05, 0.08, 0.12, 0.15]:
                    epsilon = epsilon_factor * perimeter
                    approx = cv2.approxPolyDP(hull, epsilon, True)
                    
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        # 检查四边形的形状是否合理
                        if is_reasonable_quadrilateral(approx):
                            if area > best_area:
                                best_area = area
                                best_rect = process_quadrilateral(approx)
                                print(f"找到更好的四边形: 参数组{i+1}-{j+1}, epsilon={epsilon_factor}, 面积={area}")
            
            # 显示当前尝试的边缘检测结果（用于调试）
            if i == 0 and j == 0:  # 只显示第一组参数的结果
                cv2.imshow(f'edges_debug', edges)
    
    return best_rect

def is_reasonable_quadrilateral(approx):
    """
    检查四边形是否合理（不要太扁或太奇怪）
    """
    points = np.reshape(approx, (4, 2))
    
    # 计算四条边的长度
    edges = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        edge_length = np.linalg.norm(p2 - p1)
        edges.append(edge_length)
    
    # 检查边长比例是否合理
    min_edge = min(edges)
    max_edge = max(edges)
    
    if min_edge == 0 or max_edge / min_edge > 5:  # 最长边不应该超过最短边的5倍
        return False
    
    # 计算面积
    area = cv2.contourArea(approx)
    if area < 1000:  # 面积太小
        return False
    
    return True

def process_quadrilateral(approx):
    """
    处理四边形，确保角点顺序正确
    """
    points = np.reshape(approx, (4, 2))
    
    # 按x坐标排序
    points = points[np.lexsort((points[:,0],))]
    
    # 左侧两点按y坐标排序
    left_points = points[:2]
    left_points = left_points[np.lexsort((left_points[:,1],))]
    lt, lb = left_points
    
    # 右侧两点按y坐标排序
    right_points = points[2:]
    right_points = right_points[np.lexsort((right_points[:,1],))]
    rt, rb = right_points
    
    return (lt, lb, rt, rb)

# 读取图像
#img = cv2.imread('../data/raw/bd317d54.webp')
#img = cv2.imread('../data/raw/IMG20171015161921.jpg')
#img = cv2.imread('../data/raw/OGS3.jpeg')
img = cv2.imread('../data/raw/IMG20160706171004.jpg')

if img is None:
    print("没找到照片")
    exit()

original_img = img.copy()
original_scale = 1.0

# 显示原图信息
height, width = img.shape[:2]
print(f"原图尺寸: {width}x{height}")

# 1. 先缩放图片到合适尺寸
img, scale_factor = resize_image_if_needed(img, max_width=1200, max_height=1200)
if scale_factor != 1.0:
    original_scale = scale_factor

cv2.imshow("resized picture", img)

print("开始透视变换...")

# ====== 增强版透视变换部分 ======
rect = find_board_contour_robust(img)

# 检查是否找到棋盘
if rect is None:
    print('在图像文件中找不到棋盘！使用原图进行后续处理...')
    # 如果没找到棋盘轮廓，使用原图
    img_for_grid = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    print('棋盘坐标：')
    print('\t左上角：(%d,%d)'%(rect[0][0],rect[0][1]))
    print('\t左下角：(%d,%d)'%(rect[1][0],rect[1][1]))
    print('\t右上角：(%d,%d)'%(rect[2][0],rect[2][1]))
    print('\t右下角：(%d,%d)'%(rect[3][0],rect[3][1]))

    # 显示找到的角点
    corner_img = img.copy()
    for i, p in enumerate(rect):
        cv2.circle(corner_img,(p[0],p[1]),15,(0,255,0),-1)
        # 标注角点编号
        cv2.putText(corner_img, f"{i+1}", (p[0]-5, p[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow('Found Corners', corner_img)

    # 执行透视变换
    lt, lb, rt, rb = rect
    pts2 = np.float32([lt, lb, rt, rb])
    
    # 动态计算目标尺寸，保持棋盘的宽高比
    # 计算原四边形的平均宽度和高度
    width1 = np.linalg.norm(np.array(rt) - np.array(lt))
    width2 = np.linalg.norm(np.array(rb) - np.array(lb))
    height1 = np.linalg.norm(np.array(lb) - np.array(lt))
    height2 = np.linalg.norm(np.array(rb) - np.array(rt))
    
    avg_width = (width1 + width2) / 2
    avg_height = (height1 + height2) / 2
    
    # 使用合适的目标尺寸
    target_size = min(660, max(int(avg_width), int(avg_height)))
    margin = 10
    
    pts1 = np.float32([(margin, margin), (margin, target_size-margin), 
                       (target_size-margin, margin), (target_size-margin, target_size-margin)])
    
    m = cv2.getPerspectiveTransform(pts2, pts1)
    
    # 对图像执行透视变换
    img_for_grid = cv2.warpPerspective(img, m, (target_size, target_size))
    gray = cv2.cvtColor(img_for_grid, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Perspective Corrected', img_for_grid)
    print("透视变换完成！")

# ====== 棋盘线检测部分 (优化参数) ======
print("开始棋盘线检测...")

# 自适应参数调整
img_size = max(img_for_grid.shape[:2])
if img_size > 800:
    blur_kernel = (7, 7)
    canny_low, canny_high = 40, 120
    hough_threshold = int(img_size * 0.15)
    min_line_length = int(img_size * 0.15)
    max_line_gap = int(img_size * 0.02)
elif img_size > 400:
    blur_kernel = (5, 5)
    canny_low, canny_high = 50, 150
    hough_threshold = int(img_size * 0.12)
    min_line_length = int(img_size * 0.12)
    max_line_gap = int(img_size * 0.015)
else:
    blur_kernel = (3, 3)
    canny_low, canny_high = 30, 100
    hough_threshold = int(img_size * 0.1)
    min_line_length = int(img_size * 0.1)
    max_line_gap = int(img_size * 0.01)

print(f"自适应参数: 图片尺寸={img_size}, 霍夫阈值={hough_threshold}, 最小线长={min_line_length}")

# 高斯模糊
blur = cv2.GaussianBlur(gray, blur_kernel, 0)

# 边缘检测
edges = cv2.Canny(blur, canny_low, canny_high)
cv2.imshow('edges', edges)

# 使用霍夫变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                       threshold=hough_threshold, 
                       minLineLength=min_line_length, 
                       maxLineGap=max_line_gap)

if lines is None:
    print("未检测到任何直线")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

print(f"检测到 {len(lines)} 条直线")

# 创建一个副本用于绘制检测到的线条
lines_img = img_for_grid.copy()

# 分离水平线和垂直线
horizontal_lines = []
vertical_lines = []

def line_angle(line):
    x1, y1, x2, y2 = line
    return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

def line_length(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 过滤和分类直线
min_length = img_size * 0.08  # 动态最小长度
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = abs(line_angle(line[0]))
    length = line_length(line[0])
    
    # 只考虑足够长的线条
    if length > min_length:
        # 水平线 (角度接近0或180度)
        if angle < 15 or angle > 165:
            horizontal_lines.append(line[0])
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 垂直线 (角度接近90度)
        elif 75 < angle < 105:
            vertical_lines.append(line[0])
            cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

print(f"水平线: {len(horizontal_lines)} 条")
print(f"垂直线: {len(vertical_lines)} 条")

cv2.imshow('detected_lines', lines_img)

# 合并相近的平行线
def merge_lines(lines, is_horizontal=True):
    if not lines:
        return []
    
    merged = []
    tolerance = max(15, img_size * 0.025)  # 动态容差
    
    # 按位置排序（水平线按y坐标，垂直线按x坐标）
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
            # 计算当前组的平均线
            if current_group:
                avg_line = np.mean(current_group, axis=0).astype(int)
                merged.append(avg_line)
            current_group = [lines[i]]
    
    # 处理最后一组
    if current_group:
        avg_line = np.mean(current_group, axis=0).astype(int)
        merged.append(avg_line)
    
    return merged

# 合并相近的线条
merged_horizontal = merge_lines(horizontal_lines, True)
merged_vertical = merge_lines(vertical_lines, False)

print(f"合并后水平线: {len(merged_horizontal)} 条")
print(f"合并后垂直线: {len(merged_vertical)} 条")

# 绘制合并后的线条
merged_img = img_for_grid.copy()
for line in merged_horizontal:
    cv2.line(merged_img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
for line in merged_vertical:
    cv2.line(merged_img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)

cv2.imshow('merged_lines', merged_img)

# 计算交点
def line_intersection(line1, line2):
    """计算两条线的交点"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (int(x), int(y))

# 找到所有交点
intersections = []
for h_line in merged_horizontal:
    for v_line in merged_vertical:
        intersection = line_intersection(h_line, v_line)
        if intersection:
            intersections.append(intersection)

print(f"找到 {len(intersections)} 个交点")

# 绘制交点
grid_img = img_for_grid.copy()
for point in intersections:
    cv2.circle(grid_img, point, 3, (0, 0, 255), -1)

cv2.imshow('grid_points', grid_img)

# 如果交点数量合理，尝试构建19x19的标准围棋网格
min_intersections = 200 if img_size > 600 else 100  # 动态阈值
if len(intersections) > min_intersections:
    print("开始构建19x19围棋网格...")
    
    # 按行列排序交点
    intersections.sort(key=lambda p: (p[1], p[0]))  # 先按y排序，再按x排序
    
    # 找到网格的边界
    min_x = min(p[0] for p in intersections)
    max_x = max(p[0] for p in intersections)
    min_y = min(p[1] for p in intersections)
    max_y = max(p[1] for p in intersections)
    
    print(f"网格边界: x({min_x}, {max_x}), y({min_y}, {max_y})")
    
    # 按行分组交点
    rows = []
    current_row = []
    last_y = intersections[0][1]
    row_tolerance = max(20, img_size * 0.03)  # 动态行容差
    
    for point in intersections:
        if abs(point[1] - last_y) < row_tolerance:
            current_row.append(point)
        else:
            if current_row:
                # 按x坐标排序当前行
                current_row.sort(key=lambda p: p[0])
                rows.append(current_row)
            current_row = [point]
            last_y = point[1]
    
    # 添加最后一行
    if current_row:
        current_row.sort(key=lambda p: p[0])
        rows.append(current_row)
    
    print(f"检测到 {len(rows)} 行")
    
    # 过滤掉点数太少的行（可能是噪声）
    min_points_per_row = 12  # 降低要求
    valid_rows = [row for row in rows if len(row) >= min_points_per_row]
    print(f"有效行数: {len(valid_rows)}")
    
    # 如果有效行数接近19行，尝试提取标准19x19网格
    if len(valid_rows) >= 15:  # 进一步降低要求
        # 选择中间的19行（如果超过19行的话）
        if len(valid_rows) > 19:
            start_idx = (len(valid_rows) - 19) // 2
            selected_rows = valid_rows[start_idx:start_idx + 19]
        else:
            # 如果不足19行，尝试选择分布最均匀的行
            if len(valid_rows) >= 15:
                # 简单策略：均匀选择19行
                indices = np.linspace(0, len(valid_rows)-1, 19).astype(int)
                selected_rows = [valid_rows[i] for i in indices]
            else:
                selected_rows = valid_rows
        
        print(f"选择了 {len(selected_rows)} 行构建网格")
        
        # 构建标准19x19网格
        go_grid_points = []
        grid_img_with_go_points = img_for_grid.copy()
        
        for row_idx, row in enumerate(selected_rows):
            # 每行选择19个点
            if len(row) >= 19:
                # 如果这一行点数超过19，选择最均匀分布的19个点
                if len(row) > 19:
                    # 选择x坐标最均匀分布的19个点
                    indices = np.linspace(0, len(row)-1, 19).astype(int)
                    row_points = [row[i] for i in indices]
                else:
                    row_points = row[:19]
            else:
                # 如果点数不足19，尝试插值
                if len(row) >= 2:
                    # 简单线性插值生成19个点
                    start_x = row[0][0]
                    end_x = row[-1][0]
                    y = row[0][1]  # 使用第一个点的y坐标
                    
                    row_points = []
                    for i in range(19):
                        x = start_x + (end_x - start_x) * i / 18
                        row_points.append((int(x), int(y)))
                else:
                    continue  # 跳过点数太少的行
            
            # 添加到围棋网格点列表
            for col_idx, point in enumerate(row_points):
                go_grid_points.append((point[0], point[1], row_idx, col_idx))
                
                # 绘制围棋网格点
                cv2.circle(grid_img_with_go_points, point, 2, (0, 255, 255), -1)  # 黄色小点
                
                # 在关键点上标注坐标
                if row_idx % 3 == 0 and col_idx % 3 == 0:  # 每3个点标注一次
                    cv2.putText(grid_img_with_go_points, f"({row_idx},{col_idx})", 
                               (point[0]-15, point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.3, (0, 255, 255), 1)
        
        print(f"构建了 {len(go_grid_points)} 个围棋网格点 (目标: 361)")
        
        # 显示只有围棋网格点的图像
        cv2.imshow('go_grid_only', grid_img_with_go_points)
        
        # 使用围棋网格点重新定位棋子
        valid_go_grid_points = [(x, y) for x, y, _, _ in go_grid_points]
        
        # 自适应圆检测参数
        min_radius = max(5, int(img_size * 0.008))
        max_radius = max(15, int(img_size * 0.025))
        min_dist = int(img_size * 0.03)
        
        # 重新进行圆检测，但这次使用围棋网格信息来辅助
        circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=min_dist, param1=100, param2=19,
                                   minRadius=min_radius, maxRadius=max_radius)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            print(f'检测到 {len(circles[0])} 个圆形')
            
            # 对每个检测到的圆，找到最近的围棋网格点
            stones = []
            final_img = img_for_grid.copy()
            
            # 先绘制围棋网格点
            for point in valid_go_grid_points:
                cv2.circle(final_img, point, 2, (0, 255, 255), -1)
            
            snap_distance = max(20, int(img_size * 0.03))  # 动态吸附距离
            
            for circle in circles[0, :]:
                cx, cy, r = circle
                
                # 找到最近的围棋网格点
                min_dist = float('inf')
                closest_grid_point = None
                closest_grid_pos = None
                
                for x, y, row, col in go_grid_points:
                    dist = math.sqrt((cx - x)**2 + (cy - y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_grid_point = (x, y)
                        closest_grid_pos = (row, col)
                
                # 如果圆心离最近围棋网格点足够近，认为这是一个有效的棋子
                if min_dist < snap_distance:
                    # 绘制圆形
                    cv2.circle(final_img, (cx, cy), r, (255, 255, 0), 2)
                    cv2.circle(final_img, (cx, cy), 2, (255, 255, 0), -1)
                    
                    # 颜色检测
                    roi = gray[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
                    if roi.size > 0:
                        avg_intensity = np.mean(roi)
                        color = 'black' if avg_intensity < 120 else 'white'
                        stones.append((color, closest_grid_pos[0], closest_grid_pos[1]))
                        
                        # 标注棋子信息
                        text_color = (255, 255, 255) if color == 'black' else (0, 0, 0)
                        cv2.putText(final_img, f"{color[0].upper()}({closest_grid_pos[0]+1},{closest_grid_pos[1]+1})", 
                                   (cx-25, cy-r-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                        
                        # 在对应的网格点上画一个小圆圈表示棋子位置
                        cv2.circle(final_img, closest_grid_point, 8, (0, 255, 0), 2)
            
            print(f"在围棋网格上识别到 {len(stones)} 个棋子")
            
            # 创建围棋棋盘状态矩阵
            board_state = np.zeros((19, 19), dtype=int)
            for color, row, col in stones:
                if 0 <= row < 19 and 0 <= col < 19:
                    board_state[row][col] = 1 if color == 'black' else 2
                    print(f"{color} 棋子在位置: ({row+1}, {col+1})")
            
            print("\n围棋棋盘状态 (0=空, 1=黑子, 2=白子):")
            print("行号:", end="  ")
            for i in range(19):
                print(f"{i+1:2d}", end="")
            print()
            
            for i in range(19):
                print(f"{i+1:2d}: ", end="")
                for j in range(19):