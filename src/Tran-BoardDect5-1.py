#整合版本：先进行透视变换，然后进行棋盘线检测
#添加自动缩放功能以适应高分辨率图片
#这一版Claude视图修改透视变换的pts1, pts2顺序，但是一如既往的失败了，透视变换根本没有生效。 抛弃
import cv2
import numpy as np
from collections import defaultdict
import math

# 读取图像
#img = cv2.imread('../data/raw/bd317d54.webp')
#img = cv2.imread('../data/raw/IMG20171015161921.jpg')
#img = cv2.imread('../data/raw/OGS3.jpeg')
#img = cv2.imread('../data/raw/IMG20160706171004.jpg')
#img = cv2.imread('../data/raw/IMG20160904165505-B.jpg')
img = cv2.imread('../data/raw/IMG20160706171004-16.jpg')

if img is None:
    print("没找到照片")
    exit()

# ====== 新增：自动缩放图片功能 ======
def auto_resize_image(image, target_width):
    """
    自动缩放图片到合适的尺寸进行处理
    target_width: 目标宽度，默认1000像素
    """
    height, width = image.shape[:2]
    print(f"原图尺寸: {width} x {height}")
    
    # 如果图片宽度超过目标宽度，则缩放
    #if width > target_width:
    if width > 2500:
        #scale_factor = target_width / width
        scale_factor = int((width * 10 / target_width))
        print("scale_factor is:", scale_factor)
        new_width = width * 10 // scale_factor
        #new_width = target_width
        #new_height = int(height * scale_factor)
        new_height = height * 10 // scale_factor
        
        #resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        #resized_img = cv2.resize(image, (width*10//scale_factor, height*10//scale_factor), interpolation=cv2.INTER_AREA)
        resized_img = cv2.resize(image, (new_width, new_width), interpolation=cv2.INTER_AREA)
        print(f"缩放后尺寸: {new_width} x {new_height} (缩放比例: {scale_factor:.3f})")
        #print(f"缩放后尺寸: {width*10//scale_factor} x {height*10//scale_factor} (缩放比例: {scale_factor:.3f})")
        return resized_img, scale_factor
    else:
        print("图片尺寸合适，无需缩放")
        return image, 1.0

# 缩放图片
img, scale_factor = auto_resize_image(img, target_width=1500)

# ====== 计算动态参数 ======
# 基准尺寸：1350像素宽度
base_width = 1350
current_width = img.shape[1]
param_scale = current_width / base_width if current_width < base_width else 1.0

print(f"当前图片宽度: {current_width}, 参数缩放比例: {param_scale:.3f}")

# ====== 计算动态参数 ======
# 基准尺寸：1350像素宽度
base_width = 1350
current_width = img.shape[1]
param_scale = current_width / base_width if current_width < base_width else 1.0

print(f"当前图片宽度: {current_width}, 参数缩放比例: {param_scale:.3f}")

# 根据图片大小选择参数集
if current_width <= 1000:
    # 小图参数集 (适用于宽度 <= 1000px)
    print("使用小图参数集")
    perspective_size = 400
    perspective_corner = 390
    hough_threshold = 40
    min_line_length = 50
    max_line_gap = 8
    line_filter_length = 25
    merge_tolerance = 8
    row_tolerance = 12
    min_dist_threshold = 12
    min_radius = 4
    max_radius = 15
    min_dist_circles = 12
    min_points_per_row = 8
    intersection_threshold = 60
    
elif current_width <= 1500:
    # 中图参数集 (适用于宽度 1000-1500px)
    print("使用中图参数集")
    perspective_size = 500
    perspective_corner = 490
    hough_threshold = 60
    min_line_length = 70
    max_line_gap = 10
    line_filter_length = 35
    merge_tolerance = 12
    row_tolerance = 15
    min_dist_threshold = 15
    min_radius = 6
    max_radius = 20
    min_dist_circles = 15
    min_points_per_row = 12
    intersection_threshold = 80
    
else:
    # 大图参数集 (适用于宽度 > 1500px)
    print("使用大图参数集")
    perspective_size = 660
    perspective_corner = 650
    hough_threshold = 80
    min_line_length = 100
    max_line_gap = 10
    line_filter_length = 50
    merge_tolerance = 15
    row_tolerance = 20
    min_dist_threshold = 20
    min_radius = 8
    max_radius = 25
    min_dist_circles = 20
    min_points_per_row = 15
    intersection_threshold = 100

print(f"参数设置:")
print(f"perspective_size={perspective_size}, perspective_corner={perspective_corner}")
print(f"hough_threshold={hough_threshold}, min_line_length={min_line_length}")
print(f"max_line_gap={max_line_gap}, line_filter_length={line_filter_length}")
print(f"merge_tolerance={merge_tolerance}, row_tolerance={row_tolerance}")
print(f"min_dist_threshold={min_dist_threshold}")
print(f"min_radius={min_radius}, max_radius={max_radius}, min_dist_circles={min_dist_circles}")
print(f"min_points_per_row={min_points_per_row}, intersection_threshold={intersection_threshold}")

# ====== 参数设置结束 ======

# ====== 缩放功能结束 ======

original_img = img.copy()
cv2.imshow("original picture", img)

print("开始透视变换...")

# ====== 添加缺失的棋盘检测代码 ======
# 转换为灰度图
im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 简单的棋盘角点检测 - 这里你需要根据实际需求实现
# 如果没有棋盘检测，直接跳过透视变换
rect = None  # 初始化为None，表示没有检测到棋盘轮廓

# 如果你有棋盘检测代码，可以在这里添加
# 例如：
# rect = detect_board_corners(img)  # 这个函数需要你实现

# ====== 透视变换部分 - 修正版 ======
if rect is None:
    print('在图像文件中找不到棋盘！使用原图进行后续处理...')
    # 如果没找到棋盘轮廓，使用原图
    img = original_img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    print('棋盘坐标：')
    print('\t左上角:(%d,%d)'%(rect[0][0],rect[0][1]))
    print('\t左下角:(%d,%d)'%(rect[1][0],rect[1][1]))
    print('\t右上角:(%d,%d)'%(rect[2][0],rect[2][1]))
    print('\t右下角:(%d,%d)'%(rect[3][0],rect[3][1]))

    # 显示找到的角点
    corner_img = img.copy()
    for p in rect:
        cv2.circle(corner_img,(p[0],p[1]),15,(0,255,0),-1)
    cv2.imshow('Found Corners', corner_img)

    # 执行透视变换 - 修正版
    lt, lb, rt, rb = rect  # 左上, 左下, 右上, 右下
    
    # 设置边界留白，避免边缘被切掉
    margin = 20
    target_size = perspective_size - 2 * margin
    
    # 目标坐标：保持与源坐标相同的顺序 (左上, 左下, 右上, 右下)
    pts1 = np.float32([
        [margin, margin],                    # 左上角
        [margin, margin + target_size],      # 左下角  
        [margin + target_size, margin],      # 右上角
        [margin + target_size, margin + target_size]  # 右下角
    ])
    
    # 源坐标：当前找到的棋盘四个角的坐标
    pts2 = np.float32([lt, lb, rt, rb])
    
    print(f"透视变换参数:")
    print(f"目标画布大小: {perspective_size}x{perspective_size}")
    print(f"有效区域大小: {target_size}x{target_size} (边距: {margin})")
    print(f"源坐标: 左上{lt}, 左下{lb}, 右上{rt}, 右下{rb}")
    print(f"目标坐标: {pts1}")
    
    # 生成透视矩阵
    m = cv2.getPerspectiveTransform(pts2, pts1)
    
    # 对原图执行透视变换
    img = cv2.warpPerspective(original_img, m, (perspective_size, perspective_size))
    gray = cv2.warpPerspective(im_gray, m, (perspective_size, perspective_size))
    
    cv2.imshow('Perspective Corrected', img)
    print("透视变换完成！")

# ====== 棋盘线检测部分 ======
print("开始棋盘线检测...")

# 高斯模糊
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(blur, 50, 150)
cv2.imshow('edges', edges)

# 使用霍夫变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

if lines is None:
    print("未检测到任何直线")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

print(f"检测到 {len(lines)} 条直线")

# 创建一个副本用于绘制检测到的线条
lines_img = img.copy()

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
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = abs(line_angle(line[0]))
    length = line_length(line[0])
    
    # 只考虑足够长的线条
    if length > line_filter_length:
        # 水平线 (角度接近0或180度)
        if angle < 10 or angle > 170:
            horizontal_lines.append(line[0])
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 垂直线 (角度接近90度)
        elif 80 < angle < 100:
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
    tolerance = merge_tolerance  # 使用动态参数
    
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
merged_img = img.copy()
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
grid_img = img.copy()
for point in intersections:
    cv2.circle(grid_img, point, 3, (0, 0, 255), -1)

cv2.imshow('grid_points', grid_img)

# 如果交点数量合理，尝试构建19x19的标准围棋网格
intersection_threshold = max(50, int(100 * param_scale))  # 动态调整交点阈值
if len(intersections) > intersection_threshold:
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
    min_points_per_row = max(10, int(15 * param_scale))  # 动态调整每行最少点数
    valid_rows = [row for row in rows if len(row) >= min_points_per_row]
    print(f"有效行数: {len(valid_rows)}")
    
    # 如果有效行数接近19行，尝试提取标准19x19网格
    if len(valid_rows) >= 18:  # 允许少量误差
        # 选择中间的19行（如果超过19行的话）
        if len(valid_rows) > 19:
            start_idx = (len(valid_rows) - 19) // 2
            selected_rows = valid_rows[start_idx:start_idx + 19]
        else:
            selected_rows = valid_rows[:19]  # 取前19行
        
        print(f"选择了 {len(selected_rows)} 行构建网格")
        
        # 构建标准19x19网格
        go_grid_points = []
        grid_img_with_go_points = img.copy()
        
        for row_idx, row in enumerate(selected_rows):
            # 每行选择19个点
            if len(row) >= 19:
                # 如果这一行点数超过19，选择最均匀分布的19个点
                if len(row) > 19:
                    # 选择x坐标最均匀分布的19个点
                    step = len(row) / 19
                    selected_points = []
                    for i in range(19):
                        idx = min(int(i * step), len(row) - 1)
                        selected_points.append(row[idx])
                    row_points = selected_points
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
    
        # 重新进行圆检测，但这次使用围棋网格信息来辅助
        circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=min_dist_circles, param1=100, param2=19,
                                   minRadius=min_radius, maxRadius=max_radius)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            print(f'检测到 {len(circles[0])} 个圆形')
            
            # 对每个检测到的圆，找到最近的围棋网格点
            stones = []
            final_img = img.copy()
            
            # 先绘制围棋网格点
            for point in valid_go_grid_points:
                cv2.circle(final_img, point, 2, (0, 255, 255), -1)
            
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
                if min_dist < min_dist_threshold:  # 使用动态参数
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
                    if board_state[i][j] == 0:
                        print(" .", end="")
                    elif board_state[i][j] == 1:
                        print(" ●", end="")
                    else:
                        print(" ○", end="")
                print()
            
            # 显示最终结果
            cv2.imshow('final_go_board', final_img)
            
    else:
        print("未能检测到足够的网格行来构建19x19围棋棋盘")
        
else:
    print("检测到的交点数量不足，无法构建围棋网格")

cv2.waitKey(0)
cv2.destroyAllWindows()