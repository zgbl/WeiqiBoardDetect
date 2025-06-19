# 基于原版本，增加倾斜四边形网格绘制功能
import cv2
import numpy as np
from collections import defaultdict
import math

# 读取图像
#img = cv2.imread('../data/raw/bd317d54.webp')
img = cv2.imread('../data/raw/IMG20171015161921.jpg')
#img = cv2.imread('../data/raw/OGS3.jpeg')

original_img = img.copy()

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(blur, 50, 150)
cv2.imshow('edges', edges)

# 使用霍夫变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=10)

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
    if length > 50:
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
    #tolerance = 15  # 像素容差
    tolerance = 10
    
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
    cv2.circle(grid_img, point, 5, (0, 0, 255), -1)

cv2.imshow('grid_points', grid_img)

# 新增功能：构建倾斜的四边形围棋网格
def build_tilted_go_grid(intersections, img):
    """基于检测到的交点构建倾斜的19x19围棋网格"""
    if len(intersections) < 100:
        print("交点数量不足，无法构建网格")
        return None, None
    
    print("开始构建倾斜19x19围棋网格...")
    
    # 找到所有交点的边界
    min_x = min(p[0] for p in intersections)
    max_x = max(p[0] for p in intersections)
    min_y = min(p[1] for p in intersections)
    max_y = max(p[1] for p in intersections)
    
    print(f"交点边界: x({min_x}-{max_x}), y({min_y}-{max_y})")
    
    # 按行分组交点 - 使用更大的容差确保包含所有点
    intersections.sort(key=lambda p: (p[1], p[0]))
    
    rows = []
    current_row = []
    last_y = intersections[0][1]
    row_tolerance = max(20, (max_y - min_y) / 30)  # 动态调整容差
    
    for point in intersections:
        if abs(point[1] - last_y) < row_tolerance:
            current_row.append(point)
        else:
            if current_row:
                current_row.sort(key=lambda p: p[0])
                rows.append(current_row)
            current_row = [point]
            last_y = point[1]
    
    if current_row:
        current_row.sort(key=lambda p: p[0])
        rows.append(current_row)
    
    print(f"检测到 {len(rows)} 行")
    for i, row in enumerate(rows):
        print(f"第{i}行: {len(row)}个点, y范围: {min(p[1] for p in row)}-{max(p[1] for p in row)}")
    
    # 更宽松的行筛选条件
    valid_rows = [row for row in rows if len(row) >= 10]  # 降低阈值
    print(f"有效行数: {len(valid_rows)}")
    
    if len(valid_rows) < 15:
        print("有效行数不足，无法构建19x19网格")
        return None, None
    
    # 如果行数不足19，尝试扩展
    if len(valid_rows) < 19:
        print(f"当前只有{len(valid_rows)}行，尝试构建{len(valid_rows)}x19网格")
        selected_rows = valid_rows
    else:
        # 选择19行，优先选择中间部分
        if len(valid_rows) > 19:
            start_idx = (len(valid_rows) - 19) // 2
            selected_rows = valid_rows[start_idx:start_idx + 19]
        else:
            selected_rows = valid_rows
    
    actual_rows = len(selected_rows)
    print(f"最终选择 {actual_rows} 行构建网格")
    
    # 构建网格点矩阵
    grid_matrix = np.zeros((19, 19, 2), dtype=int)
    
    for row_idx, row in enumerate(selected_rows):
        if row_idx >= 19:  # 防止越界
            break
        
        # 找到这一行的x坐标范围
        row_min_x = min(p[0] for p in row)
        row_max_x = max(p[0] for p in row)
        row_y = int(np.mean([p[1] for p in row]))  # 使用平均y坐标
        
        print(f"处理第{row_idx}行: {len(row)}个点, x范围({row_min_x}-{row_max_x}), y={row_y}")
        
        # 为这一行生成19个均匀分布的点
        if len(row) >= 19:
            # 如果点数足够，选择最均匀分布的19个点
            if len(row) > 19:
                # 使用更智能的选点策略
                indices = np.linspace(0, len(row) - 1, 19, dtype=int)
                row_points = [row[i] for i in indices]
            else:
                row_points = row[:19]
        else:
            # 点数不足时，使用线性插值生成19个点
            row_points = []
            for i in range(19):
                if len(row) >= 2:
                    # 基于实际点进行插值
                    t = i / 18.0
                    x = int(row_min_x + t * (row_max_x - row_min_x))
                    row_points.append((x, row_y))
                else:
                    # 如果点太少，使用边界信息
                    x = int(min_x + (max_x - min_x) * i / 18)
                    row_points.append((x, row_y))
        
        # 填充网格矩阵
        for col_idx, point in enumerate(row_points):
            if col_idx < 19:
                grid_matrix[row_idx][col_idx] = [point[0], point[1]]
                
    # 如果实际行数少于19，通过插值填充剩余行
    if actual_rows < 19:
        print(f"插值填充剩余 {19 - actual_rows} 行")
        for row_idx in range(actual_rows, 19):
            # 基于已有行进行插值
            if actual_rows >= 2:
                # 使用第一行和最后一行进行插值
                first_row = 0
                last_row = actual_rows - 1
                t = (row_idx - first_row) / (18.0)  # 在整个19行范围内的位置
                
                for col_idx in range(19):
                    if np.any(grid_matrix[first_row][col_idx]) and np.any(grid_matrix[last_row][col_idx]):
                        x1, y1 = grid_matrix[first_row][col_idx]
                        x2, y2 = grid_matrix[last_row][col_idx]
                        
                        # 线性插值
                        x = int(x1 + t * (x2 - x1))
                        y = int(y1 + t * (y2 - y1))
                        grid_matrix[row_idx][col_idx] = [x, y]
    
    return grid_matrix, selected_rows

def draw_tilted_grid_lines(img, grid_matrix):
    """在原图上绘制倾斜的网格线"""
    grid_img = img.copy()
    
    # 统计有效点数量
    valid_points = 0
    for row in range(19):
        for col in range(19):
            if np.any(grid_matrix[row][col]):
                valid_points += 1
    
    print(f"网格矩阵中有效点数量: {valid_points}")
    
    # 绘制水平线（连接每行的点）
    for row in range(19):
        for col in range(18):
            pt1 = tuple(grid_matrix[row][col])
            pt2 = tuple(grid_matrix[row][col + 1])
            # 检查点是否有效（非零点）
            if np.any(grid_matrix[row][col]) and np.any(grid_matrix[row][col + 1]):
                cv2.line(grid_img, pt1, pt2, (0, 255, 0), 1)
    
    # 绘制垂直线（连接每列的点）
    for col in range(19):
        for row in range(18):
            pt1 = tuple(grid_matrix[row][col])
            pt2 = tuple(grid_matrix[row + 1][col])
            # 检查点是否有效（非零点）
            if np.any(grid_matrix[row][col]) and np.any(grid_matrix[row + 1][col]):
                cv2.line(grid_img, pt1, pt2, (255, 0, 0), 1)
    
    # 绘制网格点
    for row in range(19):
        for col in range(19):
            if np.any(grid_matrix[row][col]):  # 检查是否为有效点
                pt = tuple(grid_matrix[row][col])
                cv2.circle(grid_img, pt, 2, (0, 0, 255), -1)
                
                # 在关键点标注坐标
                if row % 6 == 0 and col % 6 == 0:
                    cv2.putText(grid_img, f"({row},{col})", 
                               (pt[0]-15, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.3, (255, 255, 0), 1)
    
    return grid_img

# 构建倾斜网格
grid_matrix, selected_rows = build_tilted_go_grid(intersections, img)

if grid_matrix is not None:
    # 绘制倾斜的四边形网格
    tilted_grid_img = draw_tilted_grid_lines(img, grid_matrix)
    cv2.imshow('tilted_go_grid', tilted_grid_img)
    
    print("成功构建倾斜的19x19围棋网格")
    
    # 基于倾斜网格进行棋子识别
    circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT,
                               dp=1, minDist=20, param1=100, param2=19,
                               minRadius=8, maxRadius=25)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f'检测到 {len(circles[0])} 个圆形')
        
        stones = []
        final_img = tilted_grid_img.copy()
        
        for circle in circles[0, :]:
            cx, cy, r = circle
            
            # 找到最近的网格点
            min_dist = float('inf')
            closest_grid_pos = None
            closest_grid_point = None
            
            for row in range(19):
                for col in range(19):
                    gx, gy = grid_matrix[row][col]
                    if gx != 0 or gy != 0:  # 确保是有效点
                        dist = math.sqrt((cx - gx)**2 + (cy - gy)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_grid_pos = (row, col)
                            closest_grid_point = (gx, gy)
            
            # 如果圆心离最近网格点足够近
            if min_dist < 20:
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
                    
                    # 在对应的网格点上画圆圈
                    cv2.circle(final_img, closest_grid_point, 8, (0, 255, 0), 2)
        
        print(f"在倾斜围棋网格上识别到 {len(stones)} 个棋子")
        
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
        
        cv2.imshow('final_tilted_go_board', final_img)
    
else:
    print("未能构建倾斜围棋网格")

cv2.waitKey(0)
cv2.destroyAllWindows()