import cv2
import numpy as np
from collections import defaultdict
import math

# 读取图像
#img = cv2.imread('../data/raw/bd317d54.webp')
img = cv2.imread('../data/raw/IMG20171015161921.jpg')

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
    tolerance = 15  # 像素容差
    
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
if len(intersections) > 100:  # 降低阈值
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
    row_tolerance = 20  # 同一行的y坐标容差
    
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
    valid_rows = [row for row in rows if len(row) >= 15]  # 至少15个点才算有效行
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
                                   dp=1, minDist=20, param1=100, param2=19,
                                   minRadius=8, maxRadius=25)
        
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
                if min_dist < 20:  # 20像素的容差
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