#这个版本基本把棋盘交叉点都准确定位出来了。是目前最佳版本。6/18/2025 13:47
import cv2
import numpy as np
from collections import defaultdict
import math

# 读取图像
#img = cv2.imread('../data/raw/cndb1.jpg')
#img = cv2.imread('../data/raw/OGS4.jpg')
img = cv2.imread('../data/raw/bd317d54.webp')
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

# 如果交点数量合理（接近19x19=361），尝试构建网格
if len(intersections) > 300:  # 大概的阈值
    # 按行列排序交点
    intersections.sort(key=lambda p: (p[1], p[0]))  # 先按y排序，再按x排序
    
    # 尝试找到规律的网格结构
    # 这里可以进一步优化网格点的识别
    
    # 重新进行圆检测，但这次使用网格信息来辅助
    circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT,
                               dp=1, minDist=20, param1=100, param2=19,
                               minRadius=8, maxRadius=25)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f'检测到 {len(circles[0])} 个圆形')
        
        # 对每个检测到的圆，找到最近的网格交点
        stones = []
        for circle in circles[0, :]:
            cx, cy, r = circle
            
            # 找到最近的交点
            min_dist = float('inf')
            closest_intersection = None
            for intersection in intersections:
                dist = math.sqrt((cx - intersection[0])**2 + (cy - intersection[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_intersection = intersection
            
            # 如果圆心离最近交点足够近，认为这是一个有效的棋子
            if min_dist < 15:  # 15像素的容差
                # 绘制圆形
                cv2.circle(grid_img, (cx, cy), r, (255, 255, 0), 2)
                cv2.circle(grid_img, (cx, cy), 2, (255, 255, 0), -1)
                
                # 颜色检测 (简化版本)
                roi = gray[cy-r:cy+r, cx-r:cx+r]
                if roi.size > 0:
                    avg_intensity = np.mean(roi)
                    color = 'black' if avg_intensity < 120 else 'white'
                    stones.append((color, closest_intersection[0], closest_intersection[1]))
                    
                    # 标注颜色
                    cv2.putText(grid_img, color[0].upper(), (cx-10, cy-r-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        print(f"识别到 {len(stones)} 个棋子")
        for stone in stones:
            color, x, y = stone
            print(f"{color} 棋子位置: ({x}, {y})")

# 显示最终结果
cv2.imshow('final_result', grid_img)
cv2.waitKey(0)
cv2.destroyAllWindows()