import cv2
import numpy as np

def line_intersection(line1, line2):
    """计算两条直线的交点"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None  # 平行线
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    
    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)
    return (x, y)

def fix_board_corners_improved(detected_points):
    """
    改进的棋盘角点修复，处理倾斜棋盘的情况
    """
    points = np.array(detected_points)
    
    print("原始检测点：")
    for i, pt in enumerate(points):
        print(f"  点{i+1}: ({pt[0]}, {pt[1]})")
    
    # 更智能的角点识别方法
    # 先按x坐标排序，找到最左和最右的点
    sorted_by_x = sorted(points, key=lambda p: p[0])
    leftmost = sorted_by_x[:2]  # 最左边的两个点
    rightmost = sorted_by_x[2:]  # 最右边的两个点
    
    # 在左边的两个点中，y较小的是左上，y较大的是左下
    if leftmost[0][1] < leftmost[1][1]:
        left_top_partial = leftmost[0]
        left_bottom = leftmost[1]
    else:
        left_top_partial = leftmost[1] 
        left_bottom = leftmost[0]
    
    # 在右边的两个点中，y较小的是右上，y较大的是右下
    if rightmost[0][1] < rightmost[1][1]:
        right_top = rightmost[0]
        right_bottom = rightmost[1]
    else:
        right_top = rightmost[1]
        right_bottom = rightmost[0]
    
    print("\n识别的角点：")
    print(f"左上角(可能被切): {left_top_partial}")
    print(f"左下角: {left_bottom}")
    print(f"右上角: {right_top}")
    print(f"右下角: {right_bottom}")
    
    # 判断左上角是否被切掉（x坐标接近0）
    if left_top_partial[0] < 20:  # 如果x坐标很小，说明被左边缘切掉了
        print("\n左上角被切掉，需要计算真实位置")
        
        # 方法：利用棋盘的平行四边形特性
        # 左边线：从left_bottom延伸到left_top_partial的方向
        left_direction = np.array(left_top_partial) - np.array(left_bottom)
        
        # 上边线：从right_top延伸，方向应该与下边线平行
        # 下边线方向：从left_bottom到right_bottom
        bottom_direction = np.array(right_bottom) - np.array(left_bottom)
        
        # 上边线：从right_top开始，沿着与bottom_direction相反的方向
        top_line_start = np.array(right_top) - bottom_direction * 2  # 延长2倍确保相交
        top_line_end = np.array(right_top) + bottom_direction * 0.5
        
        # 左边线：从left_bottom延伸
        left_line_start = np.array(left_bottom) 
        left_line_end = np.array(left_bottom) + left_direction * 2  # 延长
        
        # 计算两线交点
        left_line = (left_line_start[0], left_line_start[1], left_line_end[0], left_line_end[1])
        top_line = (top_line_start[0], top_line_start[1], top_line_end[0], top_line_end[1])
        
        true_left_top = line_intersection(left_line, top_line)
        
        if true_left_top is None:
            print("交点计算失败，使用几何估算")
            # 备用方案：基于平行四边形的向量关系
            # left_top = right_top + (left_bottom - right_bottom)
            true_left_top = np.array(right_top) + (np.array(left_bottom) - np.array(right_bottom))
            true_left_top = tuple(true_left_top)
        
        print(f"计算的真实左上角: ({true_left_top[0]:.1f}, {true_left_top[1]:.1f})")
        
    else:
        true_left_top = tuple(left_top_partial)
        print("左上角未被切掉")
    
    # 验证结果的合理性
    print(f"\n棋盘倾斜分析：")
    print(f"左边高度差: {abs(left_bottom[1] - true_left_top[1]):.1f}")
    print(f"右边高度差: {abs(right_bottom[1] - right_top[1]):.1f}")
    print(f"上边倾斜: {abs(right_top[1] - true_left_top[1]):.1f}")
    print(f"下边倾斜: {abs(right_bottom[1] - left_bottom[1]):.1f}")
    
    # 返回修正后的四个角点：左上、右上、右下、左下
    corrected_corners = np.array([
        true_left_top,
        right_top,
        right_bottom,
        left_bottom
    ], dtype="float32")
    
    return corrected_corners

# 读取图像
image_path = "../data/raw/Homeboard4.jpg"
img = cv2.imread(image_path)

if img is None:
    print("没找到照片", image_path)
    exit()

print(f"读取图像: {image_path}")
print(f"图像尺寸: {img.shape[1]} x {img.shape[0]}")

cv2.imshow("orginal picture", img)

im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (3,3), 0)
im_edge = cv2.Canny(im_gray, 30, 50)
cv2.imshow('Go', im_edge)

contours, hierarchy = cv2.findContours(im_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
detected_points, area = None, 0
for item in contours:
    hull = cv2.convexHull(item)
    epsilon = 0.1 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        a = cv2.contourArea(approx)
        if a > area:
            area = a
            detected_points = np.reshape(approx, (4, 2))

if detected_points is None:
    print('在图像文件中找不到棋盘！')
    exit()

print('='*50)
print('原始检测到的四个点：')
for i, pt in enumerate(detected_points):
    print(f'  点{i+1}: ({pt[0]},{pt[1]})')

# 使用改进的角点修复方法
rect = fix_board_corners_improved(detected_points)

print('='*50)
print('修复后的棋盘坐标：')
print(f'  左上角：({rect[0][0]:.1f},{rect[0][1]:.1f})')
print(f'  右上角：({rect[1][0]:.1f},{rect[1][1]:.1f})')
print(f'  右下角：({rect[2][0]:.1f},{rect[2][1]:.1f})')
print(f'  左下角：({rect[3][0]:.1f},{rect[3][1]:.1f})')

# 在原图上标记角点
im = np.copy(img)
colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]  # 红绿蓝黄
labels = ['LT', 'RT', 'RB', 'LB']

for i, (p, color, label) in enumerate(zip(rect, colors, labels)):
    x, y = int(p[0]), int(p[1])
    cv2.circle(im, (x, y), 15, color, -1)
    cv2.putText(im, label, (x+20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

# 如果左上角在图像外，画箭头指示
if rect[0][0] < 0:
    arrow_y = int(rect[0][1])
    cv2.arrowedLine(im, (50, arrow_y), (10, arrow_y), (0,0,255), 5)
    cv2.putText(im, f'Real LT({rect[0][0]:.0f},{rect[0][1]:.0f})', 
                (60, arrow_y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

cv2.imshow('detected corners', im)

# 设置透视变换
output_size = 1000
margin = 50

# 目标坐标：左上、右上、右下、左下
pts1 = np.float32([
    [margin, margin],                    # 左上角
    [output_size - margin, margin],      # 右上角  
    [output_size - margin, output_size - margin],  # 右下角
    [margin, output_size - margin]       # 左下角
])

pts2 = rect

print("="*50)
print("透视变换映射:")
labels = ['左上', '右上', '右下', '左下']
for i in range(4):
    print(f"  {labels[i]}: ({pts2[i][0]:.1f},{pts2[i][1]:.1f}) -> ({pts1[i][0]:.0f},{pts1[i][1]:.0f})")

# 生成透视变换矩阵
m = cv2.getPerspectiveTransform(pts2, pts1)

# 执行透视变换
board_gray = cv2.warpPerspective(im_gray, m, (output_size, output_size))
board_bgr = cv2.warpPerspective(img, m, (output_size, output_size))

cv2.imshow('transformed board', board_gray)
cv2.imshow('transformed board color', board_bgr)

# 保存结果
cv2.imwrite('transformed_board.jpg', board_bgr)
print("变换后的棋盘已保存为 transformed_board.jpg")

print("\n按任意键继续...")
cv2.waitKey(0)
cv2.destroyAllWindows()