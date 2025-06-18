import cv2
import numpy as np
import math

def is_perspective_correction_needed(img, corners):
    """判断是否需要透视校正"""
    if corners is None:
        return False
    
    lt, lb, rt, rb = corners
    
    # 计算四边长度
    left_edge = math.sqrt((lt[0] - lb[0])**2 + (lt[1] - lb[1])**2)
    right_edge = math.sqrt((rt[0] - rb[0])**2 + (rt[1] - rb[1])**2)
    top_edge = math.sqrt((lt[0] - rt[0])**2 + (lt[1] - rt[1])**2)
    bottom_edge = math.sqrt((lb[0] - rb[0])**2 + (lb[1] - rb[1])**2)
    
    # 计算边长比例差异
    horizontal_ratio = abs(top_edge - bottom_edge) / max(top_edge, bottom_edge)
    vertical_ratio = abs(left_edge - right_edge) / max(left_edge, right_edge)
    
    # 计算角度偏差
    def calculate_angle(p1, p2, p3):
        """计算三点构成的角度"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = math.acos(np.clip(cos_angle, -1, 1)) * 180 / math.pi
        return abs(angle - 90)  # 偏离90度的程度
    
    # 计算四个角的角度偏差
    angle_deviations = [
        calculate_angle(rb, lt, lb),  # 左上角
        calculate_angle(lt, lb, rb),  # 左下角
        calculate_angle(lb, rt, lt),  # 右上角
        calculate_angle(rt, rb, lb)   # 右下角
    ]
    
    max_angle_deviation = max(angle_deviations)
    
    # 判断是否需要透视校正
    # 如果边长比例差异大于20%或角度偏差大于15度，则需要校正
    needs_correction = (horizontal_ratio > 0.2 or vertical_ratio > 0.2 or max_angle_deviation > 15)
    
    print(f"透视校正判断:")
    print(f"  水平边长比例差异: {horizontal_ratio:.3f}")
    print(f"  垂直边长比例差异: {vertical_ratio:.3f}")
    print(f"  最大角度偏差: {max_angle_deviation:.1f}度")
    print(f"  是否需要透视校正: {'是' if needs_correction else '否'}")
    
    return needs_correction

def find_board_corners(img):
    """找到棋盘的四个角点"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(blur, 50, 150)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 寻找最大的矩形轮廓（应该是棋盘）
    max_area = 0
    best_contour = None
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area > max_area and area > 10000:  # 过滤太小的轮廓
            # 尝试逼近为矩形
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果逼近结果是四边形，认为可能是棋盘
            if len(approx) == 4:
                max_area = area
                best_contour = approx
    
    if best_contour is not None:
        # 将角点按照左上、左下、右上、右下的顺序排列
        points = best_contour.reshape(4, 2)
        
        # 计算重心
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        
        # 按象限分类角点
        lt = None  # 左上
        lb = None  # 左下
        rt = None  # 右上
        rb = None  # 右下
        
        for point in points:
            x, y = point
            if x < center_x and y < center_y:  # 左上
                lt = point
            elif x < center_x and y > center_y:  # 左下
                lb = point
            elif x > center_x and y < center_y:  # 右上
                rt = point
            elif x > center_x and y > center_y:  # 右下
                rb = point
        
        if all(p is not None for p in [lt, lb, rt, rb]):
            return lt, lb, rt, rb
    
    return None

def apply_perspective_transform(img, corners, board_size=660):
    """应用透视变换"""
    lt, lb, rt, rb = corners
    
    # 源图像的四个角点
    pts2 = np.float32([lt, lb, rt, rb])
    
    # 目标图像的四个角点（标准正方形）
    margin = 30  # 边距
    pts1 = np.float32([
        (margin, margin),              # 左上
        (margin, board_size - margin), # 左下
        (board_size - margin, margin), # 右上
        (board_size - margin, board_size - margin)  # 右下
    ])
    
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts2, pts1)
    
    # 应用透视变换
    warped = cv2.warpPerspective(img, M, (board_size, board_size))
    
    return warped, M

def detect_board_region(img):
    """检测棋盘区域，返回棋盘边界"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用霍夫直线检测找到网格线
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=10)
    cv2.imshow("edges", edges)

    if lines is None:
        print("未检测到足够的直线，使用整个图像")
        return 0, 0, img.shape[1], img.shape[0]
    
    # 分离水平线和垂直线
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        
        if angle < 15 or angle > 165:  # 水平线
            h_lines.append(line[0])
        elif 75 < angle < 105:  # 垂直线
            v_lines.append(line[0])
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        print("未检测到足够的网格线，使用整个图像")
        return 0, 0, img.shape[1], img.shape[0]
    
    # 找到边界线
    h_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    v_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
    
    top_y = (h_lines[0][1] + h_lines[0][3]) / 2
    bottom_y = (h_lines[-1][1] + h_lines[-1][3]) / 2
    left_x = (v_lines[0][0] + v_lines[0][2]) / 2
    right_x = (v_lines[-1][0] + v_lines[-1][2]) / 2
    
    return int(left_x), int(top_y), int(right_x), int(bottom_y)

def create_valid_intersection_points(img, corners=None):
    """创建有效的围棋交叉点（19x19=361个点）"""
    if corners is not None:
        # 如果有角点信息，使用角点定义的区域
        lt, lb, rt, rb = corners
        
        # 计算棋盘区域
        left_x = min(lt[0], lb[0])
        right_x = max(rt[0], rb[0])
        top_y = min(lt[1], rt[1])
        bottom_y = max(lb[1], rb[1])
        
        # 添加边距，确保只选择内部交叉点
        margin_x = (right_x - left_x) * 0.05  # 5%边距
        margin_y = (bottom_y - top_y) * 0.05
        
        board_left = left_x + margin_x
        board_right = right_x - margin_x
        board_top = top_y + margin_y
        board_bottom = bottom_y - margin_y
        
    else:
        # 如果没有角点信息，尝试自动检测棋盘区域
        left_x, top_y, right_x, bottom_y = detect_board_region(img)
        
        # 添加边距
        margin_x = (right_x - left_x) * 0.1  # 10%边距
        margin_y = (bottom_y - top_y) * 0.1
        
        board_left = left_x + margin_x
        board_right = right_x - margin_x
        board_top = top_y + margin_y
        board_bottom = bottom_y - margin_y
    
    # 生成19x19的交叉点
    step_x = (board_right - board_left) / 18  # 18个间隔，19个点
    step_y = (board_bottom - board_top) / 18
    
    intersection_points = []
    for i in range(19):
        for j in range(19):
            x = board_left + j * step_x
            y = board_top + i * step_y
            intersection_points.append((int(x), int(y), i, j))
    
    print(f"生成了19x19共{len(intersection_points)}个有效交叉点")
    print(f"棋盘区域: ({board_left:.1f}, {board_top:.1f}) 到 ({board_right:.1f}, {board_bottom:.1f})")
    
    return intersection_points

def detect_stones_on_intersections(img, intersection_points):
    """在交叉点上检测棋子"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # 使用HoughCircles检测圆形
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=100,
        param2=18,
        minRadius=8,
        maxRadius=35
    )
    
    stones = []
    result_img = img.copy()
    
    # 绘制所有有效交叉点
    for x, y, _, _ in intersection_points:
        cv2.circle(result_img, (x, y), 2, (0, 255, 255), -1)  # 黄色小点
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f'检测到 {len(circles[0])} 个圆形')
        
        tolerance = 25  # 容差范围
        
        for circle in circles[0, :]:
            cx, cy, r = circle
            
            # 找到最近的交叉点
            min_dist = float('inf')
            closest_intersection = None
            
            for x, y, row, col in intersection_points:
                dist = math.sqrt((cx - x)**2 + (cy - y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_intersection = (x, y, row, col)
            
            # 如果圆心离交叉点足够近，认为是有效棋子
            if min_dist < tolerance and closest_intersection:
                gx, gy, grow, gcol = closest_intersection
                
                # 颜色判断：分析圆形区域的平均亮度
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]
                
                # 根据亮度判断黑白子
                color = 'black' if mean_val < 100 else 'white'
                stones.append((color, grow, gcol))
                
                # 绘制检测结果
                circle_color = (0, 0, 255) if color == 'black' else (255, 255, 255)
                cv2.circle(result_img, (cx, cy), r, circle_color, 2)
                cv2.circle(result_img, (cx, cy), 2, circle_color, -1)
                
                # 在交叉点上标记（绿色圆圈表示有棋子的交叉点）
                cv2.circle(result_img, (gx, gy), 8, (0, 255, 0), 2)
                
                # 添加文字标注
                text = f"{color[0].upper()}({grow+1},{gcol+1})"
                text_color = (255, 255, 255) if color == 'black' else (0, 0, 0)
                cv2.putText(result_img, text, (cx-30, cy-r-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    return stones, result_img

def print_board_state(stones):
    """打印棋盘状态"""
    board_state = np.zeros((19, 19), dtype=int)
    
    for color, row, col in stones:
        if 0 <= row < 19 and 0 <= col < 19:
            board_state[row][col] = 1 if color == 'black' else 2
            print(f"{color} 棋子在位置: ({row+1}, {col+1})")
    
    print("\n围棋棋盘状态 (0=空, 1=黑子, 2=白子):")
    print("列号:", end="  ")
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

def main():
    # 读取图像
    img_path = '../data/raw/IMG20171015161921.jpg'  # 修改为你的图像路径
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"无法读取图像文件: {img_path}")
        print("请检查路径是否正确")
        return
    
    print("原始图像大小:", img.shape)
    cv2.imshow('original', img)
    
    # 1. 寻找棋盘角点
    print("正在寻找棋盘角点...")
    corners = find_board_corners(img)
    
    if corners is not None:
        lt, lb, rt, rb = corners
        print(f"找到角点:")
        print(f"  左上: {lt}")
        print(f"  左下: {lb}")
        print(f"  右上: {rt}")
        print(f"  右下: {rb}")
        
        # 在原图上标记角点
        corner_img = img.copy()
        corners_list = [lt, lb, rt, rb]
        corner_labels = ['LT', 'LB', 'RT', 'RB']
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        
        for i, (corner, label, color) in enumerate(zip(corners_list, corner_labels, colors)):
            cv2.circle(corner_img, tuple(map(int, corner)), 10, color, -1)
            cv2.putText(corner_img, label, (int(corner[0])-20, int(corner[1])-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow('corners', corner_img)
        
        # 判断是否需要透视校正
        if is_perspective_correction_needed(img, corners):
            print("正在进行透视变换...")
            processed_img, transform_matrix = apply_perspective_transform(img, corners)
            cv2.imshow('warped_board', processed_img)
            # 透视变换后重新计算交叉点
            intersection_points = create_valid_intersection_points(processed_img)
        else:
            print("图像不需要透视校正，直接使用原图")
            processed_img = img
            intersection_points = create_valid_intersection_points(img, corners)
    else:
        print("未找到明显的棋盘角点，尝试自动检测棋盘区域")
        processed_img = img
        intersection_points = create_valid_intersection_points(img)
    
    # 2. 在处理后的图像上检测棋子
    print("正在检测棋子...")
    stones, result_img = detect_stones_on_intersections(processed_img, intersection_points)
    
    print(f"检测到 {len(stones)} 个棋子")
    
    # 3. 显示结果
    cv2.imshow('final_result', result_img)
    
    # 4. 打印棋盘状态
    if stones:
        print_board_state(stones)
    else:
        print("未检测到任何棋子")
    
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()