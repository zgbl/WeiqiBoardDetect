import cv2
import numpy as np

def preprocess_for_wood_grain(img):
    """
    专门针对木纹棋盘的预处理
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 多尺度模糊，减少木纹干扰
    blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 2. 形态学操作，强化棋盘边缘
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(blur1, cv2.MORPH_GRADIENT, kernel)
    
    # 3. 自适应阈值，处理光照不均
    adaptive = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 4. 多种边缘检测方法
    edges1 = cv2.Canny(blur1, 20, 60)  # 降低阈值
    edges2 = cv2.Canny(blur2, 30, 80)  # 不同参数
    edges3 = cv2.Canny(morph, 50, 150)  # 形态学增强后
    
    # 5. 组合边缘
    combined_edges = cv2.bitwise_or(edges1, edges2)
    combined_edges = cv2.bitwise_or(combined_edges, edges3)
    
    # 6. 闭运算，连接断开的边缘
    kernel_close = np.ones((3,3), np.uint8)
    final_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_close)
    
    return gray, final_edges, [blur1, blur2, morph, adaptive, edges1, edges2, edges3, combined_edges]

def find_board_contours_robust(edges):
    """
    更鲁棒的棋盘轮廓检测
    """
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for contour in contours:
        # 计算面积
        area = cv2.contourArea(contour)
        if area < 50000:  # 过滤小轮廓
            continue
            
        # 凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # 计算轮廓的凸性（convexity）
        convexity = area / hull_area if hull_area > 0 else 0
        
        # 尝试不同的epsilon值
        perimeter = cv2.arcLength(hull, True)
        for epsilon_factor in [0.02, 0.05, 0.08, 0.1, 0.15]:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(approx) == 4:
                # 检查是否为凸四边形
                if cv2.isContourConvex(approx):
                    # 计算四边形的质量评分
                    points = np.reshape(approx, (4, 2))
                    score = evaluate_quadrilateral_quality(points, area, convexity)
                    candidates.append((points, area, score, epsilon_factor))
    
    # 按评分排序
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    return candidates

def evaluate_quadrilateral_quality(points, area, convexity):
    """
    评估四边形作为棋盘的质量
    """
    score = 0
    
    # 1. 面积分数（面积越大越好）
    score += min(area / 100000, 10)
    
    # 2. 凸性分数
    score += convexity * 5
    
    # 3. 形状分数（接近矩形的程度）
    # 计算四条边的长度
    sides = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        side_length = np.linalg.norm(p2 - p1)
        sides.append(side_length)
    
    # 对边应该相等
    opposite_ratio1 = min(sides[0], sides[2]) / max(sides[0], sides[2])
    opposite_ratio2 = min(sides[1], sides[3]) / max(sides[1], sides[3])
    score += (opposite_ratio1 + opposite_ratio2) * 2
    
    # 4. 角度分数（接近90度）
    angles = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        p3 = points[(i + 2) % 4]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        angles.append(angle)
    
    # 角度应该接近90度（π/2）
    angle_score = sum([1 - abs(angle - np.pi/2) / (np.pi/2) for angle in angles])
    score += angle_score
    
    return score

def order_corners_improved(points):
    """
    改进的角点排序，更准确地识别左上角
    """
    points = np.array(points)
    
    # 计算质心
    center = np.mean(points, axis=0)
    
    # 计算每个点相对于质心的角度
    angles = []
    for point in points:
        angle = np.arctan2(point[1] - center[1], point[0] - center[0])
        angles.append(angle)
    
    # 按角度排序
    sorted_indices = np.argsort(angles)
    
    # 找到最左上的点作为起始点
    leftmost_top = None
    min_sum = float('inf')
    start_idx = 0
    
    for i, idx in enumerate(sorted_indices):
        point = points[idx]
        # 使用 x + y 最小的点作为左上角
        if point[0] + point[1] < min_sum:
            min_sum = point[0] + point[1]
            leftmost_top = point
            start_idx = i
    
    # 从左上角开始，顺时针排列
    ordered_indices = []
    for i in range(4):
        ordered_indices.append(sorted_indices[(start_idx + i) % 4])
    
    ordered_points = points[ordered_indices]
    
    # 验证顺序是否正确
    print("角点排序验证：")
    labels = ['左上', '右上', '右下', '左下']
    for i, (point, label) in enumerate(zip(ordered_points, labels)):
        print(f"  {label}: ({point[0]:.0f}, {point[1]:.0f})")
    
    return ordered_points.astype("float32")

def line_intersection(line1, line2):
    """计算两条直线的交点"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)
    return (x, y)

def fix_cropped_corners(corners):
    """
    修复被裁剪的角点
    """
    # 检查是否有角点在图像边缘附近
    image_width = 2000  # 假设的图像宽度，实际使用时传入真实值
    
    fixed_corners = corners.copy()
    
    # 检查左上角是否被裁剪（x坐标很小）
    if corners[0][0] < 30:
        print("检测到左上角被裁剪，进行修复...")
        
        # 使用平行四边形性质修复
        # 在平行四边形中：left_top = right_top + (left_bottom - right_bottom)
        vector_correction = corners[3] - corners[2]  # left_bottom - right_bottom
        corrected_left_top = corners[1] + vector_correction  # right_top + correction
        
        fixed_corners[0] = corrected_left_top
        print(f"修复后的左上角: ({corrected_left_top[0]:.1f}, {corrected_left_top[1]:.1f})")
    
    return fixed_corners

# 主函数
def detect_board_robust(image_path):
    """
    鲁棒的棋盘检测主函数
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    print(f"图像尺寸: {img.shape[1]} x {img.shape[0]}")
    
    # 预处理
    gray, edges, debug_images = preprocess_for_wood_grain(img)
    
    # 显示预处理结果
    cv2.imshow('Original', cv2.resize(img, (800, 600)))
    cv2.imshow('Final Edges', edges)
    
    # 寻找轮廓
    candidates = find_board_contours_robust(edges)
    
    if not candidates:
        print("未找到合适的棋盘轮廓！")
        return None
    
    print(f"找到 {len(candidates)} 个候选轮廓")
    
    # 选择最佳候选
    best_points, best_area, best_score, best_epsilon = candidates[0]
    print(f"最佳候选: 面积={best_area:.0f}, 评分={best_score:.2f}, epsilon={best_epsilon:.3f}")
    
    # 排序角点
    ordered_corners = order_corners_improved(best_points)
    
    # 修复被裁剪的角点
    final_corners = fix_cropped_corners(ordered_corners)
    
    # 可视化结果
    result_img = img.copy()
    colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]
    labels = ['LT', 'RT', 'RB', 'LB']
    
    for i, (corner, color, label) in enumerate(zip(final_corners, colors, labels)):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(result_img, (x, y), 15, color, -1)
        cv2.putText(result_img, label, (x+20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    cv2.imshow('Detected Corners', result_img)
    
    return final_corners, result_img

# 使用示例
if __name__ == "__main__":
    image_path = "../data/raw/Homeboard4.jpg"
    
    result = detect_board_robust(image_path)
    
    if result is not None:
        corners, result_img = result
        
        print("\n最终棋盘角点:")
        labels = ['左上', '右上', '右下', '左下']
        for i, (corner, label) in enumerate(zip(corners, labels)):
            print(f"  {label}: ({corner[0]:.1f}, {corner[1]:.1f})")
        
        # 透视变换
        output_size = 1000
        margin = 50
        
        dst_points = np.float32([
            [margin, margin],
            [output_size - margin, margin],
            [output_size - margin, output_size - margin],
            [margin, output_size - margin]
        ])
        
        M = cv2.getPerspectiveTransform(corners, dst_points)
        
        # 读取原图进行变换
        original = cv2.imread(image_path)
        warped = cv2.warpPerspective(original, M, (output_size, output_size))
        
        cv2.imshow('Warped Board', warped)
        cv2.imwrite('warped_board_robust.jpg', warped)
        print("变换后的棋盘已保存为 warped_board_robust.jpg")
    
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()