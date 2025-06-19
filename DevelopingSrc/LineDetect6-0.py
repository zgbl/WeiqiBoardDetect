import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform

def detect_board_corners(img):
    """
    检测棋盘的四个角点
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 预处理
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    equalized = cv2.equalizeHist(blur)
    
    # 使用Harris角点检测
    corners_harris = cv2.cornerHarris(equalized, blockSize=3, ksize=3, k=0.04)
    
    # 也使用goodFeaturesToTrack作为备选
    corners_good = cv2.goodFeaturesToTrack(equalized, 
                                         maxCorners=100,
                                         qualityLevel=0.01,
                                         minDistance=30,
                                         useHarrisDetector=True,
                                         k=0.04)
    
    # 找到Harris响应的局部最大值
    harris_threshold = 0.01 * corners_harris.max()
    corner_candidates = []
    
    # 非极大值抑制
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(corners_harris, kernel)
    local_maxima = (corners_harris == dilated) & (corners_harris > harris_threshold)
    
    corner_points = np.where(local_maxima)
    for y, x in zip(corner_points[0], corner_points[1]):
        corner_candidates.append((x, y, corners_harris[y, x]))
    
    # 按响应强度排序
    corner_candidates.sort(key=lambda x: x[2], reverse=True)
    
    print(f"检测到 {len(corner_candidates)} 个角点候选")
    
    # 可视化所有角点候选
    img_corners = img.copy()
    for i, (x, y, response) in enumerate(corner_candidates[:20]):  # 显示前20个
        color = (0, 255, 0) if i < 4 else (0, 255, 255)
        cv2.circle(img_corners, (int(x), int(y)), 5, color, -1)
        cv2.putText(img_corners, f"{i}", (int(x)+8, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imshow("All Corner Candidates", img_corners)
    cv2.moveWindow("All Corner Candidates", 0, 400)
    
    return find_board_corners_from_candidates(corner_candidates, img.shape)

def find_board_corners_from_candidates(corner_candidates, img_shape):
    """
    从角点候选中找到棋盘的四个角
    """
    if len(corner_candidates) < 4:
        return None
    
    height, width = img_shape[:2]
    
    # 方法1: 找到最外围的四个点
    points = [(x, y) for x, y, _ in corner_candidates[:min(20, len(corner_candidates))]]
    
    # 计算凸包
    points_array = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points_array)
    hull_points = hull.reshape(-1, 2)
    
    # 如果凸包点数>=4，选择四个最外围的点
    if len(hull_points) >= 4:
        board_corners = select_four_corners(hull_points)
        if board_corners is not None:
            return board_corners
    
    # 方法2: 按区域选择（左上，右上，左下，右下）
    board_corners = select_corners_by_region(points, width, height)
    
    return board_corners

def select_four_corners(hull_points):
    """
    从凸包点中选择四个最合适的角点（形成最大的矩形）
    """
    if len(hull_points) < 4:
        return None
    
    # 如果正好4个点，直接返回
    if len(hull_points) == 4:
        return order_corners(hull_points)
    
    # 如果超过4个点，选择形成最大面积四边形的4个点
    from itertools import combinations
    
    max_area = 0
    best_corners = None
    
    # 尝试所有4点组合
    for combo in combinations(hull_points, 4):
        area = cv2.contourArea(np.array(combo, dtype=np.float32))
        if area > max_area:
            max_area = area
            best_corners = combo
    
    if best_corners is not None:
        return order_corners(np.array(best_corners))
    
    return None

def select_corners_by_region(points, width, height):
    """
    按图像区域选择四个角点
    """
    # 将图像分为四个区域，每个区域选择一个最强的角点
    regions = {
        'top_left': [],
        'top_right': [],
        'bottom_left': [],
        'bottom_right': []
    }
    
    mid_x, mid_y = width // 2, height // 2
    
    for x, y in points:
        if x < mid_x and y < mid_y:
            regions['top_left'].append((x, y))
        elif x >= mid_x and y < mid_y:
            regions['top_right'].append((x, y))
        elif x < mid_x and y >= mid_y:
            regions['bottom_left'].append((x, y))
        else:
            regions['bottom_right'].append((x, y))
    
    # 每个区域选择最外围的点
    corners = []
    
    # 左上角：选择x+y最小的
    if regions['top_left']:
        corner = min(regions['top_left'], key=lambda p: p[0] + p[1])
        corners.append(corner)
    
    # 右上角：选择x-y最大的
    if regions['top_right']:
        corner = max(regions['top_right'], key=lambda p: p[0] - p[1])
        corners.append(corner)
    
    # 左下角：选择y-x最大的
    if regions['bottom_left']:
        corner = max(regions['bottom_left'], key=lambda p: p[1] - p[0])
        corners.append(corner)
    
    # 右下角：选择x+y最大的
    if regions['bottom_right']:
        corner = max(regions['bottom_right'], key=lambda p: p[0] + p[1])
        corners.append(corner)
    
    if len(corners) == 4:
        return order_corners(np.array(corners, dtype=np.float32))
    
    return None

def order_corners(corners):
    """
    将四个角点按照 [左上, 右上, 右下, 左下] 的顺序排列
    """
    # 计算中心点
    center = np.mean(corners, axis=0)
    
    # 按相对于中心点的角度排序
    def angle_from_center(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])
    
    # 按角度排序
    corners_with_angles = [(corner, angle_from_center(corner)) for corner in corners]
    corners_with_angles.sort(key=lambda x: x[1])
    
    # 重新排列为 [左上, 右上, 右下, 左下]
    ordered = [corner for corner, _ in corners_with_angles]
    
    # 调整顺序：从左上角开始
    # 找到最左上的点（x+y最小）
    top_left_idx = min(range(4), key=lambda i: ordered[i][0] + ordered[i][1])
    
    # 重新排列，使左上角在第一位
    reordered = ordered[top_left_idx:] + ordered[:top_left_idx]
    
    return np.array(reordered, dtype=np.float32)

def generate_grid_from_corners(corners, n_lines=9):
    """
    根据四个角点生成规整的网格线
    """
    if corners is None or len(corners) != 4:
        return [], []
    
    # corners 应该是 [左上, 右上, 右下, 左下]
    top_left, top_right, bottom_right, bottom_left = corners
    
    print(f"棋盘角点:")
    print(f"  左上: ({top_left[0]:.1f}, {top_left[1]:.1f})")
    print(f"  右上: ({top_right[0]:.1f}, {top_right[1]:.1f})")
    print(f"  右下: ({bottom_right[0]:.1f}, {bottom_right[1]:.1f})")
    print(f"  左下: ({bottom_left[0]:.1f}, {bottom_left[1]:.1f})")
    
    horizontal_lines = []
    vertical_lines = []
    
    # 生成水平线
    for i in range(n_lines):
        t = i / (n_lines - 1)  # 参数从0到1
        
        # 左边界点（从左上到左下）
        left_x = top_left[0] + t * (bottom_left[0] - top_left[0])
        left_y = top_left[1] + t * (bottom_left[1] - top_left[1])
        
        # 右边界点（从右上到右下）
        right_x = top_right[0] + t * (bottom_right[0] - top_right[0])
        right_y = top_right[1] + t * (bottom_right[1] - top_right[1])
        
        horizontal_lines.append((int(left_x), int(left_y), int(right_x), int(right_y)))
    
    # 生成垂直线
    for i in range(n_lines):
        t = i / (n_lines - 1)  # 参数从0到1
        
        # 上边界点（从左上到右上）
        top_x = top_left[0] + t * (top_right[0] - top_left[0])
        top_y = top_left[1] + t * (top_right[1] - top_left[1])
        
        # 下边界点（从左下到右下）
        bottom_x = bottom_left[0] + t * (bottom_right[0] - bottom_left[0])
        bottom_y = bottom_left[1] + t * (bottom_right[1] - bottom_left[1])
        
        vertical_lines.append((int(top_x), int(top_y), int(bottom_x), int(bottom_y)))
    
    return horizontal_lines, vertical_lines

def validate_board_corners(corners, img_shape):
    """
    验证检测到的角点是否合理
    """
    if corners is None or len(corners) != 4:
        return False
    
    height, width = img_shape[:2]
    
    # 检查角点是否在图像边界内
    for corner in corners:
        x, y = corner
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
    
    # 检查是否形成合理的四边形
    area = cv2.contourArea(corners)
    img_area = width * height
    
    # 面积应该是图像面积的一个合理比例
    if area < img_area * 0.1 or area > img_area * 0.9:
        return False
    
    # 检查四边形是否过于扭曲
    # 计算对角线长度比例
    diag1 = np.linalg.norm(corners[0] - corners[2])
    diag2 = np.linalg.norm(corners[1] - corners[3])
    
    if diag1 == 0 or diag2 == 0:
        return False
    
    ratio = max(diag1, diag2) / min(diag1, diag2)
    if ratio > 3:  # 对角线长度比例不应该过大
        return False
    
    return True

# 主程序
def main():
    # 读取图像
    img = cv2.imread("../data/raw/Pic1.jpg")
    if img is None:
        print("无法读取图像")
        return
    
    # 显示原图
    cv2.imshow("Original", img)
    cv2.moveWindow("Original", 0, 0)
    
    # 检测棋盘角点
    corners = detect_board_corners(img)
    
    if corners is not None and validate_board_corners(corners, img.shape):
        print("成功检测到棋盘角点！")
        
        # 生成网格
        h_lines, v_lines = generate_grid_from_corners(corners, n_lines=9)
        
        # 绘制结果
        img_result = img.copy()
        
        # 绘制角点
        for i, corner in enumerate(corners):
            cv2.circle(img_result, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
            cv2.putText(img_result, f"C{i+1}", 
                       (int(corner[0])+10, int(corner[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 绘制网格线
        for x1, y1, x2, y2 in h_lines + v_lines:
            cv2.line(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow("Board Detection Result", img_result)
        cv2.moveWindow("Board Detection Result", 400, 0)
        
        print(f"生成了 {len(h_lines)} 条水平线和 {len(v_lines)} 条垂直线")
        
    else:
        print("未能检测到有效的棋盘角点")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()