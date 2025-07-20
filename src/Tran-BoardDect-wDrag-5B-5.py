# 精简清理了代码的版本。可用。
import cv2
import numpy as np
from collections import defaultdict
import math
#from Detectpiece import detect_pieces_by_regions
import Detectpiece

class InteractiveCornerAdjuster:
    """
    交互式角点调整器，允许用户拖动和确认棋盘角点。
    """
    def __init__(self, image, initial_corners):
        """
        初始化交互式角点调整器
        Args:
            image: 原始图像
            initial_corners: 初始角点 (lt, lb, rt, rb)
        """
        self.original_image = image.copy()
        self.display_image = image.copy()
        self.corners = list(initial_corners)
        self.corner_labels = ['左上', '左下', '右上', '右下']
        self.corner_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255)]

        self.dragging = False
        self.selected_corner = -1
        self.corner_radius = 15
        self.confirmed = False

        self.window_name = "角点调整 - 拖动角点后按空格确认，ESC取消"

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.update_display()

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，处理角点拖动事件"""
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, corner in enumerate(self.corners):
                dist = math.sqrt((x - corner[0])**2 + (y - corner[1])**2)
                if dist < self.corner_radius:
                    self.dragging = True
                    self.selected_corner = i
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_corner != -1:
                self.corners[self.selected_corner] = [x, y]
                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.selected_corner = -1

    def update_display(self):
        """更新显示图像，绘制角点和提示信息"""
        self.display_image = self.original_image.copy()

        if len(self.corners) == 4:
            pts = np.array([self.corners[0], self.corners[2], self.corners[3], self.corners[1]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(self.display_image, [pts], True, (0, 0, 255), 2)

        for i, (corner, label, color) in enumerate(zip(self.corners, self.corner_labels, self.corner_colors)):
            cv2.circle(self.display_image, tuple(corner), self.corner_radius, color, -1)
            cv2.circle(self.display_image, tuple(corner), self.corner_radius, (0, 0, 0), 2)
            cv2.putText(self.display_image, label,
                       (corner[0] - 20, corner[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(self.display_image, f"({corner[0]},{corner[1]})",
                       (corner[0] - 30, corner[1] + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(self.display_image, "拖动角点调整位置",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(self.display_image, "空格键确认 | ESC取消",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(self.window_name, self.display_image)

    def validate_corners(self):
        """验证角点是否合理，包括数量、是否在图像范围内、以及形成区域的大小"""
        if len(self.corners) != 4:
            return False, "角点数量不正确"

        height, width = self.original_image.shape[:2]
        for i, corner in enumerate(self.corners):
            if corner[0] < 0 or corner[0] >= width or corner[1] < 0 or corner[1] >= height:
                return False, f"角点 {self.corner_labels[i]} 超出图像范围"

        pts = np.array([self.corners[0], self.corners[2], self.corners[3], self.corners[1]], np.int32)
        area = cv2.contourArea(pts)
        img_area = width * height

        if area < img_area * 0.01:
            return False, "选择区域太小"

        if area > img_area * 0.95:
            return False, "选择区域太大"

        return True, "角点验证通过"

    def run(self):
        """运行交互式调整界面，等待用户操作"""
        print("开始交互式角点调整...")
        print("操作说明：")
        print("- 用鼠标拖动角点进行调整")
        print("- 按空格键确认调整结果")
        print("- 按ESC键取消调整")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # 空格键确认
                is_valid, message = self.validate_corners()
                if is_valid:
                    print(f"角点调整完成: {message}")
                    self.confirmed = True
                    break
                else:
                    print(f"角点验证失败: {message}")
                    continue

            elif key == 27:  # ESC键取消
                print("取消角点调整")
                self.confirmed = False
                break

        cv2.destroyWindow(self.window_name)
        return self.confirmed, tuple(self.corners)

def interactive_corner_adjustment(image, initial_corners):
    """
    交互式角点调整的主函数
    Args:
        image: 原始图像
        initial_corners: 初始角点 (lt, lb, rt, rb)
    Returns:
        confirmed: 是否确认调整
        final_corners: 最终角点坐标
    """
    adjuster = InteractiveCornerAdjuster(image, initial_corners)
    return adjuster.run()

def auto_resize_image(image, target_width):
    """
    自动缩放图片到合适的尺寸进行处理
    Args:
        image: 原始图像
        target_width: 目标宽度
    Returns:
        resized_img: 缩放后的图像
        scale_factor: 缩放比例
    """
    height, width = image.shape[:2]
    print(f"原图尺寸: {width} x {height}")

    if width > 2500:
        scale_factor = int((width * 10 / target_width))
        new_width = width * 10 // scale_factor
        new_height = height * 10 // scale_factor
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"缩放后尺寸: {new_width} x {new_height} (缩放比例: {scale_factor:.3f})")
        return resized_img, scale_factor
    else:
        print("图片尺寸合适，无需缩放")
        return image, 1.0

def improve_edge_detection(img):
    """
    改进的边缘检测，减少图片边界干扰
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    mask = np.ones_like(gray) * 255
    border_size = max(10, min(width, height) // 50)
    mask[:border_size, :] = 0
    mask[-border_size:, :] = 0
    mask[:, :border_size] = 0
    mask[:, -border_size:] = 0

    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blurred, 50, 100)
    edges = cv2.bitwise_and(edges, mask)

    return edges, mask

def find_board_contour_improved(contours, img_shape):
    """
    改进的棋盘轮廓查找，避免选择整个图片边界，并进行合理性检查
    """
    height, width = img_shape[:2]
    img_area = height * width

    candidates = []

    for item in contours:
        hull = cv2.convexHull(item)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            ps = np.reshape(approx, (4,2))
            area = cv2.contourArea(approx)

            # 排除过大或过小的轮廓
            if area > img_area * 0.8 or area < img_area * 0.05:
                continue

            # 检查四个角点是否太接近图片边界
            margin = min(width, height) * 0.02
            too_close_to_edge = False
            for point in ps:
                x, y = point
                if (x < margin or x > width - margin or
                    y < margin or y > height - margin):
                    too_close_to_edge = True
                    break
            if too_close_to_edge:
                continue

            # 检查四边形的长宽比是否合理
            x_coords = ps[:, 0]
            y_coords = ps[:, 1]
            width_approx = max(x_coords) - min(x_coords)
            height_approx = max(y_coords) - min(y_coords)
            aspect_ratio = max(width_approx, height_approx) / min(width_approx, height_approx)
            if aspect_ratio > 2.0:
                continue

            candidates.append((approx, area, aspect_ratio))

    if not candidates:
        return None

    # 选择面积最大但长宽比合理的候选
    candidates.sort(key=lambda x: (-x[1], x[2]))
    best_candidate = candidates[0]

    # 重新整理角点顺序 (左上, 左下, 右上, 右下)
    ps = np.reshape(best_candidate[0], (4,2))
    ps = ps[np.lexsort((ps[:,0],))]
    lt, lb = ps[:2][np.lexsort((ps[:2,1],))]
    rt, rb = ps[2:][np.lexsort((ps[2:,1],))]

    return (lt, lb, rt, rb)

def validate_board_corners(corners, img):
    """
    验证检测到的棋盘角点是否合理，检查对角线和边长比例
    """
    if corners is None:
        return False

    lt, lb, rt, rb = corners

    diag1 = np.sqrt((rt[0] - lb[0])**2 + (rt[1] - lb[1])**2)
    diag2 = np.sqrt((lt[0] - rb[0])**2 + (lt[1] - rb[1])**2)
    diag_ratio = max(diag1, diag2) / min(diag1, diag2)
    if diag_ratio > 1.5:
        return False

    side_lengths = [
        np.sqrt((rt[0] - lt[0])**2 + (rt[1] - lt[1])**2),
        np.sqrt((rb[0] - rt[0])**2 + (rb[1] - rb[1])**2),
        np.sqrt((lb[0] - rb[0])**2 + (lb[1] - rb[1])**2),
        np.sqrt((lt[0] - lb[0])**2 + (lt[1] - lb[1])**2)
    ]
    min_side = min(side_lengths)
    max_side = max(side_lengths)
    side_ratio = max_side / min_side
    if side_ratio > 1.8:
        return False

    return True

def line_intersection(line1, line2):
    """计算两条线的交点"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10: # 接近0表示平行或重合
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (int(x), int(y))

def merge_lines(lines, is_horizontal=True, tolerance=10):
    """合并相近的平行线"""
    if not lines:
        return []

    merged = []
    
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

# V5-2 版本，
# 读取图像
#img = cv2.imread('../data/raw/bd317d54.webp')
#img = cv2.imread('../data/raw/IMG20171015161921.jpg')
#img = cv2.imread('../data/raw/OGS3.jpeg')
img = cv2.imread('../data/raw/IMG20160706171004.jpg')
#img = cv2.imread('../data/raw/IMG20161205130156-16.jpg') # 这个图片现在不能检测
#img = cv2.imread('../data/raw/IMG20160904165505-B.jpg')
#img = cv2.imread('../data/raw/IMG20160706171004-12.jpg')
#img = cv2.imread('../data/raw/WechatIMG123.jpg')
#img = cv2.imread('../data/raw/WechatIMG124.jpg')
#img = cv2.imread('../data/raw/ogs_empty2c.jpg')
#img = cv2.imread('../data/raw/WechatIMG123.jpg')


if img is None:
    print("没找到照片")
    exit()

original_img = img.copy()
cv2.imshow("Origin Picture", original_img)
cv2.moveWindow("Origin Picture", 0, 0)

# 自动缩放图片
img, scale_factor = auto_resize_image(img, target_width=1500)

# 根据图片大小选择参数集
current_width = img.shape[1]
if current_width <= 1000:
    print("使用小图参数集")
    perspective_size = 400
    hough_threshold = 40
    min_line_length = 50
    max_line_gap = 8
    line_filter_length = 25
    merge_tolerance_val = 8
    row_tolerance = 12
    min_dist_threshold = 12
    min_radius = 4
    max_radius = 15
    min_dist_circles = 12
    min_points_per_row = 8
elif current_width <= 1500:
    print("使用中图参数集")
    perspective_size = 510
    hough_threshold = 60
    min_line_length = 70
    max_line_gap = 10
    line_filter_length = 35
    merge_tolerance_val = 20
    row_tolerance = 15
    min_dist_threshold = 15
    min_radius = 6
    max_radius = 20
    min_dist_circles = 15
    min_points_per_row = 12
else:
    print("使用大图参数集")
    perspective_size = 660
    hough_threshold = 80
    min_line_length = 100
    max_line_gap = 10
    line_filter_length = 50
    merge_tolerance_val = 15
    row_tolerance = 20
    min_dist_threshold = 20
    min_radius = 8
    max_radius = 25
    min_dist_circles = 20
    min_points_per_row = 15

print(f"参数设置: perspective_size={perspective_size}, hough_threshold={hough_threshold}, ...")


# 棋盘角点检测尝试多种方案
rect = None
im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 定义 im_gray 用于后续使用

# 方案1: 改进的边缘检测
print("开始改进的透视变换...")
print("尝试方案1: 改进的边缘检测...")
im_edge_improved, mask = improve_edge_detection(img)
cv2.imshow('edge_improved', im_edge_improved)
cv2.moveWindow("edge_improved", 300, 0)
contours, hierarchy = cv2.findContours(im_edge_improved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"方案1找到 {len(contours)} 个轮廓")
rect = find_board_contour_improved(contours, img.shape)
if rect is not None and not validate_board_corners(rect, img):
    print("方案1角点验证失败")
    rect = None

# 方案2: 原始Canny + 轻微dilate (如果方案1失败)
if rect is None:
    print("尝试方案2: 原始Canny + 轻微dilate...")
    im_gray_blur = cv2.GaussianBlur(im_gray, (3,3), 0) # 使用 im_gray
    im_edge_original = cv2.Canny(im_gray_blur, 50, 100)
    kernel = np.ones((2, 2), np.uint8)
    im_edge_dilated = cv2.dilate(im_edge_original, kernel, iterations=1)
    cv2.imshow('edge_dilated', im_edge_dilated)
    cv2.moveWindow("edge_dilated", 600, 0)
    contours2, hierarchy2 = cv2.findContours(im_edge_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"方案2找到 {len(contours2)} 个轮廓")
    rect = find_board_contour_improved(contours2, img.shape)
    if rect is not None and not validate_board_corners(rect, img):
        print("方案2角点验证失败")
        rect = None

# 方案3: 宽松条件的原始方法 (如果方案2仍然失败)
if rect is None:
    print("尝试方案3: 宽松条件的原始方法...")
    # 确保 im_edge_original 在此作用域内已定义
    if 'im_edge_original' not in locals():
        im_edge_original = cv2.Canny(cv2.GaussianBlur(im_gray, (3,3), 0), 50, 100)
    contours3, hierarchy3 = cv2.findContours(im_edge_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area = 0
    temp_rect = None # 为此方案使用临时 rect
    for item in contours3:
        hull = cv2.convexHull(item)
        epsilon = 0.1 * cv2.arcLength(hull, True) # 更宽松的epsilon
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            ps = np.reshape(approx, (4,2))
            a = cv2.contourArea(approx)
            img_area = img.shape[0] * img.shape[1]
            if a > img_area * 0.95: # 只排除明显的整图边界
                continue
            if a > area:
                area = a
                ps = ps[np.lexsort((ps[:,0],))]
                lt, lb = ps[:2][np.lexsort((ps[:2,1],))]
                rt, rb = ps[2:][np.lexsort((ps[2:,1],))]
                temp_rect = (lt, lb, rt, rb)
    if temp_rect is not None:
        rect = temp_rect
        print("方案3找到棋盘角点。")


# 定义透视变换相关参数，确保在任何情况下都已定义
padding = 10
extra_pixels = 3
# 初始化 img_processed 和 gray_for_hough 为原始图像作为后备
img_processed = original_img.copy()
gray_for_hough = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# 默认的 perspective_withpadding_size，如果未进行透视变换
perspective_withpadding_size = min(img_processed.shape[0], img_processed.shape[1])


# 根据是否找到棋盘来决定后续处理
if rect is None:
    print('未能找到棋盘！将使用原图进行后续处理，但棋盘线和棋子检测可能不准确。')
    # 参数已设置为后备值
else:
    print('棋盘坐标：')
    print('\t左上角:(%d,%d)'%(rect[0][0],rect[0][1]))
    print('\t左下角:(%d,%d)'%(rect[1][0],rect[1][1]))
    print('\t右上角:(%d,%d)'%(rect[2][0],rect[2][1]))
    print('\t右下角:(%d,%d)'%(rect[3][0],rect[3][1]))

    # 显示自动检测到的角点
    corner_img = img.copy()
    for p in rect:
        cv2.circle(corner_img,(p[0],p[1]),15,(0,255,0),-1)
    cv2.imshow('Auto_Detected_corner_img', corner_img)
    cv2.moveWindow("Auto_Detected_corner_img", 900, 0)

    # === 添加交互式调整 ===
    print("开始交互式角点调整...")
    confirmed, adjusted_corners = interactive_corner_adjustment(img, rect)

    if confirmed:
        rect = adjusted_corners
        print("用户确认了调整后的角点")
        print('最终角点：')
        print(f'\t左上角:({rect[0][0]},{rect[0][1]})')
        print(f'\t左下角:({rect[1][0]},{rect[1][1]})')
        print(f'\t右上角:({rect[2][0]},{rect[2][1]})')
        print(f'\t右下角:({rect[3][0]},{rect[3][1]})')
    else:
        print("用户取消了调整，使用原始检测结果")

    # === 调试代码：在原始图像上绘制检测到的棋盘边界 ===
    """debug_original_with_rect = original_img.copy()
    lt, lb, rt, rb = rect
    pts_for_poly = np.array([lt, rt, rb, lb], np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug_original_with_rect, [pts_for_poly], True, (0, 0, 255), 3)
    cv2.circle(debug_original_with_rect, tuple(lt), 15, (0, 255, 255), -1)
    cv2.circle(debug_original_with_rect, tuple(rt), 15, (0, 255, 255), -1)
    cv2.circle(debug_original_with_rect, tuple(rb), 15, (0, 255, 255), -1)
    cv2.circle(debug_original_with_rect, tuple(lb), 15, (0, 255, 255), -1)
    cv2.imshow('debug_original_with_rect', debug_original_with_rect)
    cv2.moveWindow('debug_original_with_rect', 1200, 0)
    """

    # 执行透视变换
    lt, lb, rt, rb = rect # 使用（可能已调整的）rect

    # 根据选择的 perspective_size 和 padding 重新计算 perspective_withpadding_size
    perspective_withpadding_size = perspective_size + 2 * padding

    # 定义透视变换的目标点
    pts1_standard = np.float32([
        [padding - extra_pixels, padding - extra_pixels],
        [padding + perspective_size + extra_pixels, padding - extra_pixels],
        [padding + perspective_size + extra_pixels, padding + perspective_size + extra_pixels],
        [padding - extra_pixels, padding + perspective_size + extra_pixels]
    ])
    # 定义透视变换的源点，并向外扩展
    pts2_ordered = np.float32([
        [lt[0] - extra_pixels, lt[1] - extra_pixels],
        [rt[0] + extra_pixels, rt[1] - extra_pixels],
        [rb[0] + extra_pixels, rb[1] + extra_pixels],
        [lb[0] - extra_pixels, lb[1] + extra_pixels]
    ])

    m = cv2.getPerspectiveTransform(pts2_ordered, pts1_standard)
    print(f"original_img shape: {original_img.shape}")
    #cv2.imwrite("debug_original_img.jpg", original_img)
    cv2.imwrite("debug_original_img.jpg", img)
    #img_processed = cv2.warpPerspective(original_img, m, (perspective_withpadding_size, perspective_withpadding_size))
    #gray_for_hough = cv2.warpPerspective(im_gray, m, (perspective_withpadding_size, perspective_withpadding_size)) # 使用 im_gray
    img_processed = cv2.warpPerspective(img, m, (perspective_withpadding_size, perspective_withpadding_size))
    gray_for_hough = cv2.warpPerspective(im_gray, m, (perspective_withpadding_size, perspective_withpadding_size))

    cv2.imshow('transformed', img_processed)
    cv2.moveWindow("transformed", 1500, 0)
    print("透视变换完成！")

# 使用 img_processed 进行所有后续需要（可能已变换的）图像的操作
# 使用 gray_for_hough 进行所有后续需要（可能已变换的）灰度图像的操作

# 棋盘线和交点计算
print("开始棋盘线检测...")

blur = cv2.GaussianBlur(gray_for_hough, (5, 5), 0)
edges = cv2.Canny(blur, 50, 100)
cv2.imshow('Hough_Edge', edges)
cv2.moveWindow('Hough_Edge', 1300, 0)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

if lines is None:
    print("未检测到任何直线。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

print(f"检测到 {len(lines)} 条直线")

horizontal_lines = []
vertical_lines = []

def line_angle(line):
    """计算直线的角度"""
    x1, y1, x2, y2 = line
    return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

def line_length(line):
    """计算直线的长度"""
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 过滤和分类直线为水平线和垂直线
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = abs(line_angle(line[0]))
    length = line_length(line[0])

    if length > line_filter_length:
        if angle < 10 or angle > 170: # 水平线
            horizontal_lines.append(line[0])
        elif 80 < angle < 100: # 垂直线
            vertical_lines.append(line[0])

print(f"水平线: {len(horizontal_lines)} 条")
print(f"垂直线: {len(vertical_lines)} 条")

# 合并相近的平行线
merged_horizontal = merge_lines(horizontal_lines, True, merge_tolerance_val)
merged_vertical = merge_lines(vertical_lines, False, merge_tolerance_val)

print(f"合并后水平线: {len(merged_horizontal)} 条")
print(f"合并后垂直线: {len(merged_vertical)} 条")

# 计算所有交点
intersections = []
intersection_buffer = 10 # 允许交点略微超出边界的缓冲区

for h_line in merged_horizontal:
    for v_line in merged_vertical:
        intersection = line_intersection(h_line, v_line)
        if intersection:
            # 过滤掉超出图像范围太远的交点
            if (intersection[0] >= -intersection_buffer and
                intersection[0] < perspective_withpadding_size + intersection_buffer and
                intersection[1] >= -intersection_buffer and
                intersection[1] < perspective_withpadding_size + intersection_buffer):
                intersections.append(intersection)

print(f"找到 {len(intersections)} 个交点")

grid_img = img_processed.copy() # 使用 img_processed
for point in intersections:
    cv2.circle(grid_img, point, 3, (0, 0, 255), -1)
cv2.imshow('All_Detected_Intersections', grid_img)
cv2.moveWindow('All_Detected_Intersections', 1500, 0)


# 构建19x19围棋网格
# 动态调整交点阈值和每行最少点数
base_width_for_params = 1350
param_scale = current_width / base_width_for_params if current_width < base_width_for_params else 1.0
intersection_threshold_val = max(50, int(100 * param_scale))

if len(intersections) > intersection_threshold_val:
    print("开始构建19x19围棋网格...")

    # 按行列排序交点
    intersections.sort(key=lambda p: (p[1], p[0]))

    # 按行分组交点
    rows = []
    current_row = []
    last_y = intersections[0][1]

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

    min_points_per_row_val = max(10, int(15 * param_scale))
    valid_rows = [row for row in rows if len(row) >= min_points_per_row_val]
    print(f"有效行数: {len(valid_rows)}")

    if len(valid_rows) >= 18: # 允许少量误差
        # 选择中间的19行
        if len(valid_rows) > 19:
            start_idx = (len(valid_rows) - 19) // 2
            selected_rows = valid_rows[start_idx:start_idx + 19]
        else:
            selected_rows = valid_rows[:19]

        print(f"选择了 {len(selected_rows)} 行构建网格")

        go_grid_points = []
        grid_xy = np.zeros((19, 19, 2), dtype=np.int32)

        # 计算每个格子的预期尺寸
        cell_interval = (perspective_size - 1) / 18.0

        # 构建理想的19x19网格点坐标
        for row_idx in range(19):
            for col_idx in range(19):
                x_ideal = int(col_idx * cell_interval)
                y_ideal = int(row_idx * cell_interval)
                go_grid_points.append((x_ideal, y_ideal, row_idx, col_idx))
                grid_xy[row_idx, col_idx] = [x_ideal, y_ideal]

        final_img = img_processed.copy() # 使用 img_processed

        # 绘制计算出的网格线 (需要加上 padding)
        for r in range(19):
            cv2.line(final_img, tuple((grid_xy[r, 0] + padding).astype(int)), tuple((grid_xy[r, 18] + padding).astype(int)), (0, 0, 0), 1)
        for c in range(19):
            cv2.line(final_img, tuple((grid_xy[0, c] + padding).astype(int)), tuple((grid_xy[18, c] + padding).astype(int)), (0, 0, 0), 1)

        # 绘制计算出的网格点 (需要加上 padding)
        for point_data in go_grid_points:
            cv2.circle(final_img, (point_data[0] + padding, point_data[1] + padding), 2, (0, 255, 255), -1)

        # 重新进行圆形检测，使用透视矫正后的灰度图
        circles = cv2.HoughCircles(gray_for_hough, method=cv2.HOUGH_GRADIENT,  #替换成detected_pieces 不要求棋子是圆形。
                                   dp=1, minDist=min_dist_circles, param1=100, param2=25,  #param2 19>30 7/19
                                   minRadius=min_radius, maxRadius=max_radius) 
        
        #detected_pieces = Detectpiece.detect_pieces_by_regions(corrected_board, grid_points, 
        #                                 grid_spacing_x, grid_spacing_y)  # 参数名都不对
        
        # 重新进行棋子检测，使用透视矫正后的灰度图
        detected_pieces = Detectpiece.detect_pieces_by_regions(img_processed, go_grid_points, 
                                            cell_interval, cell_interval)

        # 转换为原来的格式继续处理
        circles_equivalent = []
        for piece in detected_pieces:
            cx, cy = piece['center']
            intensity = piece['intensity'] 
            color = piece['color']
            circles_equivalent.append((cx, cy, intensity, color))

        stones = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            print(f'检测到 {len(circles[0])} 个圆形')

            for circle in circles[0, :]:
                cx, cy, r = circle

                # 找到最近的围棋网格点
                min_dist = float('inf')
                closest_grid_point_coords = None
                closest_grid_pos_indices = None

                for row_idx in range(19):
                    for col_idx in range(19):
                        # grid_xy 存储的是没有 padding 的理想坐标，cx, cy 是有 padding 的图像坐标
                        x_grid, y_grid = grid_xy[row_idx, col_idx]
                        dist = math.sqrt((cx - (x_grid + padding))**2 + (cy - (y_grid + padding))**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_grid_point_coords = (x_grid + padding, y_grid + padding)
                            closest_grid_pos_indices = (row_idx, col_idx)

                # 如果圆心离最近围棋网格点足够近，认为这是一个有效的棋子
                if closest_grid_point_coords is not None and min_dist < min_dist_threshold:
                    # 颜色检测
                    roi = gray_for_hough[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
                    if roi.size > 0:
                        # 使用中心区域的强度，排除边缘噪声
                        center_roi = gray_for_hough[max(0, cy-r//2):cy+r//2, max(0, cx-r//2):cx+r//2]
                        if center_roi.size > 0:
                            avg_intensity = np.mean(center_roi)
                            std_intensity = np.std(center_roi)
                            # 只有强度比较均匀的区域才认为是棋子
                            if std_intensity < 30:
                                color = 'black' if avg_intensity < 100 else 'white'
                                #print(f'圆心({cx},{cy}) 强度:{avg_intensity:.1f} 判断为:{color}')  # 加这行
                            else:
                                #print(f'圆心({cx},{cy}) 强度不均匀,跳过 std:{std_intensity:.1f}')
                                continue  # 跳过这个圆形
                        else:
                            continue

                        # 添加棋子到列表，使用0-based索引
                        stones.append((color, closest_grid_pos_indices[0], closest_grid_pos_indices[1]))

                        # 绘制检测到的圆形
                        cv2.circle(final_img, (cx, cy), r, (255, 255, 0), 2)
                        cv2.circle(final_img, (cx, cy), 2, (255, 255, 0), -1)

                        # 标注棋子信息
                        text_color = (255, 255, 255) if color == 'black' else (0, 0, 0)
                        cv2.putText(final_img, f"{color[0].upper()}({closest_grid_pos_indices[0]+1},{closest_grid_pos_indices[1]+1})",
                                   (cx-25, cy-r-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

                        # 在对应的网格点上画一个小圆圈表示棋子位置
                        cv2.circle(final_img, closest_grid_point_coords, 8, (0, 255, 0), 2)

            print(f"在围棋网格上识别到 {len(stones)} 个棋子")

            # 创建围棋棋盘状态矩阵
            board_state = np.zeros((19, 19), dtype=int)
            for color, row, col in stones:
                if 0 <= row < 19 and 0 <= col < 19:
                    board_state[row][col] = 1 if color == 'black' else 2

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

            cv2.imshow('Final_Board', final_img)
            cv2.moveWindow('Final_Board', 1600, 0)

        else:
            print("未检测到任何圆形棋子。")
            cv2.imshow('Final_Board', final_img)
            cv2.moveWindow('Final_Board', 1600, 0)

    else: # 如果有效行数不足以构建19x19网格
        print("未能构建完整的19x19围棋网格。")
        cv2.imshow('Final_Board', img_processed) # 使用 img_processed
        cv2.moveWindow('Final_Board', 1600, 0)

else: # 如果交点数量不足
    print("交点数量不足，无法构建围棋网格。")
    cv2.imshow('Final_Board', img_processed) # 使用 img_processed
    cv2.moveWindow('Final_Board', 1600, 0)

# 生成只包含棋盘和棋子的新图像
if 'grid_xy' in locals() and 'stones' in locals() and rect is not None:
    try:
        board_only_img = np.ones((perspective_withpadding_size, perspective_withpadding_size, 3), dtype=np.uint8) * 255

        # 绘制水平棋盘线
        for r in range(19):
            start_point = tuple((grid_xy[r, 0] + padding).astype(int))
            end_point = tuple((grid_xy[r, 18] + padding).astype(int))
            if (0 <= start_point[0] < perspective_withpadding_size and 0 <= start_point[1] < perspective_withpadding_size and
                0 <= end_point[0] < perspective_withpadding_size and 0 <= end_point[1] < perspective_withpadding_size):
                cv2.line(board_only_img, start_point, end_point, (0, 0, 0), 1)

        # 绘制垂直棋盘线
        for c in range(19):
            start_point = tuple((grid_xy[0, c] + padding).astype(int))
            end_point = tuple((grid_xy[18, c] + padding).astype(int))
            if (0 <= start_point[0] < perspective_withpadding_size and 0 <= start_point[1] < perspective_withpadding_size and
                0 <= end_point[0] < perspective_withpadding_size and 0 <= end_point[1] < perspective_withpadding_size):
                cv2.line(board_only_img, start_point, end_point, (0, 0, 0), 1)

        # 绘制检测到的棋子
        for color, row, col in stones:
            if 0 <= row < 19 and 0 <= col < 19:
                center_x, center_y = (grid_xy[row, col] + padding).astype(int)
                if (0 <= center_x < perspective_withpadding_size and 0 <= center_y < perspective_withpadding_size):
                    stone_draw_radius = max(2, int(min_radius * 1.9))
                    stone_color_bgr = (0, 0, 0) if color == 'black' else (255, 255, 255)
                    cv2.circle(board_only_img, (center_x, center_y), stone_draw_radius, stone_color_bgr, -1)
                    if color == 'white':
                        cv2.circle(board_only_img, (center_x, center_y), stone_draw_radius, (0, 0, 0), 1)

        cv2.imshow('NewImg_w_Board_stone', board_only_img)
        cv2.moveWindow('NewImg_w_Board_stone', 1800, 0)
        print("已生成只包含棋盘和棋子的新图像:NewImg_w_Board_stone。")

    except Exception as e:
        print(f"生成棋盘图像时出错: {e}")
        print("未能构建完整的19x19围棋网格，无法生成只包含棋盘和棋子的新图像。")
else:
    print("未能检测到足够的网格行来构建19x19围棋棋盘，或 grid_xy/stones 未被正确初始化。")


cv2.waitKey(0)
cv2.destroyAllWindows()