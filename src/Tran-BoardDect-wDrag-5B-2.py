# V5-2 版本，
import cv2
import numpy as np
from collections import defaultdict
import math

class InteractiveCornerAdjuster:
    def __init__(self, image, initial_corners):
        """
        初始化交互式角点调整器
        
        Args:
            image: 原始图像
            initial_corners: 初始角点 (lt, lb, rt, rb)
        """
        self.original_image = image.copy()
        self.display_image = image.copy()
        self.corners = list(initial_corners)  # [lt, lb, rt, rb]
        self.corner_labels = ['左上', '左下', '右上', '右下']
        self.corner_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255)]
        
        # 交互参数
        self.dragging = False
        self.selected_corner = -1
        self.corner_radius = 15
        self.hover_radius = 20
        self.confirmed = False
        
        # 窗口名称
        self.window_name = "角点调整 - 拖动角点后按空格确认，ESC取消"
        
        # 创建窗口和设置鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击了某个角点
            for i, corner in enumerate(self.corners):
                dist = math.sqrt((x - corner[0])**2 + (y - corner[1])**2)
                if dist < self.corner_radius:
                    self.dragging = True
                    self.selected_corner = i
                    break
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_corner != -1:
                # 拖动角点
                self.corners[self.selected_corner] = [x, y]
                self.update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.selected_corner = -1
    
    def update_display(self):
        """更新显示图像"""
        self.display_image = self.original_image.copy()
        
        # 绘制四边形边界
        if len(self.corners) == 4:
            # 连接角点形成四边形
            pts = np.array([self.corners[0], self.corners[2], self.corners[3], self.corners[1]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(self.display_image, [pts], True, (0, 0, 255), 2)
        
        # 绘制角点
        for i, (corner, label, color) in enumerate(zip(self.corners, self.corner_labels, self.corner_colors)):
            # 绘制角点圆圈
            cv2.circle(self.display_image, tuple(corner), self.corner_radius, color, -1)
            cv2.circle(self.display_image, tuple(corner), self.corner_radius, (0, 0, 0), 2)
            
            # 绘制标签
            cv2.putText(self.display_image, label, 
                       (corner[0] - 20, corner[1] - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 绘制坐标
            cv2.putText(self.display_image, f"({corner[0]},{corner[1]})", 
                       (corner[0] - 30, corner[1] + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 绘制操作提示
        cv2.putText(self.display_image, "拖动角点调整位置", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(self.display_image, "空格键确认 | ESC取消", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def validate_corners(self):
        """验证角点是否合理"""
        if len(self.corners) != 4:
            return False, "角点数量不正确"
        
        # 检查角点是否在图像范围内
        height, width = self.original_image.shape[:2]
        for i, corner in enumerate(self.corners):
            if corner[0] < 0 or corner[0] >= width or corner[1] < 0 or corner[1] >= height:
                return False, f"角点 {self.corner_labels[i]} 超出图像范围"
        
        # 检查四边形是否合理（简单检查面积）
        pts = np.array([self.corners[0], self.corners[2], self.corners[3], self.corners[1]], np.int32)
        area = cv2.contourArea(pts)
        img_area = width * height
        
        if area < img_area * 0.01:
            return False, "选择区域太小"
        
        if area > img_area * 0.95:
            return False, "选择区域太大"
        
        return True, "角点验证通过"
    
    def run(self):
        """运行交互式调整"""
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
                    # 可以选择继续调整或者退出
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

# 使用示例：将此功能集成到你的主代码中
def integrate_interactive_adjustment():
    """
    展示如何将交互式调整集成到现有代码中
    """
    # 这是你现有代码中找到角点后的部分
    # 假设你已经有了 img 和 rect
    
    # 原有代码...
    # rect = find_board_contour_improved(contours, img.shape)
    
    # 如果找到了角点，进行交互式调整
    if rect is not None:
        print('自动检测到的棋盘角点：')
        print(f'\t左上角:({rect[0][0]},{rect[0][1]})')
        print(f'\t左下角:({rect[1][0]},{rect[1][1]})')
        print(f'\t右上角:({rect[2][0]},{rect[2][1]})')
        print(f'\t右下角:({rect[3][0]},{rect[3][1]})')
        
        # 启动交互式调整
        confirmed, adjusted_corners = interactive_corner_adjustment(img, rect)
        
        if confirmed:
            print("用户确认了调整后的角点")
            rect = adjusted_corners  # 使用调整后的角点
            print('最终角点：')
            print(f'\t左上角:({rect[0][0]},{rect[0][1]})')
            print(f'\t左下角:({rect[1][0]},{rect[1][1]})')
            print(f'\t右上角:({rect[2][0]},{rect[2][1]})')
            print(f'\t右下角:({rect[3][0]},{rect[3][1]})')
        else:
            print("用户取消了调整，使用原始检测结果")
        
        # 继续后续的透视变换等处理...
    else:
        print('未能找到棋盘角点')

# 读取图像
#img = cv2.imread('../data/raw/bd317d54.webp')
#img = cv2.imread('../data/raw/IMG20171015161921.jpg')
#img = cv2.imread('../data/raw/OGS3.jpeg')
#img = cv2.imread('../data/raw/IMG20160706171004.jpg')
#img = cv2.imread('../data/raw/IMG20161205130156-16.jpg') # 这个图片现在不能检测
#img = cv2.imread('../data/raw/IMG20160904165505-B.jpg')
#img = cv2.imread('../data/raw/IMG20160706171004-12.jpg')
#img = cv2.imread('../data/raw/WechatIMG123.jpg')
#img = cv2.imread('../data/raw/WechatIMG124.jpg')
#img = cv2.imread('../data/raw/ogs_empty2c.jpg')
img = cv2.imread('../data/raw/WechatIMG123.jpg')


if img is None:
    print("没找到照片")
    exit()

# ====== 自动缩放图片功能 ======
def auto_resize_image(image, target_width):
    """
    自动缩放图片到合适的尺寸进行处理
    target_width: 目标宽度, 默认1000像素
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
print(f"DEBUG: After auto_resize_image, img shape: {img.shape}") # DEBUG 1

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
    perspective_size = 510
    perspective_corner = 490
    hough_threshold = 60
    min_line_length = 70
    max_line_gap = 10
    line_filter_length = 35
    #merge_tolerance = 12
    merge_tolerance = 20
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

# ====== 改进的边缘检测函数 ======
def improve_edge_detection(img):
    """
    改进的边缘检测，减少图片边界干扰
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("用到了138行的 gray = ")
    height, width = gray.shape
    
    # 创建掩码，排除图片边界区域
    mask = np.ones_like(gray) * 255
    border_size = max(10, min(width, height) // 50)  # 动态边界大小
    mask[:border_size, :] = 0
    mask[-border_size:, :] = 0
    mask[:, :border_size] = 0
    mask[:, -border_size:] = 0
    
    # 高斯滤波
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 100)
    #edges = cv2.Canny(gray, 50, 100)  #实验性地跳过高斯模糊
    
    # 应用掩码，去除边界区域
    edges = cv2.bitwise_and(edges, mask)
    
    return edges, mask

# ====== 改进的轮廓查找函数 ======
def find_board_contour_improved(contours, img_shape):
    """
    改进的棋盘轮廓查找，避免选择整个图片边界
    """
    height, width = img_shape[:2]
    img_area = height * width
    
    candidates = []
    
    for item in contours:
        hull = cv2.convexHull(item)
        epsilon = 0.02 * cv2.arcLength(hull, True)  # 减小epsilon，更精确
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            ps = np.reshape(approx, (4,2))
            area = cv2.contourArea(approx)
            
            # 排除过大的轮廓（可能是整个图片边界）
            if area > img_area * 0.8:
                print(f"排除过大轮廓，面积: {area:.0f} (图片面积的 {area/img_area:.1%})")
                continue
            
            # 排除过小的轮廓
            if area < img_area * 0.05:
                print(f"排除过小轮廓，面积: {area:.0f}")
                continue
            
            # 检查四个角点是否太接近图片边界
            margin = min(width, height) * 0.02  # 2%的边界余量
            too_close_to_edge = False
            
            for point in ps:
                x, y = point
                if (x < margin or x > width - margin or 
                    y < margin or y > height - margin):
                    too_close_to_edge = True
                    break
            
            if too_close_to_edge:
                print(f"排除太接近边界的轮廓")
                continue
            
            # 检查四边形的长宽比是否合理（棋盘应该接近正方形）
            x_coords = ps[:, 0]
            y_coords = ps[:, 1]
            width_approx = max(x_coords) - min(x_coords)
            height_approx = max(y_coords) - min(y_coords)
            aspect_ratio = max(width_approx, height_approx) / min(width_approx, height_approx)
            
            if aspect_ratio > 2.0:  # 长宽比不能太极端
                print(f"排除长宽比过大的轮廓: {aspect_ratio:.2f}")
                continue
            
            candidates.append((approx, area, aspect_ratio))
    
    if not candidates:
        return None
    
    # 选择面积最大但长宽比合理的候选
    candidates.sort(key=lambda x: (-x[1], x[2]))  # 按面积降序，长宽比升序
    best_candidate = candidates[0]
    
    print(f"选择棋盘轮廓: 面积={best_candidate[1]:.0f}, 长宽比={best_candidate[2]:.2f}")
    
    # 重新整理角点顺序
    ps = np.reshape(best_candidate[0], (4,2))
    ps = ps[np.lexsort((ps[:,0],))]
    lt, lb = ps[:2][np.lexsort((ps[:2,1],))]
    rt, rb = ps[2:][np.lexsort((ps[2:,1],))]
    
    return (lt, lb, rt, rb)

# ====== 角点验证函数 ======
def validate_board_corners(corners, img):
    """
    验证检测到的棋盘角点是否合理
    """
    if corners is None:
        return False
    
    lt, lb, rt, rb = corners
    
    # 检查角点是否形成合理的四边形
    # 计算对角线长度
    diag1 = np.sqrt((rt[0] - lb[0])**2 + (rt[1] - lb[1])**2)
    diag2 = np.sqrt((lt[0] - rb[0])**2 + (lt[1] - rb[1])**2)
    
    # 对角线长度应该相近
    diag_ratio = max(diag1, diag2) / min(diag1, diag2)
    if diag_ratio > 1.5:
        print(f"对角线长度比例不合理: {diag_ratio:.2f}")
        return False
    
    # 检查四条边的长度
    side_lengths = [
        np.sqrt((rt[0] - lt[0])**2 + (rt[1] - lt[1])**2),  # 上边
        np.sqrt((rb[0] - rt[0])**2 + (rb[1] - rb[1])**2),  # 右边
        np.sqrt((lb[0] - rb[0])**2 + (lb[1] - rb[1])**2),  # 下边
        np.sqrt((lt[0] - lb[0])**2 + (lt[1] - lb[1])**2)   # 左边
    ]
    
    # 四条边长度应该相对均匀
    min_side = min(side_lengths)
    max_side = max(side_lengths)
    side_ratio = max_side / min_side
    
    if side_ratio > 1.8:
        print(f"边长比例不合理: {side_ratio:.2f}")
        return False
    
    print(f"角点验证通过: 对角线比={diag_ratio:.2f}, 边长比={side_ratio:.2f}")
    return True

# ====== 开始处理 ======
original_img = img.copy()
cv2.imshow("original picture", img)

print("开始改进的透视变换...")

# ====== 透视变换部分 ======
im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("im_gray", im_gray)

# 方案1: 使用改进的边缘检测（不使用dilate）
print("尝试方案1: 改进的边缘检测...")
im_edge_improved, mask = improve_edge_detection(img)
cv2.imshow('Improved Edge Detection', im_edge_improved)
cv2.moveWindow("Improved Edge Detection", 300, 0)

# 查找轮廓
contours, hierarchy = cv2.findContours(im_edge_improved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"方案1找到 {len(contours)} 个轮廓")

# 使用改进的轮廓查找 
rect = find_board_contour_improved(contours, img.shape)

# 验证角点
if rect is not None and not validate_board_corners(rect, img):
    print("方案1角点验证失败")
    rect = None

# 方案2: 如果方案1失败，尝试原始方法
if rect is None:
    print("尝试方案2: 原始Canny + 轻微dilate...")
    im_gray_blur = cv2.GaussianBlur(im_gray, (3,3), 0)
    im_edge_original = cv2.Canny(im_gray_blur, 50, 100)
    
    # 轻微dilate（比原来更保守）
    kernel = np.ones((2, 2), np.uint8)
    im_edge_dilated = cv2.dilate(im_edge_original, kernel, iterations=1)
    
    #cv2.imshow('Original + Light Dilate', im_edge_dilated)
    #cv2.moveWindow("Original + Light Dilate", 600, 0)
    
    # 再次查找轮廓
    contours2, hierarchy2 = cv2.findContours(im_edge_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"方案2找到 {len(contours2)} 个轮廓")
    
    rect = find_board_contour_improved(contours2, img.shape)
    
    if rect is not None and not validate_board_corners(rect, img):
        print("方案2角点验证失败")
        rect = None

# 方案3: 如果还是失败，尝试更宽松的条件
if rect is None:
    print("尝试方案3: 宽松条件的原始方法...")
    contours3, hierarchy3 = cv2.findContours(im_edge_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 使用更宽松的原始方法
    area = 0
    for item in contours3:
        hull = cv2.convexHull(item)
        epsilon = 0.1 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            ps = np.reshape(approx, (4,2))
            a = cv2.contourArea(approx)
            
            # 只排除明显的整图边界（面积>95%）
            img_area = img.shape[0] * img.shape[1]
            if a > img_area * 0.95:
                continue
                
            if a > area:
                area = a
                ps = ps[np.lexsort((ps[:,0],))]
                lt, lb = ps[:2][np.lexsort((ps[:2,1],))]
                rt, rb = ps[2:][np.lexsort((ps[2:,1],))]
                rect = (lt, lb, rt, rb)

# 检查是否找到棋盘
if rect is None:
    print('未能找到棋盘！将使用原图进行后续处理，但棋盘线和棋子检测可能不准确。')
    # If no board contour is found, revert to original image for display, but further processing won't be reliable
    img = original_img

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print("用到了359行的 gray = ")
    
    # Set dummy values for perspective_size and perspective_corner for downstream use
    # These will likely result in incorrect grid calculations if no rect was found
    perspective_size = min(img.shape[0], img.shape[1]) 
    perspective_corner = perspective_size - 10 # Just a placeholder
    
else:
    print('棋盘坐标：')
    print('\t左上角:(%d,%d)'%(rect[0][0],rect[0][1]))
    print('\t左下角:(%d,%d)'%(rect[1][0],rect[1][1]))
    print('\t右上角:(%d,%d)'%(rect[2][0],rect[2][1]))
    print('\t右下角:(%d,%d)'%(rect[3][0],rect[3][1]))

    # Display found corners
    corner_img = img.copy()  # 注意这里 img 还是原始尺寸的图像
    for i, p in enumerate(rect):
        cv2.circle(corner_img,(p[0],p[1]),15,(0,255,0),-1)
    cv2.imshow('Auto Detected Corners', corner_img)
    cv2.moveWindow("Auto Detected Corners", 100, 100)
    #for p in rect:
    #    cv2.circle(corner_img,(p[0],p[1]),15,(0,255,0),-1)

    # === 添加交互式调整 ===
    print("开始交互式角点调整...")
    confirmed, adjusted_corners = interactive_corner_adjustment(img, rect)

    if confirmed:
        rect = adjusted_corners
        print("使用调整后的角点")
        print('最终角点：')
        print(f'\t左上角:({rect[0][0]},{rect[0][1]})')
        print(f'\t左下角:({rect[1][0]},{rect[1][1]})')
        print(f'\t右上角:({rect[2][0]},{rect[2][1]})')
        print(f'\t右下角:({rect[3][0]},{rect[3][1]})')
    else:
        print("使用原始检测的角点")

    cv2.imshow('Found Corners', corner_img)
    cv2.moveWindow("Found Corners", 900, 0)

    # === 新增调试代码：在原始图像上绘制检测到的棋盘边界 ===  7/13 调试代码
    debug_original_with_rect = original_img.copy()
    lt, lb, rt, rb = rect # 确保 rect 已经被解包

        # 将角点连接起来，绘制一个矩形或多边形
    # 注意：rect 中的点顺序是 (lt, lb, rt, rb)，需要调整为 (tl, tr, br, bl) 才能正确绘制多边形
    pts_for_poly = np.array([lt, rt, rb, lb], np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug_original_with_rect, [pts_for_poly], True, (0, 0, 255), 3) # 红色粗线
    
    # 也可以再次绘制角点，确保它们位置正确
    cv2.circle(debug_original_with_rect, tuple(lt), 15, (0, 255, 255), -1) # 黄色圆点
    cv2.circle(debug_original_with_rect, tuple(rt), 15, (0, 255, 255), -1)
    cv2.circle(debug_original_with_rect, tuple(rb), 15, (0, 255, 255), -1)
    cv2.circle(debug_original_with_rect, tuple(lb), 15, (0, 255, 255), -1)
    
    cv2.imshow('Debug: Rect on Original Image', debug_original_with_rect)
    cv2.moveWindow('Debug: Rect on Original Image', 100, 100) # 调整位置，避免重叠
    #到这行之上 都是7/13 临时增加的调试代码，

    # Perform perspective transform
    lt, lb, rt, rb = rect

    padding = 10  # 空白边距像素，可调整
    perspective_withpadding_size  = perspective_size + 2 * padding

    # 添加额外的扩展像素
    extra_pixels = 3  # 可以调整为1-3个像素
    
    # === 关键修正: 修正透视变换的目标坐标 `pts1` ===
    # 目标点应覆盖整个 perspective_size x perspective_size 区域
    #pts1_standard = np.float32([[0, 0], [perspective_size - 1, 0], [perspective_size - 1, perspective_size - 1], [0, perspective_size - 1]])
    # 方案1：让棋盘占据除了padding之外的整个区域
    #pts1_standard = np.float32([[padding, padding], 
    #                            [perspective_size + padding + extra_pixels, padding], 
    #                            [perspective_size + padding + extra_pixels, perspective_size + padding + extra_pixels], 
    #                            [padding, perspective_size + padding + extra_pixels]])
    pts1_standard = np.float32([
    [padding - extra_pixels, padding - extra_pixels],
    [padding + perspective_size + extra_pixels, padding - extra_pixels],
    [padding + perspective_size + extra_pixels, padding + perspective_size + extra_pixels],
    [padding - extra_pixels, padding + perspective_size + extra_pixels]
])
    #pts1_standard = np.float32([[0, 0], 
    #                    [perspective_size, 0], 
    #                    [perspective_size, perspective_size], 
    #                    [0, perspective_size]])


    # Reorder the detected corners for cv2.getPerspectiveTransform
    # (top-left, top-right, bottom-right, bottom-left)
    #pts2_ordered = np.float32([lt, rt, rb, lb]) 
    
    # 调整源点：让检测到的角点向外扩展
    pts2_ordered = np.float32([
    [lt[0] - extra_pixels, lt[1] - extra_pixels],  # 左上角向左上扩展
    [rt[0] + extra_pixels, rt[1] - extra_pixels],  # 右上角向右上扩展  
    [rb[0] + extra_pixels, rb[1] + extra_pixels],  # 右下角向右下扩展
    [lb[0] - extra_pixels, lb[1] + extra_pixels]   # 左下角向左下扩展
])
    
    # Using pts1_standard and pts2_ordered
    m = cv2.getPerspectiveTransform(pts2_ordered, pts1_standard)
    
    # 添加一个平移矩阵，使图像整体向右下偏移 padding 像素
    #M_translate = np.array([[1, 0, padding],
    #                        [0, 1, padding],
    #                        [0, 0, 1]], dtype=np.float32)

    # 新的透视变换矩阵：先透视变换，再加平移
    #m_padded = M_translate @ m

    print(f"perspective_size: {perspective_size}", "Line 429")
    print(f"padding: {padding}")
    print(f"perspective_withpadding_size: {perspective_withpadding_size}")
    print(f"pts1_standard: {pts1_standard}")
    print(f"pts2_ordered: {pts2_ordered}", "Line 433")
    # Apply perspective transform to original image and its grayscale version
    #img = cv2.warpPerspective(original_img, m, (perspective_size, perspective_size))
    #gray = cv2.warpPerspective(im_gray, m, (perspective_size, perspective_size))
    #img = cv2.warpPerspective(original_img, m_padded, (perspective_withpadding_size, perspective_withpadding_size))
    img = cv2.warpPerspective(original_img, m, (perspective_withpadding_size, perspective_withpadding_size))
    #gray = cv2.warpPerspective(im_gray, m_padded, (perspective_withpadding_size, perspective_withpadding_size))
    gray2 = cv2.warpPerspective(im_gray, m, (perspective_withpadding_size, perspective_withpadding_size))
    print("用到了445行的 gray = ")
    #img = cv2.warpPerspective(original_img, m, (perspective_size, perspective_size))
    #gray = cv2.warpPerspective(im_gray, m, (perspective_size, perspective_size))  

    print(f"DEBUG: After warpPerspective, img shape: {img.shape}") # DEBUG 2
    print(f"DEBUG: After warpPerspective, gray shape: {gray2.shape}") # DEBUG 3

    cv2.imshow('Perspective Corrected', img)
    cv2.moveWindow("Perspective Corrected", 1200, 0)
    print("透视变换完成！")

# ====== 棋盘线和交点计算部分 ======
print("开始棋盘线检测...")

cv2.imshow("gray2", gray2)

# 高斯模糊
blur = cv2.GaussianBlur(gray2, (5, 5), 0)
print(f"DEBUG: Before Canny, blur shape: {blur.shape}") # DEBUG 4
cv2.imshow("blur", blur)

# 边缘检测
edges = cv2.Canny(blur, 50, 100)
#edges = cv2.Canny(blur, 50, 150)
#edges = im_edge_improved.copy() # 或者直接使用 im_edge_improved
print(f"DEBUG: Before HoughLinesP, edges shape: {edges.shape}") # DEBUG 5
cv2.imshow('edges', edges)
# 如果您仍然想在高斯模糊后应用Canny，确保是针对im_edge_improved进行，或者直接使用im_edge_improved
# 比如：
# blur_improved_edges = cv2.GaussianBlur(im_edge_improved, (3, 3), 0) # 可以选择性地再次模糊
# edges = cv2.Canny(blur_improved_edges, 50, 150) # 如果需要再次Canny，参数可能需要调整
cv2.imshow('edges for Hough', edges) # 打开这行，确认霍夫变换使用的图是正确的
cv2.moveWindow('edges for Hough', 600, 0) # 移动一下窗口，方便查看

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

print(f"水平线: {len(horizontal_lines)} 条, 470行")
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

print("合并后的水平线坐标:")
for line in merged_horizontal:
    y_avg = (line[1] + line[3]) // 2
    print(f"y={y_avg}, line={line}")

print("合并后的垂直线坐标:")
for line in merged_vertical:
    x_avg = (line[0] + line[2]) // 2
    print(f"x={x_avg}, line={line}")

print(f"合并后水平线: {len(merged_horizontal)} 条")
print(f"合并后垂直线: {len(merged_vertical)} 条")

# 绘制合并后的线条
merged_img = img.copy()
for line in merged_horizontal:
    cv2.line(merged_img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
for line in merged_vertical:
    cv2.line(merged_img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)

#cv2.imshow('merged_lines', merged_img)

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
# 定义一个小的像素缓冲区，以容纳可能略微超出边界的交点
# 例如，5到10个像素通常足够
intersection_buffer = 10 

for h_line in merged_horizontal:
    for v_line in merged_vertical:
        intersection = line_intersection(h_line, v_line)
        if intersection:
            # 放宽交点过滤的边界，允许在 perspective_size 范围外有少量像素
            if (intersection[0] >= -intersection_buffer and 
                intersection[0] < perspective_size + intersection_buffer and
                intersection[1] >= -intersection_buffer and 
                intersection[1] < perspective_size + intersection_buffer):
                intersections.append(intersection)

print(f"找到 {len(intersections)} 个交点. 代码550行")

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
        grid_img_with_go_points = img.copy() # This will be the image to show calculated grid points
        
        # Initialize grid_xy here. Crucially, its dimensions should align with perspective_size
        grid_xy = np.zeros((19, 19, 2), dtype=np.int32)

        # === 核心修正：基于 perspective_size 重新计算棋盘网格点，而不是依赖于不精确的检测点 ===
        # 计算每个格子的预期尺寸
        # 如果棋盘是 19x19，则有 18 个格子间隔
        cell_interval = (perspective_size - 1) / 18.0 
        
        for row_idx in range(19):
            for col_idx in range(19):
                # 计算理想的网格点坐标
                x_ideal = int(col_idx * cell_interval)
                y_ideal = int(row_idx * cell_interval)
                
                # 如果有足够精确的检测点，可以尝试微调这些理想点到最近的检测点
                # 但为了确保棋盘线不越界，此处优先使用理想的、基于 perspective_size 的点
                # 如果需要微调，可以在确保不越界的前提下进行
                
                go_grid_points.append((x_ideal, y_ideal, row_idx, col_idx))
                grid_xy[row_idx, col_idx] = [x_ideal, y_ideal] # Fill grid_xy
                
                # 绘制围棋网格点（在最终图像上）
                cv2.circle(grid_img_with_go_points, (x_ideal + padding, y_ideal + padding), 2, (0, 255, 255), -1)  # 黄色小点
                
                # 在关键点上标注坐标
                if row_idx % 3 == 0 and col_idx % 3 == 0:  # 每3个点标注一次
                    cv2.putText(grid_img_with_go_points, f"({row_idx},{col_idx})", 
                               (x_ideal-15, y_ideal-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.3, (0, 255, 255), 1)
        
        print(f"构建了 {len(go_grid_points)} 个围棋网格点 (目标: 361)")
        
        # Display calculated grid points (on the temporary extended canvas)
        cv2.imshow('Calculated Grid Points', grid_img_with_go_points)
        cv2.moveWindow('Calculated Grid Points', 1500, 0) # Moved to 1500 to avoid overlap
        print("已显示计算出的网格点。")

        # Use the standard perspective corrected image for final_img
        final_img = img.copy() 
        
        # Draw the calculated grid lines on final_img
        for r in range(19):
            cv2.line(final_img, tuple(grid_xy[r, 0].astype(int)), tuple(grid_xy[r, 18].astype(int)), (0, 0, 0), 1) # Black lines
        for c in range(19):
            cv2.line(final_img, tuple(grid_xy[0, c].astype(int)), tuple(grid_xy[18, c].astype(int)), (0, 0, 0), 1) # Black lines
        
        # Draw calculated grid points (yellow small dots) on final_img
        for point_data in go_grid_points:
            cv2.circle(final_img, (point_data[0], point_data[1]), 2, (0, 255, 255), -1)

        # 重新进行圆检测，使用原始的透视矫正图像
        circles = cv2.HoughCircles(gray2, method=cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=min_dist_circles, param1=100, param2=19,
                                   minRadius=min_radius, maxRadius=max_radius)
        
        stones = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            print(f'检测到 {len(circles[0])} 个圆形')
            
            for circle in circles[0, :]:
                cx, cy, r = circle
                
                # 找到最近的围棋网格点
                min_dist = float('inf')
                closest_grid_point_coords = None
                closest_grid_pos_indices = None # (row, col)
                
                # === 在这里，我们使用 grid_xy 查找最近的网格点，而不是 go_grid_points ===
                for row_idx in range(19):
                    for col_idx in range(19):
                        x_grid, y_grid = grid_xy[row_idx, col_idx]
                        dist = math.sqrt((cx - x_grid)**2 + (cy - y_grid)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_grid_point_coords = (x_grid, y_grid)
                            closest_grid_pos_indices = (row_idx, col_idx)
                
                # 如果圆心离最近围棋网格点足够近，认为这是一个有效的棋子
                if closest_grid_point_coords is not None and min_dist < min_dist_threshold:
                    # 颜色检测
                    roi = gray2[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
                    if roi.size > 0:
                        avg_intensity = np.mean(roi)
                        color = 'black' if avg_intensity < 120 else 'white'
                        
                        # Add stone to list, using 1-based indexing for display (Go board standard)
                        stones.append((color, closest_grid_pos_indices[0], closest_grid_pos_indices[1]))
                        
                        # 绘制检测到的圆形
                        cv2.circle(final_img, (cx, cy), r, (255, 255, 0), 2) # Cyan boundary
                        cv2.circle(final_img, (cx, cy), 2, (255, 255, 0), -1) # Cyan center dot
                        
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
            
            # 显示最终结果
            cv2.imshow('final_go_board', final_img)

        else:
            print("未检测到任何圆形棋子。")

    else: # If valid_rows < 18 or other issues with grid construction
        print("未能构建完整的19x19围棋网格。")
        cv2.imshow('final_go_board', img) # Show the perspective corrected image without stones if grid is bad

else: # If intersections count is too low
    print("交点数量不足，无法构建围棋网格。")
    cv2.imshow('final_go_board', img) # Show the perspective corrected image without stones

# 生成只包含棋盘和棋子的新图像
# Ensure grid_xy and stones are defined before this block
if 'grid_xy' in locals() and 'stones' in locals():
    try:
        # Create a pure white background new image, same size as perspective_size
        #board_only_img = np.ones((perspective_size, perspective_size, 3), dtype=np.uint8) * 255 
        board_only_img = np.ones((perspective_withpadding_size, perspective_withpadding_size, 3), dtype=np.uint8) * 255 
        
        # 绘制水平棋盘线
        for r in range(19):
            start_point = tuple((grid_xy[r, 0] + padding).astype(int))
            end_point = tuple((grid_xy[r, 18] + padding).astype(int))
            
            # Check if coordinates are within image bounds
            if (0 <= start_point[0] < perspective_withpadding_size and 0 <= start_point[1] < perspective_withpadding_size and
                0 <= end_point[0] < perspective_withpadding_size and 0 <= end_point[1] < perspective_withpadding_size):
                cv2.line(board_only_img, start_point, end_point, (0, 0, 0), 1)

        # 绘制垂直棋盘线
        for c in range(19):
            start_point = tuple((grid_xy[0, c] + padding).astype(int))
            end_point = tuple((grid_xy[18, c] + padding).astype(int))
            
            # Check if coordinates are within image bounds
            if (0 <= start_point[0] < perspective_withpadding_size and 0 <= start_point[1] < perspective_withpadding_size and
                0 <= end_point[0] < perspective_withpadding_size and 0 <= end_point[1] < perspective_withpadding_size):
                cv2.line(board_only_img, start_point, end_point, (0, 0, 0), 1)

        # 绘制检测到的棋子
        for color, row, col in stones:
            if 0 <= row < 19 and 0 <= col < 19:
                center_x, center_y = (grid_xy[row, col] + padding).astype(int)
                
                # Check if coordinates are within image bounds
                if (0 <= center_x < perspective_withpadding_size and 0 <= center_y < perspective_withpadding_size):
                    # Use a fixed radius to draw stones for visual consistency on the new board
                    stone_draw_radius = max(2, int(min_radius * 1.9)) 
                    
                    # Set drawing color based on stone color
                    stone_color_bgr = (0, 0, 0) if color == 'black' else (255, 255, 255) 
                    cv2.circle(board_only_img, (center_x, center_y), stone_draw_radius, stone_color_bgr, -1)
                    
                    # Add black border for white stones to make them more visible on white background
                    if color == 'white':
                        cv2.circle(board_only_img, (center_x, center_y), stone_draw_radius, (0, 0, 0), 1) 
        
        cv2.imshow('NewBoard with Stone', board_only_img)
        cv2.moveWindow('NewBoard with Stone', 1800, 0) # Moved to 1800 to avoid overlap
        print("已生成只包含棋盘和棋子的新图像。")
        
    except Exception as e:
        print(f"生成棋盘图像时出错: {e}")
        print("未能构建完整的19x19围棋网格，无法生成只包含棋盘和棋子的新图像。")
        
else:
    print("未能检测到足够的网格行来构建19x19围棋棋盘，或 grid_xy/stones 未被正确初始化。")


cv2.waitKey(0)
cv2.destroyAllWindows()