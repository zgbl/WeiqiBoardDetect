# 增强版围棋棋盘检测 - 添加手动角点选择功能
# 在自动检测失败时允许用户手动点击四个角点
# 这版没写完 抛弃。
import cv2
import numpy as np
from collections import defaultdict
import math

class GoboardDetector:
    def __init__(self):
        self.manual_corners = []
        self.manual_mode = False
        self.corner_names = ['左上角', '左下角', '右上角', '右下角']
        self.current_corner_idx = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于手动选择角点"""
        if event == cv2.EVENT_LBUTTONDOWN and self.manual_mode:
            if len(self.manual_corners) < 4:
                self.manual_corners.append((x, y))
                print(f"已选择{self.corner_names[len(self.manual_corners)-1]}: ({x}, {y})")
                
                # 在图像上标记选择的点
                img_copy = param.copy()
                for i, corner in enumerate(self.manual_corners):
                    cv2.circle(img_copy, corner, 8, (0, 255, 0), -1)
                    cv2.putText(img_copy, str(i+1), (corner[0]-10, corner[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(self.manual_corners) < 4:
                    cv2.putText(img_copy, f"请点击{self.corner_names[len(self.manual_corners)]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(img_copy, "按任意键继续处理", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow('Manual Corner Selection', img_copy)

    def auto_resize_image(self, image, target_width=1500):
        """自动缩放图片到合适的尺寸进行处理"""
        height, width = image.shape[:2]
        print(f"原图尺寸: {width} x {height}")
        
        if width > 2500:
            scale_factor = int((width * 10 / target_width))
            print("scale_factor is:", scale_factor)
            new_width = width * 10 // scale_factor
            new_height = height * 10 // scale_factor
            
            resized_img = cv2.resize(image, (new_width, new_width), interpolation=cv2.INTER_AREA)
            print(f"缩放后尺寸: {new_width} x {new_height} (缩放比例: {scale_factor:.3f})")
            return resized_img, scale_factor
        else:
            print("图片尺寸合适，无需缩放")
            return image, 1.0

    def get_dynamic_params(self, img_width):
        """根据图片大小动态调整参数"""
        if img_width <= 1000:
            print("使用小图参数集")
            return {
                'perspective_size': 400,
                'perspective_corner': 390,
                'hough_threshold': 40,
                'min_line_length': 50,
                'max_line_gap': 8,
                'line_filter_length': 25,
                'merge_tolerance': 8,
                'row_tolerance': 12,
                'min_dist_threshold': 12,
                'min_radius': 4,
                'max_radius': 15,
                'min_dist_circles': 12,
                'min_points_per_row': 8,
                'intersection_threshold': 60
            }
        elif img_width <= 1500:
            print("使用中图参数集")
            return {
                'perspective_size': 500,
                'perspective_corner': 490,
                'hough_threshold': 60,
                'min_line_length': 70,
                'max_line_gap': 10,
                'line_filter_length': 35,
                'merge_tolerance': 12,
                'row_tolerance': 15,
                'min_dist_threshold': 15,
                'min_radius': 6,
                'max_radius': 20,
                'min_dist_circles': 15,
                'min_points_per_row': 12,
                'intersection_threshold': 80
            }
        else:
            print("使用大图参数集")
            return {
                'perspective_size': 660,
                'perspective_corner': 650,
                'hough_threshold': 80,
                'min_line_length': 100,
                'max_line_gap': 10,
                'line_filter_length': 50,
                'merge_tolerance': 15,
                'row_tolerance': 20,
                'min_dist_threshold': 20,
                'min_radius': 8,
                'max_radius': 25,
                'min_dist_circles': 20,
                'min_points_per_row': 15,
                'intersection_threshold': 100
            }

    def improve_edge_detection(self, img):
        """改进的边缘检测，减少图片边界干扰"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # 创建掩码，排除图片边界区域
        mask = np.ones_like(gray) * 255
        border_size = max(10, min(width, height) // 50)
        mask[:border_size, :] = 0
        mask[-border_size:, :] = 0
        mask[:, :border_size] = 0
        mask[:, -border_size:] = 0
        
        # 多种滤波组合
        blurred1 = cv2.GaussianBlur(gray, (3,3), 0)
        blurred2 = cv2.bilateralFilter(gray, 9, 75, 75)  # 保边滤波
        
        # 多种边缘检测方法
        edges1 = cv2.Canny(blurred1, 50, 100)
        edges2 = cv2.Canny(blurred2, 30, 80)
        edges3 = cv2.Canny(gray, 40, 120)
        
        # 组合边缘
        edges_combined = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
        
        # 应用掩码
        edges_final = cv2.bitwise_and(edges_combined, mask)
        
        return edges_final, mask

    def find_board_contour_enhanced(self, contours, img_shape):
        """增强的棋盘轮廓查找"""
        height, width = img_shape[:2]
        img_area = height * width
        
        candidates = []
        
        for item in contours:
            # 多种近似方法
            hull = cv2.convexHull(item)
            
            # 尝试不同的epsilon值
            for epsilon_factor in [0.015, 0.02, 0.025, 0.03]:
                epsilon = epsilon_factor * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    ps = np.reshape(approx, (4,2))
                    area = cv2.contourArea(approx)
                    
                    # 更宽松的面积限制
                    if area < img_area * 0.03 or area > img_area * 0.9:
                        continue
                    
                    # 检查角点位置
                    margin = min(width, height) * 0.015  
                    too_close_to_edge = False
                    
                    for point in ps:
                        x, y = point
                        if (x < margin or x > width - margin or 
                            y < margin or y > height - margin):
                            too_close_to_edge = True
                            break
                    
                    if too_close_to_edge:
                        continue
                    
                    # 长宽比检查
                    x_coords = ps[:, 0]
                    y_coords = ps[:, 1]
                    width_approx = max(x_coords) - min(x_coords)
                    height_approx = max(y_coords) - min(y_coords)
                    aspect_ratio = max(width_approx, height_approx) / min(width_approx, height_approx)
                    
                    if aspect_ratio > 2.5:  # 更宽松
                        continue
                    
                    # 检查四边形的形状质量
                    perimeter = cv2.arcLength(approx, True)
                    rectangularity = (4 * math.pi * area) / (perimeter * perimeter)
                    
                    candidates.append((approx, area, aspect_ratio, rectangularity, epsilon_factor))
        
        if not candidates:
            return None
        
        # 综合评分排序
        def score_candidate(candidate):
            approx, area, aspect_ratio, rectangularity, epsilon_factor = candidate
            # 面积权重40%，长宽比权重30%，矩形度权重30%
            area_score = area / img_area  # 面积越大越好
            aspect_score = 1.0 / (1.0 + aspect_ratio)  # 长宽比越接近1越好
            rect_score = rectangularity  # 矩形度越高越好
            
            total_score = 0.4 * area_score + 0.3 * aspect_score + 0.3 * rect_score
            return total_score
        
        candidates.sort(key=score_candidate, reverse=True)
        best_candidate = candidates[0]
        
        print(f"选择棋盘轮廓: 面积={best_candidate[1]:.0f}, 长宽比={best_candidate[2]:.2f}, 矩形度={best_candidate[3]:.3f}")
        
        # 重新整理角点顺序
        ps = np.reshape(best_candidate[0], (4,2))
        ps = ps[np.lexsort((ps[:,0],))]
        lt, lb = ps[:2][np.lexsort((ps[:2,1],))]
        rt, rb = ps[2:][np.lexsort((ps[2:,1],))]
        
        return (lt, lb, rt, rb)

    def validate_board_corners(self, corners, img):
        """验证检测到的棋盘角点是否合理"""
        if corners is None:
            return False
        
        lt, lb, rt, rb = corners
        
        # 对角线长度检查
        diag1 = np.sqrt((rt[0] - lb[0])**2 + (rt[1] - lb[1])**2)
        diag2 = np.sqrt((lt[0] - rb[0])**2 + (lt[1] - rb[1])**2)
        
        diag_ratio = max(diag1, diag2) / min(diag1, diag2)
        if diag_ratio > 1.8:  # 更宽松
            print(f"对角线长度比例不合理: {diag_ratio:.2f}")
            return False
        
        # 边长检查
        side_lengths = [
            np.sqrt((rt[0] - lt[0])**2 + (rt[1] - lt[1])**2),
            np.sqrt((rb[0] - rt[0])**2 + (rb[1] - rt[1])**2),
            np.sqrt((lb[0] - rb[0])**2 + (lb[1] - rb[1])**2),
            np.sqrt((lt[0] - lb[0])**2 + (lt[1] - lb[1])**2)
        ]
        
        min_side = min(side_lengths)
        max_side = max(side_lengths)
        side_ratio = max_side / min_side
        
        if side_ratio > 2.2:  # 更宽松
            print(f"边长比例不合理: {side_ratio:.2f}")
            return False
        
        print(f"角点验证通过: 对角线比={diag_ratio:.2f}, 边长比={side_ratio:.2f}")
        return True

    def manual_corner_selection(self, img):
        """手动选择棋盘角点"""
        print("\n=== 手动角点选择模式 ===")
        print("请按顺序点击四个角点：")
        print("1. 左上角")
        print("2. 左下角") 
        print("3. 右上角")
        print("4. 右下角")
        print("点击完成后按任意键继续...")
        
        self.manual_corners = []
        self.manual_mode = True
        
        # 创建窗口和鼠标回调
        cv2.namedWindow('Manual Corner Selection')
        cv2.setMouseCallback('Manual Corner Selection', self.mouse_callback, img)
        
        # 显示提示
        img_copy = img.copy()
        cv2.putText(img_copy, "请点击左上角", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Manual Corner Selection', img_copy)
        
        # 等待用户完成选择
        while len(self.manual_corners) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                print("用户取消手动选择")
                cv2.destroyWindow('Manual Corner Selection')
                return None
        
        # 等待用户确认
        cv2.waitKey(0)
        cv2.destroyWindow('Manual Corner Selection')
        
        self.manual_mode = False
        
        # 重新排列角点顺序：左上、左下、右上、右下
        if len(self.manual_corners) == 4:
            corners = self.manual_corners
            print("手动选择的角点：")
            print(f"左上角: {corners[0]}")  
            print(f"左下角: {corners[1]}")
            print(f"右上角: {corners[2]}")
            print(f"右下角: {corners[3]}")
            
            return tuple(corners)
        
        return None

    def detect_board_auto(self, img):
        """自动检测棋盘角点"""
        print("开始自动棋盘检测...")
        
        # 方案1: 改进的边缘检测
        print("尝试方案1: 改进的边缘检测...")
        im_edge_improved, mask = self.improve_edge_detection(img)
        
        contours, hierarchy = cv2.findContours(im_edge_improved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"方案1找到 {len(contours)} 个轮廓")
        
        rect = self.find_board_contour_enhanced(contours, img.shape)
        
        if rect is not None and self.validate_board_corners(rect, img):
            return rect
        
        # 方案2: 原始方法的变体
        print("尝试方案2: 多层边缘检测...")
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 尝试不同的预处理组合
        preprocessing_methods = [
            lambda x: cv2.GaussianBlur(x, (3,3), 0),
            lambda x: cv2.GaussianBlur(x, (5,5), 0),
            lambda x: cv2.bilateralFilter(x, 9, 75, 75),
            lambda x: cv2.medianBlur(x, 5)
        ]
        
        canny_params = [
            (30, 80), (50, 100), (40, 120), (60, 140)
        ]
        
        for prep_method in preprocessing_methods:
            for low, high in canny_params:
                processed = prep_method(im_gray)
                edges = cv2.Canny(processed, low, high)
                
                # 尝试不同的形态学操作
                for kernel_size in [2, 3]:
                    for iterations in [1, 2]:
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        edges_morph = cv2.dilate(edges, kernel, iterations=iterations)
                        
                        contours, _ = cv2.findContours(edges_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if len(contours) > 0:
                            rect = self.find_board_contour_enhanced(contours, img.shape)
                            if rect is not None and self.validate_board_corners(rect, img):
                                print(f"方案2成功: 预处理={prep_method.__name__}, Canny=({low},{high}), 核={kernel_size}, 迭代={iterations}")
                                return rect
        
        print("自动检测失败")
        return None

    def detect_board(self, img_path):
        """主检测函数"""
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return None
        
        # 缩放图片
        img, scale_factor = self.auto_resize_image(img)
        original_img = img.copy()
        
        # 获取动态参数
        params = self.get_dynamic_params(img.shape[1])
        
        cv2.imshow("Original Image", img)
        cv2.moveWindow("Original Image", 0, 0)
        
        # 尝试自动检测
        rect = self.detect_board_auto(img)
        
        # 如果自动检测失败，提供手动选择选项
        if rect is None:
            print("\n自动检测失败!")
            choice = input("是否要手动选择角点? (y/n): ").lower().strip()
            
            if choice == 'y' or choice == 'yes':
                rect = self.manual_corner_selection(img)
            else:
                print("跳过角点检测，使用原图进行后续处理")
                return original_img, None, params
        
        if rect is not None:
            print('棋盘角点坐标：')
            corner_names = ['左上角', '左下角', '右上角', '右下角']
            for i, corner in enumerate(rect):
                print(f'\t{corner_names[i]}:({corner[0]},{corner[1]})')
            
            # 显示找到的角点
            corner_img = img.copy()
            colors = [(0,255,0), (0,255,255), (255,0,0), (255,0,255)]
            for i, p in enumerate(rect):
                cv2.circle(corner_img, (p[0],p[1]), 15, colors[i], -1)
                cv2.putText(corner_img, str(i+1), (p[0]-5, p[1]+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            cv2.imshow('Found Corners', corner_img)
            cv2.moveWindow("Found Corners", 400, 0)
            
            # 执行透视变换
            lt, lb, rt, rb = rect
            pts1 = np.float32([(10,10), (10,params['perspective_corner']), 
                              (params['perspective_corner'],10), (params['perspective_corner'],params['perspective_corner'])])
            pts2 = np.float32([lt, lb, rt, rb])
            m = cv2.getPerspectiveTransform(pts2, pts1)
            
            corrected_img = cv2.warpPerspective(original_img, m, (params['perspective_size'], params['perspective_size']))
            
            cv2.imshow('Perspective Corrected', corrected_img)
            cv2.moveWindow("Perspective Corrected", 800, 0)
            print("透视变换完成！")
            
            return corrected_img, rect, params
        
        return original_img, None, params

# 使用示例
if __name__ == "__main__":
    detector = GoboardDetector()
    
    # 测试图片路径列表
    test_images = [
        '../data/raw/bd317d54.webp',
        '../data/raw/IMG20171015161921.jpg',
        '../data/raw/OGS3.jpeg',
        '../data/raw/IMG20160706171004.jpg',
        '../data/raw/IMG20160904165505-B.jpg',
        '../data/raw/IMG20160706171004-12.jpg'
    ]
    
    # 选择要测试的图片
    img_path = '../data/raw/IMG20160904165505-B.jpg'  # 修改为你要测试的图片路径
    
    result_img, corners, params = detector.detect_board(img_path)
    
    if result_img is not None:
        print("检测完成！")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("检测失败！")

    # 没写完