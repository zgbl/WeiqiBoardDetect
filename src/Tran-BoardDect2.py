# 增强版围棋棋盘检测系统 - 提升鲁棒性
# 这一版完全不行。原来好的部分也坏了。
import cv2
import numpy as np
from collections import defaultdict
import math

class RobustGoBoardDetector:
    def __init__(self):
        # 动态参数配置
        self.target_size = 800  # 目标处理尺寸
        self.min_board_area_ratio = 0.1  # 棋盘最小面积占图片比例
        self.max_board_area_ratio = 0.9  # 棋盘最大面积占图片比例
        
    def preprocess_image(self, img):
        """智能图像预处理 - 解决高分辨率问题"""
        original_img = img.copy()
        h, w = img.shape[:2]
        
        print(f"原始图像尺寸: {w}x{h}")
        
        # 如果图像太大，先缩放到合适尺寸进行处理
        scale_factor = 1.0
        if max(w, h) > self.target_size:
            scale_factor = self.target_size / max(w, h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"缩放到: {new_w}x{new_h}, 缩放比例: {scale_factor:.3f}")
        
        return img, original_img, scale_factor
    
    def enhanced_board_detection(self, img):
        """增强的棋盘检测算法"""
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 多尺度边缘检测
        results = []
        
        # 尝试不同的预处理组合
        preprocessing_methods = [
            # (高斯模糊, Canny阈值1, Canny阈值2, 描述)
            ((3, 3), 30, 80, "标准参数"),
            ((5, 5), 20, 60, "更敏感"),
            ((3, 3), 50, 120, "更严格"),
            ((7, 7), 25, 75, "更多模糊"),
        ]
        
        for blur_kernel, canny1, canny2, desc in preprocessing_methods:
            print(f"尝试 {desc}: 模糊{blur_kernel}, Canny({canny1},{canny2})")
            
            # 预处理
            processed = cv2.GaussianBlur(im_gray, blur_kernel, 0)
            
            # 可选的直方图均衡化
            processed = cv2.equalizeHist(processed)
            
            im_edge = cv2.Canny(processed, canny1, canny2)
            
            # 形态学操作清理边缘
            kernel = np.ones((2,2), np.uint8)
            im_edge = cv2.morphologyEx(im_edge, cv2.MORPH_CLOSE, kernel)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(im_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            board_candidates = self.find_board_candidates(contours, img.shape)
            results.extend([(candidate, desc) for candidate in board_candidates])
        
        if not results:
            print("未找到任何棋盘候选区域")
            return None, None
        
        # 选择最佳候选区域
        best_rect, best_desc = self.select_best_board(results, img.shape)
        print(f"选择了最佳候选区域: {best_desc}")
        
        return best_rect, im_gray
    
    def find_board_candidates(self, contours, img_shape):
        """寻找棋盘候选区域"""
        candidates = []
        img_area = img_shape[0] * img_shape[1]
        
        for contour in contours:
            # 多种逼近策略
            for epsilon_factor in [0.02, 0.05, 0.08, 0.12]:
                hull = cv2.convexHull(contour)
                epsilon = epsilon_factor * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    area = cv2.contourArea(approx)
                    area_ratio = area / img_area
                    
                    # 面积筛选
                    if self.min_board_area_ratio <= area_ratio <= self.max_board_area_ratio:
                        # 形状质量评估
                        quality_score = self.evaluate_board_quality(approx)
                        
                        # 标准化坐标
                        rect = self.normalize_rectangle(approx)
                        candidates.append((rect, area, quality_score))
        
        # 按质量评分排序
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [candidate[0] for candidate in candidates[:3]]  # 返回前3个候选
    
    def evaluate_board_quality(self, approx):
        """评估棋盘候选区域的质量"""
        points = approx.reshape(4, 2)
        
        # 计算四边长度
        sides = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            side_length = np.linalg.norm(p2 - p1)
            sides.append(side_length)
        
        sides = np.array(sides)
        
        # 评估指标
        # 1. 边长相似性 (围棋棋盘应该接近正方形)
        side_ratio_score = 1.0 - (np.std(sides) / np.mean(sides))
        
        # 2. 角度接近90度
        angles = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(abs(angle - np.pi/2))
        
        angle_score = 1.0 - (np.mean(angles) / (np.pi/4))
        
        # 3. 面积占比合理性
        area = cv2.contourArea(approx)
        area_score = min(1.0, area / 100000)  # 鼓励较大的区域
        
        # 综合评分
        total_score = (side_ratio_score * 0.4 + 
                      angle_score * 0.4 + 
                      area_score * 0.2)
        
        return max(0, total_score)
    
    def normalize_rectangle(self, approx):
        """标准化矩形坐标为左上、左下、右上、右下"""
        points = approx.reshape(4, 2)
        
        # 按x坐标排序
        points = points[np.argsort(points[:, 0])]
        
        # 左侧两点按y坐标排序
        left_points = points[:2]
        left_points = left_points[np.argsort(left_points[:, 1])]
        
        # 右侧两点按y坐标排序
        right_points = points[2:]
        right_points = right_points[np.argsort(right_points[:, 1])]
        
        # 返回 (左上, 左下, 右上, 右下)
        return (tuple(left_points[0]), tuple(left_points[1]), 
                tuple(right_points[0]), tuple(right_points[1]))
    
    def select_best_board(self, candidates, img_shape):
        """选择最佳棋盘候选区域"""
        if not candidates:
            return None, None
        
        # 简单策略：选择第一个（已按质量排序）
        return candidates[0]
    
    def adaptive_perspective_transform(self, img, rect, scale_factor):
        """自适应透视变换"""
        if rect is None:
            return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 将坐标缩放回原始尺寸
        lt, lb, rt, rb = rect
        if scale_factor != 1.0:
            lt = (int(lt[0] / scale_factor), int(lt[1] / scale_factor))
            lb = (int(lb[0] / scale_factor), int(lb[1] / scale_factor))
            rt = (int(rt[0] / scale_factor), int(rt[1] / scale_factor))
            rb = (int(rb[0] / scale_factor), int(rb[1] / scale_factor))
        
        print('棋盘坐标：')
        print(f'\t左上角：{lt}')
        print(f'\t左下角：{lb}')
        print(f'\t右上角：{rt}')
        print(f'\t右下角：{rb}')
        
        # 计算目标尺寸 - 根据检测到的棋盘大小动态调整
        board_width = max(np.linalg.norm(np.array(rt) - np.array(lt)),
                         np.linalg.norm(np.array(rb) - np.array(lb)))
        board_height = max(np.linalg.norm(np.array(lb) - np.array(lt)),
                          np.linalg.norm(np.array(rb) - np.array(rt)))
        
        # 选择合适的输出尺寸
        target_size = min(800, max(int(max(board_width, board_height)), 400))
        
        # 执行透视变换
        pts1 = np.float32([(10, 10), (10, target_size-10), 
                          (target_size-10, 10), (target_size-10, target_size-10)])
        pts2 = np.float32([lt, lb, rt, rb])
        
        m = cv2.getPerspectiveTransform(pts2, pts1)
        
        # 对原图执行透视变换
        img_transformed = cv2.warpPerspective(img, m, (target_size, target_size))
        gray_transformed = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2GRAY)
        
        return img_transformed, gray_transformed
    
    def enhanced_line_detection(self, gray_img):
        """增强的直线检测"""
        # 多种边缘检测策略
        edge_methods = []
        
        # 方法1：标准Canny
        blur1 = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges1 = cv2.Canny(blur1, 50, 150)
        edge_methods.append(("标准Canny", edges1))
        
        # 方法2：自适应阈值
        blur2 = cv2.GaussianBlur(gray_img, (3, 3), 0)
        edges2 = cv2.Canny(blur2, 30, 100)
        edge_methods.append(("敏感Canny", edges2))
        
        # 方法3：Sobel边缘检测
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        edges3 = np.uint8(sobel_combined / sobel_combined.max() * 255)
        edges3 = cv2.threshold(edges3, 50, 255, cv2.THRESH_BINARY)[1]
        edge_methods.append(("Sobel", edges3))
        
        all_lines = []
        
        # 对每种边缘检测方法应用霍夫变换
        for method_name, edges in edge_methods:
            # 多组霍夫参数
            hough_params = [
                # (threshold, minLineLength, maxLineGap)
                (60, 80, 15),
                (40, 60, 20),
                (80, 100, 10),
                (30, 40, 25),
            ]
            
            for threshold, minLineLength, maxLineGap in hough_params:
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                       threshold=threshold, 
                                       minLineLength=minLineLength, 
                                       maxLineGap=maxLineGap)
                
                if lines is not None:
                    print(f"{method_name} (T:{threshold},L:{minLineLength},G:{maxLineGap}): {len(lines)} 条线")
                    all_lines.extend(lines)
        
        if not all_lines:
            print("未检测到任何直线")
            return [], []
        
        print(f"总共检测到 {len(all_lines)} 条直线")
        
        # 过滤和分类直线
        return self.classify_and_filter_lines(all_lines)
    
    def classify_and_filter_lines(self, lines):
        """分类和过滤直线"""
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算角度和长度
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 只考虑足够长的线条
            if length > 30:  # 降低最小长度要求
                # 更宽松的角度判断
                if angle < 15 or angle > 165:  # 水平线
                    horizontal_lines.append(line[0])
                elif 75 < angle < 105:  # 垂直线
                    vertical_lines.append(line[0])
        
        print(f"分类后 - 水平线: {len(horizontal_lines)}, 垂直线: {len(vertical_lines)}")
        
        # 合并相近的线条
        merged_horizontal = self.merge_parallel_lines(horizontal_lines, True)
        merged_vertical = self.merge_parallel_lines(vertical_lines, False)
        
        print(f"合并后 - 水平线: {len(merged_horizontal)}, 垂直线: {len(merged_vertical)}")
        
        return merged_horizontal, merged_vertical
    
    def merge_parallel_lines(self, lines, is_horizontal):
        """改进的平行线合并算法"""
        if not lines:
            return []
        
        merged = []
        tolerance = 20  # 增加容差
        
        # 按位置排序
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
                    # 选择组中最长的线条，而不是平均
                    longest_line = max(current_group, 
                                     key=lambda l: math.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2))
                    merged.append(longest_line)
                current_group = [lines[i]]
        
        if current_group:
            longest_line = max(current_group, 
                             key=lambda l: math.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2))
            merged.append(longest_line)
        
        return merged
    
    def detect_board(self, img_path):
        """主检测函数"""
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print("无法读取图像文件")
            return None
        
        print(f"开始处理图像: {img_path}")
        
        # 预处理
        processed_img, original_img, scale_factor = self.preprocess_image(img)
        
        # 增强的棋盘检测
        rect, im_gray = self.enhanced_board_detection(processed_img)
        
        # 透视变换
        corrected_img, corrected_gray = self.adaptive_perspective_transform(
            original_img, rect, scale_factor)
        
        # 显示结果
        cv2.imshow("Original", cv2.resize(original_img, (600, 600)))
        if rect is not None:
            # 显示检测到的角点
            corner_img = processed_img.copy()
            for p in rect:
                cv2.circle(corner_img, p, 8, (0, 255, 0), -1)
            cv2.imshow("Detected Corners", cv2.resize(corner_img, (400, 400)))
        
        cv2.imshow("Corrected Board", corrected_img)
        
        # 直线检测
        h_lines, v_lines = self.enhanced_line_detection(corrected_gray)
        
        # 可视化检测结果
        if h_lines or v_lines:
            lines_img = corrected_img.copy()
            for line in h_lines:
                cv2.line(lines_img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
            for line in v_lines:
                cv2.line(lines_img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
            cv2.imshow("Detected Lines", lines_img)
        
        return corrected_img, corrected_gray, h_lines, v_lines

# 使用示例
def main():
    detector = RobustGoBoardDetector()
    
    # 测试不同的图像
    test_images = [
        #'../data/raw/IMG20160904165505-B.jpg',

        #'../data/raw/bd317d54.webp'
        #img = cv2.imread('../data/raw/IMG20171015161921.jpg')
        #img = cv2.imread('../data/raw/OGS3.jpeg')
        '../data/raw/IMG20160706171004.jpg'

        # 添加更多测试图像路径
    ]
    
    for img_path in test_images:
        print(f"\n{'='*50}")
        print(f"测试图像: {img_path}")
        print(f"{'='*50}")
        
        result = detector.detect_board(img_path)
        if result:
            corrected_img, corrected_gray, h_lines, v_lines = result
            print(f"检测完成 - 水平线: {len(h_lines)}, 垂直线: {len(v_lines)}")
        else:
            print("检测失败")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()